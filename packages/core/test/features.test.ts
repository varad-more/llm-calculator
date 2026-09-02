import { test } from 'node:test'
import assert from 'node:assert/strict'
import { size } from '../src/plan.ts'
import { loraSizing } from '../src/lora.ts'
import { speculativeThroughput, emitSpeculativeFlags } from '../src/speculative.ts'
import { planDisaggregated } from '../src/disagg.ts'
import { reverseLookup, largestContextPerModel } from '../src/reverse.ts'
import { normalizeConfig } from '../src/model.ts'
import { getModelSnapshot } from '../src/data.ts'

const spec = (id: string) => normalizeConfig(getModelSnapshot(id).config, { id })
const GiB = 2 ** 30

test('LoRA adapters cost r(m+n) per projection, not r*m*n', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const r = loraSizing(m, { maxLoras: 1, maxLoraRank: 16 }, { tp: 1, pp: 1 })
  const rank = 16, d = 4096, nh = 32, nkv = 8, dh = 128, dff = 14336
  const perLayer = rank * ((d + nh * dh) + 2 * (d + nkv * dh)) + rank * (nh * dh + d)
    + rank * 2 * (d + dff) + rank * (dff + d)
  assert.equal(r.paramsPerAdapter, 32 * perLayer)
  assert.equal(r.bytesPerAdapter, 32 * perLayer * 2)
  assert.ok(r.bytesPerAdapter < 0.1 * GiB, 'a rank-16 adapter is tens of MiB, not GiB')
})

test('LoRA slots scale with max-loras and shard with TP', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const one = loraSizing(m, { maxLoras: 1, maxLoraRank: 32 }, { tp: 1, pp: 1 })
  const eight = loraSizing(m, { maxLoras: 8, maxLoraRank: 32 }, { tp: 1, pp: 1 })
  assert.equal(eight.totalBytesPerDevice, one.totalBytesPerDevice * 8)
  const tp4 = loraSizing(m, { maxLoras: 8, maxLoraRank: 32 }, { tp: 4, pp: 1 })
  assert.equal(tp4.totalBytesPerDevice, eight.totalBytesPerDevice / 4)
  assert.deepEqual(eight.flags, ['--enable-lora', '--max-loras 8', '--max-lora-rank 32'])
})

test('speculative yield is the truncated geometric series', () => {
  const base = size({ model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', tp: 4, context: 4096, concurrency: 1 }).throughput
  const s = speculativeThroughput(base, { numSpeculativeTokens: 4, acceptanceRate: 0.8, draftStepSeconds: base.decode.stepSeconds / 20 })
  // (1 - 0.8^5) / (1 - 0.8) = 3.3616
  assert.ok(Math.abs(s.tokensPerCycle - 3.3616) < 1e-9)
  assert.ok(s.speedup > 1 && s.worthwhile, `speedup ${s.speedup}`)

  const perfect = speculativeThroughput(base, { numSpeculativeTokens: 4, acceptanceRate: 1, draftStepSeconds: 0 })
  assert.equal(perfect.tokensPerCycle, 5)
  const useless = speculativeThroughput(base, { numSpeculativeTokens: 4, acceptanceRate: 0, draftStepSeconds: 0 })
  assert.equal(useless.tokensPerCycle, 1)
})

test('a slow or inaccurate draft is reported as a net loss, not hidden', () => {
  const base = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', context: 4096, concurrency: 1 }).throughput
  const bad = speculativeThroughput(base, { numSpeculativeTokens: 8, acceptanceRate: 0.2, draftStepSeconds: base.decode.stepSeconds * 0.5 })
  assert.equal(bad.worthwhile, false)
  assert.ok(bad.speedup < 1)
  assert.match(emitSpeculativeFlags('meta-llama/Llama-3.2-1B-Instruct', { numSpeculativeTokens: 4, acceptanceRate: 0.8, draftStepSeconds: 0 })[0]!, /--speculative-config/)
})

test('disaggregation sizes both pools and finds the bottleneck', () => {
  const d = planDisaggregated({
    model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm',
    quant: 'fp8', kvDtype: 'fp8', context: 32768,
    prefill: { gpus: 8, tp: 4 }, decode: { gpus: 8, tp: 4 },
    prefillConcurrency: 8, decodeConcurrency: 64,
    promptTokens: 8192, outputTokens: 512,
    transferBytesPerSec: 900e9, // NVLink
  })
  assert.ok(d.prefill.plan.fits && d.decode.plan.fits)
  assert.ok(d.transfer.bytesPerRequest > 0)
  assert.ok(['prefill', 'decode', 'transfer'].includes(d.balance.bottleneck))
  assert.ok(d.balance.decodePerPrefill > 0)
  // the pools are sized independently: different TP, different concurrency
  assert.equal(d.prefill.plan.input.parallel.tp, 4)
  assert.equal(d.decode.plan.input.workload.concurrency, 64)
  assert.equal(d.prefill.plan.input.workload.concurrency, 8)
})

test('a slow fabric makes KV transfer the bottleneck, and says so', () => {
  const slow = planDisaggregated({
    model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm',
    kvDtype: 'fp16', quant: 'fp8', context: 32768,
    prefill: { gpus: 64, tp: 4 }, decode: { gpus: 64, tp: 4 },
    prefillConcurrency: 4, decodeConcurrency: 32,
    promptTokens: 32768, outputTokens: 256,
    transferBytesPerSec: 12.5e9, // 100G Ethernet
  })
  assert.equal(slow.balance.bottleneck, 'transfer')
  assert.ok(slow.warnings.some((w) => w.code === 'kv_transfer_bound'))
  // fp8 KV halves the wire cost
  const fast = planDisaggregated({
    model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm',
    kvDtype: 'fp8', quant: 'fp8', context: 32768,
    prefill: { gpus: 64, tp: 4 }, decode: { gpus: 64, tp: 4 },
    prefillConcurrency: 4, decodeConcurrency: 32,
    promptTokens: 32768, outputTokens: 256,
    transferBytesPerSec: 12.5e9,
  })
  assert.ok(Math.abs(fast.transfer.bytesPerRequest * 2 - slow.transfer.bytesPerRequest) < 1)
})

test('MLA moves far less KV across the wire than GQA, per layer and in total', () => {
  const shape = {
    gpu: 'h200-sxm-141', engine: 'vllm' as const, quant: 'fp8' as const, kvDtype: 'fp16' as const,
    context: 16384, prefill: { gpus: 8, tp: 8 }, decode: { gpus: 8, tp: 8 },
    prefillConcurrency: 4, decodeConcurrency: 16, promptTokens: 16384, outputTokens: 256,
    transferBytesPerSec: 50e9,
  }
  const llama = planDisaggregated({ ...shape, model: 'meta-llama/Llama-3.1-70B-Instruct' })
  const ds = planDisaggregated({ ...shape, model: 'deepseek-ai/DeepSeek-V3' })
  // Per layer: Llama's GQA-8 caches 2*8*128*2 = 4096 B/token; DeepSeek's MLA caches
  // (512+64)*2 = 1152 B/token. 3.56x, despite DeepSeek being 10x the model.
  const perLayerRatio = (llama.transfer.bytesPerRequest / 80) / (ds.transfer.bytesPerRequest / 61)
  assert.ok(Math.abs(perLayerRatio - 4096 / 1152) < 1e-9, `got ${perLayerRatio}x`)
  assert.ok(ds.transfer.bytesPerRequest < llama.transfer.bytesPerRequest,
    'a 671B MLA model moves less KV per request than a 70B GQA model')
  // and the MLA latent is replicated, so TP must not multiply its transfer size
  const dsTp1 = planDisaggregated({ ...shape, model: 'deepseek-ai/DeepSeek-V3', prefill: { gpus: 1, tp: 1 }, decode: { gpus: 1, tp: 1 } })
  assert.equal(dsTp1.transfer.bytesPerRequest, ds.transfer.bytesPerRequest)
})

test('reverse lookup only returns configurations that actually fit', () => {
  const rows = reverseLookup({ gpu: 'a10g-24', gpuCount: 1, contexts: [4096, 32768], limit: 40 })
  assert.ok(rows.length > 0)
  for (const r of rows) {
    assert.equal(r.fits, true)
    assert.ok(r.freeBytes >= 0)
    const refit = size({ model: r.model, gpu: 'a10g-24', engine: 'vllm', tp: r.tp, quant: r.quant, kvDtype: r.kvDtype, context: r.context, concurrency: 1 })
    assert.equal(refit.plan.fits, true, `${r.model} ${r.quant} ${r.context} claimed to fit`)
  }
  // a 24 GiB card cannot serve a 405B model at any quant in this set
  assert.ok(!rows.some((r) => r.model.includes('405B')))
})

test('more GPUs unlock strictly more configurations', () => {
  const one = reverseLookup({ gpu: 'h100-sxm-80', gpuCount: 1, contexts: [8192] }).length
  const eight = reverseLookup({ gpu: 'h100-sxm-80', gpuCount: 8, contexts: [8192] }).length
  assert.ok(eight > one, `${eight} vs ${one}`)
})

test('largest-context-per-model gives one row per model/quant', () => {
  const rows = largestContextPerModel({ gpu: 'h100-sxm-80', gpuCount: 2, quants: ['fp8'], kvDtypes: ['fp8'] })
  const keys = rows.map((r) => `${r.model}|${r.quant}`)
  assert.equal(new Set(keys).size, keys.length)
  assert.ok(rows.every((r) => r.command.startsWith('vllm serve')))
})
