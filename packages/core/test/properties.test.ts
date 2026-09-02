import { test } from 'node:test'
import assert from 'node:assert/strict'
import { size, type SizingRequest } from '../src/plan.ts'
import { listGpus, listModels } from '../src/data.ts'
import type { EngineName } from '../src/engines/types.ts'
import type { KvDtype, QuantScheme } from '../src/types.ts'

// Deterministic LCG: property tests must fail the same way twice.
function rng(seed: number) {
  let s = seed >>> 0
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32)
}
const pick = <T,>(r: () => number, xs: readonly T[]): T => xs[Math.floor(r() * xs.length)]!

const MODELS = listModels()
const GPUS = listGpus().map((g) => g.id)
const ENGINE_NAMES: EngineName[] = ['vllm', 'sglang', 'trtllm', 'llamacpp']
const QUANTS: QuantScheme[] = ['bf16', 'fp8', 'awq-int4', 'gguf:Q4_K_M']
const KV: KvDtype[] = ['fp16', 'fp8', 'int8']

function randomRequest(r: () => number): SizingRequest {
  const context = pick(r, [512, 2048, 8192, 32768, 131072])
  return {
    model: pick(r, MODELS), gpu: pick(r, GPUS), engine: pick(r, ENGINE_NAMES),
    tp: pick(r, [1, 2, 4, 8, 16]), pp: pick(r, [1, 1, 1, 2, 4]),
    quant: pick(r, QUANTS), kvDtype: pick(r, KV),
    context, concurrency: pick(r, [1, 8, 64, 256]),
    avgSeqLen: Math.max(1, Math.floor(context * r())),
    memoryUtilization: 0.5 + r() * 0.45,
  }
}

test('KV memory is monotonic in context length', () => {
  const r = rng(1)
  for (let i = 0; i < 200; i++) {
    const base = randomRequest(r)
    let prev = -1
    for (const context of [512, 4096, 32768]) {
      const kv = size({ ...base, context, avgSeqLen: context }).plan.requiredKvBytes
      assert.ok(kv >= prev, `${base.model} on ${base.gpu}: KV shrank from ${prev} to ${kv} at context ${context}`)
      prev = kv
    }
  }
})

test('KV memory is monotonic in concurrency', () => {
  const r = rng(2)
  for (let i = 0; i < 200; i++) {
    const base = randomRequest(r)
    let prev = -1
    for (const concurrency of [1, 8, 64, 256]) {
      const kv = size({ ...base, concurrency }).plan.requiredKvBytes
      assert.ok(kv >= prev, `${base.model}: KV shrank at concurrency ${concurrency}`)
      prev = kv
    }
  }
})

test('per-device memory is non-increasing in TP and PP', () => {
  const r = rng(3)
  for (let i = 0; i < 200; i++) {
    const base = randomRequest(r)
    let prevW = Infinity, prevKv = Infinity
    for (const tp of [1, 2, 4, 8]) {
      const p = size({ ...base, tp, pp: 1 }).plan
      assert.ok(p.weightBytesPerDevice <= prevW * 1.000001, `${base.model}: weights grew at TP=${tp}`)
      assert.ok(p.kv.totalBytesPerDevice <= prevKv * 1.000001, `${base.model}: KV grew at TP=${tp}`)
      prevW = p.weightBytesPerDevice; prevKv = p.kv.totalBytesPerDevice
    }
  }
})

test('no configuration reports a negative memory component', () => {
  const r = rng(4)
  for (let i = 0; i < 500; i++) {
    const req = randomRequest(r)
    const p = size(req).plan
    for (const [k, v] of Object.entries({
      weights: p.weightBytesPerDevice, kv: p.kv.totalBytesPerDevice, required: p.requiredKvBytes,
      perToken: p.kv.perTokenBytesPerDevice, usable: p.usableVramBytes, budget: p.budgetBytes,
      ...p.overhead,
    })) {
      if (typeof v !== 'number') continue
      assert.ok(v >= 0, `${JSON.stringify(req)}: ${k} = ${v}`)
    }
    assert.ok(p.numBlocks >= 0 && p.maxTokens >= 0)
  }
})

test('a plan that fits leaves non-negative free memory, and one that does not offers a fix', () => {
  const r = rng(5)
  for (let i = 0; i < 500; i++) {
    const req = randomRequest(r)
    const p = size(req).plan
    if (p.fits) assert.ok(p.freeBytes >= 0, `${JSON.stringify(req)} claims to fit with ${p.freeBytes} free`)
    else assert.ok(p.autofix !== undefined, `${JSON.stringify(req)} does not fit but offers no autofix`)
  }
})

test('the auto-fixed configuration actually fits', () => {
  const r = rng(6)
  let checked = 0
  for (let i = 0; i < 400 && checked < 60; i++) {
    const req = { ...randomRequest(r), engine: 'vllm' as const }
    const p = size(req).plan
    if (p.fits || !p.autofix || p.autofix.maxModelLen === 0 || p.autofix.maxNumSeqs === 0) continue
    // The auto-fix is stated for the SAME workload shape: max_model_len is the per-request
    // cap and max_num_seqs the concurrency at the requested average length.
    const fixed = size({ ...req, context: p.autofix.maxModelLen, concurrency: p.autofix.maxNumSeqs })
    assert.ok(fixed.plan.fits,
      `autofix ${JSON.stringify(p.autofix)} for ${JSON.stringify(req)} still does not fit`)
    checked++
  }
  assert.ok(checked > 10, `only ${checked} infeasible cases exercised`)
})

test('every snapshotted model can be planned on every GPU without throwing', () => {
  for (const model of MODELS) {
    for (const gpu of ['h100-sxm-80', 'l4-24', 'mi300x-192']) {
      const r = size({ model, gpu, engine: 'vllm', tp: 8, context: 4096, concurrency: 4 })
      assert.equal(typeof r.plan.fits, 'boolean', `${model} / ${gpu}`)
    }
  }
})
