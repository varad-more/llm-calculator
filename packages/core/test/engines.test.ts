import { test } from 'node:test'
import assert from 'node:assert/strict'
import { size, ENGINES } from '../src/plan.ts'
import { capturedGraphSizes, prefillActivationBytes, logitsBytes, overheadBytes } from '../src/overhead.ts'
import { normalizeConfig } from '../src/model.ts'
import { getModelSnapshot, defaultAssumptions, resolveAssumptions } from '../src/data.ts'
import { emitVllmCommand } from '../src/engines/vllm.ts'

const spec = (id: string) => normalizeConfig(getModelSnapshot(id).config, { id })
const GiB = 2 ** 30

test('prefill activation scales linearly with the chunk, and by hand', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const i = {
    model: m, parallel: { tp: 1, pp: 1 }, actDtypeBytes: 2, cudaGraphs: true,
    assumptions: defaultAssumptions(),
    workload: { concurrency: 1, avgSeqLen: 1024, maxModelLen: 1024, chunkTokens: 2048 },
  }
  // 2048 x 4096 x 2 x 4  +  2 x 2048 x 14336 x 2
  assert.equal(prefillActivationBytes(i), 2048 * 4096 * 2 * 4 + 2 * 2048 * 14336 * 2)
  assert.equal(prefillActivationBytes({ ...i, workload: { ...i.workload, chunkTokens: 4096 } }), prefillActivationBytes(i) * 2)
})

test('logits are fp32 and explode when an engine keeps every prefill position', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const base = {
    model: m, parallel: { tp: 1, pp: 1 }, actDtypeBytes: 2, cudaGraphs: true,
    assumptions: defaultAssumptions(),
    workload: { concurrency: 8, avgSeqLen: 1024, maxModelLen: 1024, chunkTokens: 8192 },
  }
  assert.equal(logitsBytes(base), 8 * 128256 * 4)
  assert.equal(logitsBytes({ ...base, logitsForAllPositions: true }), 8192 * 128256 * 4)
  const o = overheadBytes({ ...base, logitsForAllPositions: true }, 16e9)
  assert.ok(o.warnings.some((w) => w.code === 'logits_all_positions'))
})

test('CUDA graph capture count follows max_num_seqs', () => {
  assert.equal(capturedGraphSizes(1), 3)
  assert.equal(capturedGraphSizes(256), 35)
  assert.equal(capturedGraphSizes(4096), 67) // capped at 512
})

test('vLLM: 8B on one H100 fits with a large KV pool', () => {
  const r = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', context: 32768, concurrency: 64, avgSeqLen: 4096 })
  assert.equal(r.plan.fits, true)
  assert.ok(r.plan.weightBytesPerDevice / GiB > 14.9 && r.plan.weightBytesPerDevice / GiB < 15.1)
  assert.ok(r.plan.maxTokens > 300_000, `KV pool was only ${r.plan.maxTokens} tokens`)
  assert.equal(r.plan.maxTokens, r.plan.numBlocks * 16)
  assert.equal(r.label, 'predicted') // no validation case for this triple yet
})

test('vLLM: 70B on one H100 does not fit, and the reason is the weights', () => {
  const r = size({ model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', context: 8192, concurrency: 8 })
  assert.equal(r.plan.fits, false)
  assert.ok(r.plan.availableKvBytes < 0)
  assert.ok(r.plan.warnings.some((w) => w.code === 'no_kv_headroom'))
  assert.equal(r.plan.autofix!.maxModelLen, 0)
})

test('vLLM: an over-long context is auto-fixed to the largest that fits', () => {
  const r = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'l4-24', engine: 'vllm', context: 131072, concurrency: 1 })
  assert.equal(r.plan.fits, false)
  const fix = r.plan.autofix!
  assert.ok(fix.maxModelLen > 0 && fix.maxModelLen < 131072)
  assert.equal(fix.maxModelLen % 16, 0)
  // the fix must itself fit
  const refit = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'l4-24', engine: 'vllm', context: fix.maxModelLen, concurrency: 1 })
  assert.equal(refit.plan.fits, true, `autofix ${fix.maxModelLen} still does not fit`)
  assert.match(r.command, new RegExp(`--max-model-len ${fix.maxModelLen}`))
})

test('emitted vLLM flags are the real ones', () => {
  const r = size({
    model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm',
    tp: 4, context: 32768, concurrency: 64, avgSeqLen: 4096, kvDtype: 'fp8', memoryUtilization: 0.92,
    prefixCache: { enabled: true, hitRate: 0.5, sharedPrefixTokens: 1024 },
  })
  assert.deepEqual(r.flags, [
    '--tensor-parallel-size 4', '--max-model-len 32768', '--max-num-seqs 64',
    '--gpu-memory-utilization 0.92', '--max-num-batched-tokens 8192',
    '--kv-cache-dtype fp8', '--enable-prefix-caching',
  ])
  assert.ok(r.command.startsWith('vllm serve meta-llama/Llama-3.1-70B-Instruct'))
  assert.equal(emitVllmCommand(r.plan), r.command)
})

test('quantization reaches the flags', () => {
  const awq = size({ model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', quant: 'awq-int4', context: 8192, concurrency: 8 })
  assert.ok(awq.flags.includes('--quantization awq'))
  assert.equal(awq.plan.fits, true, '70B at int4 fits on one H100')
})

test('SGLang solves --mem-fraction-static and warns when it is too high', () => {
  const r = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'sglang', context: 8192, concurrency: 32, avgSeqLen: 2048 })
  const frac = r.flags.find((f) => f.startsWith('--mem-fraction-static'))!
  const value = Number(frac.split(' ')[1])
  assert.ok(value > 0.2 && value < 0.9, frac)
  assert.ok(r.flags.some((f) => f.startsWith('--tp 1')))
  const tight = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'sglang', context: 8192, concurrency: 32, memoryUtilization: 0.999 })
  assert.ok(tight.plan.warnings.some((w) => w.code === 'mem_fraction_static_too_high'))
})

test('SGLang turns prefix caching on by default (RadixAttention)', () => {
  const r = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'sglang', context: 4096, concurrency: 8 })
  assert.equal(r.plan.input.workload.prefixCache!.enabled, true)
  assert.ok(!r.flags.includes('--disable-radix-cache'))
})

test('TRT-LLM says its activation budget is frozen at build time', () => {
  const r = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'trtllm', context: 8192, concurrency: 16 })
  assert.ok(r.plan.warnings.some((w) => w.code === 'trtllm_build_time_budget'))
  assert.ok(r.flags.some((f) => f.startsWith('--max_batch_size')))
  assert.ok(r.flags.some((f) => f.startsWith('--kv_cache_free_gpu_memory_fraction')))
})

test('llama.cpp wants GGUF and says so when it does not get it', () => {
  const bad = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'rtx4090-24', engine: 'llamacpp', context: 4096, concurrency: 4 })
  assert.ok(bad.plan.warnings.some((w) => w.code === 'llamacpp_needs_gguf'))
  const good = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'rtx4090-24', engine: 'llamacpp', quant: 'gguf:Q4_K_M', context: 4096, concurrency: 4 })
  assert.ok(!good.plan.warnings.some((w) => w.code === 'llamacpp_needs_gguf'))
  assert.ok(good.plan.fits)
  assert.ok(good.flags.includes('-ngl 99'))
})

test('every engine adapter is registered and produces flags', () => {
  for (const name of Object.keys(ENGINES) as (keyof typeof ENGINES)[]) {
    const r = size({ model: 'Qwen/Qwen2.5-7B-Instruct', gpu: 'a100-sxm-80', engine: name, context: 4096, concurrency: 8, quant: name === 'llamacpp' ? 'gguf:Q4_K_M' : 'bf16' })
    assert.ok(r.flags.length > 0, name)
    assert.ok(r.command.length > 20, name)
  }
})

test('assumption overrides move the answer', () => {
  const base = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', context: 8192, concurrency: 8 })
  const roomier = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', context: 8192, concurrency: 8, assume: { cuda_context_bytes: 0 } })
  assert.ok(roomier.plan.availableKvBytes > base.plan.availableKvBytes)
  assert.equal(roomier.plan.availableKvBytes - base.plan.availableKvBytes, resolveAssumptions().cuda_context_bytes.value)
})
