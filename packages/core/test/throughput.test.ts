import { test } from 'node:test'
import assert from 'node:assert/strict'
import { size } from '../src/plan.ts'
import { allReduceSeconds, causalAttentionFlops, estimateThroughput } from '../src/throughput.ts'
import { normalizeConfig } from '../src/model.ts'
import { getModelSnapshot, getGpu, defaultAssumptions } from '../src/data.ts'
import { activeParameters, parameterCounts } from '../src/weights.ts'

const spec = (id: string) => normalizeConfig(getModelSnapshot(id).config, { id })

test('decode is bandwidth-bound and matches the closed form', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const gpu = getGpu('h100-sxm-80')
  const t = estimateThroughput({
    model: m, gpu, parallel: { tp: 1, pp: 1 },
    workload: { concurrency: 1, avgSeqLen: 1, maxModelLen: 4096, chunkTokens: 1 },
    weightBytesPerDevice: 16_060_522_496, kvBytesPerTokenPerDevice: 131072,
    kvDtype: 'fp16', assumptions: defaultAssumptions(),
  })
  const p = parameterCounts(m)
  const streamed = 16_060_522_496 * (1 - p.embedding / p.total)
  const expected = (streamed + 131072) / (3352e9 * 0.65)
  assert.ok(Math.abs(t.decode.stepSeconds - expected) / expected < 1e-9)
  // ~6.9 GiB/step at 3.35 TB/s x 0.65 MBU -> ~145 tok/s single stream
  assert.ok(t.decode.tokensPerSecond > 100 && t.decode.tokensPerSecond < 200, `${t.decode.tokensPerSecond}`)
  assert.equal(t.bound, 'memory')
})

test('batching amortises the weight read, so tok/s rises sublinearly', () => {
  const r1 = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', context: 2048, concurrency: 1, avgSeqLen: 1024 })
  const r64 = size({ model: 'meta-llama/Llama-3.1-8B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', context: 2048, concurrency: 64, avgSeqLen: 1024 })
  assert.ok(r64.throughput.decode.tokensPerSecond > r1.throughput.decode.tokensPerSecond * 10)
  assert.ok(r64.throughput.decode.tokensPerSecond < r1.throughput.decode.tokensPerSecond * 64)
  assert.ok(r64.throughput.decode.itlMs > r1.throughput.decode.itlMs, 'per-token latency gets worse with batch')
})

test('MoE streams every expert but only computes top-k', () => {
  const m = spec('mistralai/Mixtral-8x7B-Instruct-v0.1')
  const p = parameterCounts(m)
  assert.ok(activeParameters(m) / p.total < 0.3)
  const r = size({ model: 'mistralai/Mixtral-8x7B-Instruct-v0.1', gpu: 'h100-sxm-80', engine: 'vllm', tp: 2, context: 4096, concurrency: 8, quant: 'fp8' })
  // decode traffic uses resident bytes (all experts), so it is far above active-param bytes
  assert.ok(r.throughput.decode.bytesPerStep > activeParameters(m))
})

test('sliding windows cap prefill attention FLOPs', () => {
  const gemma = spec('google/gemma-3-27b-it')
  const at8k = causalAttentionFlops(gemma, 8192)
  const at16k = causalAttentionFlops(gemma, 16384)
  assert.ok(at16k / at8k < 3.5, 'windowed layers grow linearly, not quadratically')
  const llama = spec('meta-llama/Llama-3.1-8B-Instruct')
  assert.equal(causalAttentionFlops(llama, 4096), 32 * 2 * 4096 * 4096 * 4096)
  assert.equal(causalAttentionFlops(llama, 8192) / causalAttentionFlops(llama, 4096), 4)
})

test('all-reduce cost is the ring formula and vanishes at TP=1', () => {
  assert.equal(allReduceSeconds(1e6, 1, 900e9, 0.7), 0)
  assert.equal(allReduceSeconds(1e6, 2, 900e9, 0.7), (2 * 1 / 2) * (1e6 / (900e9 * 0.7)))
  assert.ok(allReduceSeconds(1e6, 8, 900e9, 0.7) > allReduceSeconds(1e6, 2, 900e9, 0.7))
})

test('TP speeds decode up but never linearly, because of comms', () => {
  const q = (tp: number) => size({ model: 'meta-llama/Llama-3.1-70B-Instruct', gpu: 'h100-sxm-80', engine: 'vllm', tp, context: 4096, concurrency: 16, avgSeqLen: 2048 }).throughput.decode.stepSeconds
  assert.ok(q(4) < q(2))
  assert.ok(q(8) < q(4))
  assert.ok(q(8) > q(2) / 4, 'TP scaling must be sublinear')
})

test('throughput is labelled a roofline, not a measurement', () => {
  const r = size({ model: 'Qwen/Qwen2.5-7B-Instruct', gpu: 'a100-sxm-80', engine: 'vllm', context: 4096, concurrency: 8 })
  assert.equal(r.throughput.method, 'roofline')
  assert.equal(r.label, 'predicted')
})
