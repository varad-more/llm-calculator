import { assume } from './data.ts'
import { kvDtypeBytes } from './kv.ts'
import { activeParameters, parameterCounts } from './weights.ts'
import type { Assumptions, GpuSpec, KvDtype, ModelSpec, Parallelism, Workload } from './types.ts'

export interface ThroughputEstimate {
  /** Always 'roofline': this is a bandwidth/FLOP bound, not a measurement. */
  method: 'roofline'
  decode: { stepSeconds: number; itlMs: number; tokensPerSecond: number; commSeconds: number; bytesPerStep: number }
  prefill: { flops: number; ttftSeconds: number; tokensPerSecond: number; commSeconds: number }
  bound: 'memory' | 'compute'
}

/**
 * Cost of one tensor-parallel all-reduce of `bytes` over `n` ranks (ring algorithm):
 * \\[ t = \\frac{2(n-1)}{n} \\cdot \\frac{\\text{bytes}}{BW_{link}\\, \\eta} \\]
 * @see docs/MATH.md#tp-communication
 */
export function allReduceSeconds(bytes: number, n: number, linkBytesPerSec: number, efficiency: number): number {
  if (n <= 1) return 0
  return ((2 * (n - 1)) / n) * (bytes / (linkBytesPerSec * efficiency))
}

/**
 * Causal self-attention FLOPs for a sequence of \\(s\\) tokens, summed over layers:
 * \\[ F_{attn} = \\sum_{\\ell} 2\\, s\\, \\min(s, W_\\ell)\\, n_h d_h \\]
 * The \\(2\\) counts \\(QK^\\top\\) and \\(PV\\) at \\(2s^2 n_h d_h\\) each, halved by causality.
 * A sliding-window layer replaces \\(s^2\\) with \\(s \\cdot W\\).
 * @see docs/MATH.md#prefill-flops
 */
export function causalAttentionFlops(model: ModelSpec, s: number): number {
  const width = model.numAttentionHeads * model.headDim
  let f = 0
  for (const layer of model.layers) {
    if (layer.kind === 'mamba' || layer.kind === 'linear') continue
    f += 2 * s * Math.min(s, layer.windowSize ?? s) * width
  }
  return f
}

/**
 * Roofline throughput. **Predicted, not measured** — it is an upper bound scaled by the
 * MBU/MFU assumptions, and it models no scheduler effects, no queueing and no batching jitter.
 *
 * Decode is memory-bound; each step streams the weights plus the whole KV cache of the batch:
 * \\[ t_{step} = \\frac{B_{w}^{dev} + b_{tok} \\sum_i \\ell_i}{BW \\cdot \\text{MBU}} + t_{comm} ,
 *    \\qquad \\text{tok/s} = S / t_{step} \\]
 * Prefill is compute-bound:
 * \\[ \\text{TTFT} = \\frac{2 P_{active} C + F_{attn}(C)}{\\text{TFLOPS}_{dense} \\cdot \\text{MFU}} + t_{comm} \\]
 * MoE decode streams **all** expert bytes (they are resident and gathered), while its FLOPs
 * count only the top-\\(k\\) experts — the two must not use the same parameter number.
 *
 * @see docs/MATH.md#throughput
 */
export function estimateThroughput(a: {
  model: ModelSpec
  gpu: GpuSpec
  parallel: Parallelism
  workload: Workload
  weightBytesPerDevice: number
  kvBytesPerTokenPerDevice: number
  kvDtype: KvDtype
  computeDtype?: 'bf16' | 'fp16' | 'fp8' | 'int8'
  assumptions: Assumptions
}): ThroughputEstimate {
  const { model, gpu, workload } = a
  const tp = Math.max(1, a.parallel.tp)
  const mbu = assume(a.assumptions, 'mbu_decode')
  const mfu = assume(a.assumptions, 'mfu_prefill')
  const eta = assume(a.assumptions, 'interconnect_efficiency')
  const link = gpu.interconnect.bidirectionalBytesPerSec

  // The embedding table is gathered per token, not streamed like a weight matrix.
  const params = parameterCounts(model)
  const embedFraction = params.total > 0 ? params.embedding / params.total : 0
  const streamedWeightBytes = a.weightBytesPerDevice * (1 - embedFraction)

  const batchTokens = workload.concurrency * workload.avgSeqLen
  const kvReadBytes = a.kvBytesPerTokenPerDevice * batchTokens
  const bytesPerStep = streamedWeightBytes + kvReadBytes
  const decodeComm = model.numLayers *
    allReduceSeconds(workload.concurrency * model.hiddenSize * 2, tp, link, eta)
  const stepSeconds = bytesPerStep / (gpu.memBandwidthBytesPerSec * mbu) + decodeComm

  const dtype = a.computeDtype ?? 'bf16'
  const tflops = (gpu.tflopsDense[dtype] ?? gpu.tflopsDense.bf16 ?? gpu.tflopsDense.fp16 ?? 0) * 1e12
  const C = Math.max(1, workload.chunkTokens)
  const flops = 2 * activeParameters(model) * C + causalAttentionFlops(model, C)
  const prefillComm = model.numLayers * allReduceSeconds(C * model.hiddenSize * 2, tp, link, eta)
  const computeSeconds = tflops > 0 ? flops / (tflops * mfu * tp) : Number.POSITIVE_INFINITY
  const ttftSeconds = computeSeconds + prefillComm

  return {
    method: 'roofline',
    decode: {
      stepSeconds, itlMs: stepSeconds * 1000,
      tokensPerSecond: workload.concurrency / stepSeconds,
      commSeconds: decodeComm, bytesPerStep,
    },
    prefill: { flops, ttftSeconds, tokensPerSecond: C / ttftSeconds, commSeconds: prefillComm },
    bound: bytesPerStep / (gpu.memBandwidthBytesPerSec * mbu) > computeSeconds ? 'memory' : 'compute',
  }
}

/** KV bytes read per decode step, per device. @see docs/MATH.md#throughput */
export function decodeKvReadBytes(model: ModelSpec, workload: Workload, kvDtype: KvDtype, shards: number): number {
  const s = kvDtypeBytes(kvDtype)
  let perToken = 0
  for (const layer of model.layers) {
    if (layer.kind === 'mla') perToken += (model.mla!.kvLoraRank + model.mla!.qkRopeHeadDim) * s
    else if (layer.kind !== 'mamba' && layer.kind !== 'linear') perToken += 2 * model.numKeyValueHeads * model.headDim * s
  }
  return (perToken / shards) * workload.concurrency * workload.avgSeqLen
}
