import { assume } from './data.ts'
import type { Assumptions, ModelSpec, Warning, Workload } from './types.ts'

export interface OverheadInput {
  model: ModelSpec
  workload: Workload
  parallel: { tp: number; pp: number }
  /** Activation dtype width in bytes (2 for bf16/fp16 compute). */
  actDtypeBytes: number
  /** vLLM/SGLang capture CUDA graphs unless run with enforce_eager. */
  cudaGraphs: boolean
  /** True for engines that compute logits for every prefill position, not just the sampled one. */
  logitsForAllPositions?: boolean
  assumptions: Assumptions
}

export interface OverheadBreakdown {
  cudaContextBytes: number
  commBytes: number
  graphBytes: number
  activationBytes: number
  logitsBytes: number
  fragmentationBytes: number
  /** Everything except activation+logits — what vLLM logs as "non-torch memory". */
  nonTorchBytes: number
  totalBytes: number
  warnings: Warning[]
}

/**
 * Number of batch sizes vLLM captures CUDA graphs for. The default list is
 * \\(\\{1,2,4\\} \\cup \\{8,16,\\dots\\}\\) truncated at `max_num_seqs`.
 * @see docs/MATH.md#cuda-graphs
 */
export function capturedGraphSizes(maxNumSeqs: number): number {
  return 3 + Math.floor(Math.min(maxNumSeqs, 512) / 8)
}

/**
 * Peak activation memory for one prefill chunk of \\(C\\) tokens.
 *
 * \\[ B_{act} = C\\, d\\, s\\, k \;+\; 2\\, C\\, d_{ff}\\, s \\]
 *
 * \\(k\\) live hidden-sized buffers (residual stream, QKV output, attention output, one
 * transient) plus the two gated-MLP intermediates. FlashAttention keeps the \\(S \\times S\\)
 * score matrix in SRAM, so there is **no** quadratic term. MoE layers use the expert
 * intermediate width, which is what actually gets materialised per token.
 *
 * @see docs/MATH.md#activation-peak
 */
export function prefillActivationBytes(i: OverheadInput): number {
  const { model, assumptions: a } = i
  const k = assume(a, 'prefill_activation_multiplier')
  const C = Math.max(1, i.workload.chunkTokens)
  const dff = model.moe && model.layers.some((l) => l.mlp === 'moe')
    ? model.moe.expertIntermediate * Math.min(model.moe.topK, model.moe.numExperts)
    : model.intermediateSize
  const tp = Math.max(1, i.parallel.tp)
  return (C * model.hiddenSize * i.actDtypeBytes * k + 2 * C * dff * i.actDtypeBytes) / tp
}

/**
 * Sampling logits, upcast to fp32: \\( B_{logits} = n\\, V\\, 4 \\).
 *
 * \\(n\\) is normally the number of sequences being sampled, but an engine that keeps
 * logits for every prefill position pays \\(n = C\\), which for a 128k-vocab model at
 * a 8192-token chunk is 4 GiB by itself.
 *
 * @see docs/MATH.md#logits
 */
export function logitsBytes(i: OverheadInput): number {
  const n = i.logitsForAllPositions ? Math.max(i.workload.chunkTokens, i.workload.concurrency) : i.workload.concurrency
  return n * i.model.vocabSize * 4
}

/**
 * Everything that is neither weights nor KV: CUDA context, NCCL buffers, captured
 * CUDA graphs, the prefill activation peak, sampling logits and allocator slack.
 *
 * \\[ B_{overhead} = B_{ctx} + [TP>1]\\,B_{nccl} + B_{graph} + B_{act} + B_{logits} + f\\,(B_{w}+B_{act}) \\]
 *
 * Every constant here comes from `data/assumptions.json` and is user-overridable.
 *
 * @see docs/MATH.md#overhead
 */
export function overheadBytes(i: OverheadInput, weightBytesPerDevice: number): OverheadBreakdown {
  const a = i.assumptions
  const tp = Math.max(1, i.parallel.tp)
  const warnings: Warning[] = []

  const cudaContextBytes = assume(a, 'cuda_context_bytes')
  const commBytes = tp > 1 || i.parallel.pp > 1 ? assume(a, 'nccl_buffer_bytes_per_rank') : 0
  const graphBytes = i.cudaGraphs
    ? assume(a, 'cudagraph_base_bytes') +
      capturedGraphSizes(i.workload.concurrency) * assume(a, 'cudagraph_bytes_per_captured_size')
    : 0
  const activationBytes = prefillActivationBytes(i)
  const logits = logitsBytes(i)
  const fragmentationBytes = assume(a, 'allocator_fragmentation_fraction') * (weightBytesPerDevice + activationBytes)

  if (i.logitsForAllPositions && logits > 1 << 30) {
    warnings.push({
      code: 'logits_all_positions',
      message: `This engine materialises fp32 logits for every prefill position: ${(logits / 2 ** 30).toFixed(1)} GiB at chunk=${i.workload.chunkTokens}, vocab=${i.model.vocabSize}. Lower --max-num-batched-tokens.`,
    })
  }

  const nonTorchBytes = cudaContextBytes + commBytes + fragmentationBytes
  return {
    cudaContextBytes, commBytes, graphBytes, activationBytes,
    logitsBytes: logits, fragmentationBytes, nonTorchBytes,
    totalBytes: nonTorchBytes + graphBytes + activationBytes + logits,
    warnings,
  }
}
