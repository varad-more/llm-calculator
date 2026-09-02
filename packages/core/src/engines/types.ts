import type {
  Assumptions, GpuSpec, KvBreakdown, KvDtype, ModelSpec, Parallelism, QuantScheme, Warning, Workload,
} from '../types.ts'
import type { OverheadBreakdown } from '../overhead.ts'
import type { WeightResult } from '../weights.ts'

export type EngineName = 'vllm' | 'sglang' | 'trtllm' | 'llamacpp'

export interface PlanInput {
  model: ModelSpec
  gpu: GpuSpec
  parallel: Parallelism
  workload: Workload
  quant: QuantScheme
  kvDtype: KvDtype
  /** vLLM --gpu-memory-utilization / SGLang --mem-fraction-static ceiling. */
  memoryUtilization: number
  blockSize: number
  cudaGraphs: boolean
  assumptions: Assumptions
  /** HF repo id to put in the emitted serve command. */
  modelRef?: string
}

export interface AllocationPlan {
  engine: EngineName
  input: PlanInput
  weights: WeightResult
  weightBytesPerDevice: number
  kv: KvBreakdown
  overhead: OverheadBreakdown
  /** Device VRAM after the driver's own reservation. */
  usableVramBytes: number
  /** usableVram x memoryUtilization — the engine's self-imposed ceiling. */
  budgetBytes: number
  /** What is left for the KV pool after weights, overhead and activations. */
  availableKvBytes: number
  numBlocks: number
  maxTokens: number
  requiredKvBytes: number
  fits: boolean
  freeBytes: number
  /** Largest (context, concurrency) pair that does fit, when the request does not. */
  autofix?: { maxModelLen: number; maxNumSeqs: number }
  warnings: Warning[]
  /** Every number above is PREDICTED unless a validation case covers this triple. */
  validated: false | { engineVersion: string; errors: Record<string, number> }
}

export interface EngineAdapter {
  name: EngineName
  plan(input: PlanInput): AllocationPlan
  emitFlags(plan: AllocationPlan): string[]
}
