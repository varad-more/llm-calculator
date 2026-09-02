import { getGpu, getModelSnapshot, resolveAssumptions } from './data.ts'
import { normalizeConfig } from './model.ts'
import { UnknownEntityError } from './errors.ts'
import { estimateThroughput, type ThroughputEstimate } from './throughput.ts'
import { validationFor, type ValidationRecord } from './validation.ts'
import { emitLlamacppFlags, llamacpp } from './engines/llamacpp.ts'
import { emitSglangFlags, sglang } from './engines/sglang.ts'
import { emitTrtllmFlags, trtllm } from './engines/trtllm.ts'
import { emitVllmFlags, vllm } from './engines/vllm.ts'
import type { AllocationPlan, EngineAdapter, EngineName, PlanInput } from './engines/types.ts'
import type { AssumptionOverrides, KvDtype, ModelSpec, QuantScheme, Workload } from './types.ts'

export const ENGINES: Record<EngineName, EngineAdapter> = { vllm, sglang, trtllm, llamacpp }
const EMITTERS: Record<EngineName, (p: AllocationPlan) => string[]> = {
  vllm: emitVllmFlags, sglang: emitSglangFlags, trtllm: emitTrtllmFlags, llamacpp: emitLlamacppFlags,
}
const SERVE: Record<EngineName, (ref: string) => string> = {
  vllm: (r) => `vllm serve ${r}`,
  sglang: (r) => `python -m sglang.launch_server --model-path ${r}`,
  trtllm: (r) => `trtllm-build --checkpoint_dir ${r}`,
  llamacpp: (r) => `llama-server -m ${r}`,
}

export interface SizingRequest {
  /** Snapshot id ("meta-llama/Llama-3.1-8B-Instruct") or a raw HF config object. */
  model: string | Record<string, any>
  gpu: string
  engine: EngineName
  tp?: number
  pp?: number
  quant?: QuantScheme
  kvDtype?: KvDtype
  /** --max-model-len */
  context: number
  /** --max-num-seqs */
  concurrency: number
  /** Mean tokens per in-flight sequence; defaults to the full context. */
  avgSeqLen?: number
  chunkTokens?: number
  memoryUtilization?: number
  blockSize?: number
  cudaGraphs?: boolean
  prefixCache?: { enabled: boolean; hitRate: number; sharedPrefixTokens: number }
  assume?: AssumptionOverrides
}

export interface SizingResult {
  plan: AllocationPlan
  throughput: ThroughputEstimate
  flags: string[]
  command: string
  /** Null when no validation case covers this (model, gpu, engine) triple. */
  validation: ValidationRecord | null
  /** Human-facing honesty label. */
  label: 'predicted' | 'validated'
}

/** Resolve a snapshot id or raw config into a ModelSpec. @see docs/MATH.md#model-normalization */
export function resolveModel(model: string | Record<string, any>): ModelSpec {
  if (typeof model !== 'string') return normalizeConfig(model)
  const snap = getModelSnapshot(model)
  return normalizeConfig(snap.config, { id: model, measuredWeightBytes: snap.measuredWeightBytes })
}

/**
 * End-to-end sizing: normalize the model, run the engine's allocator, estimate throughput,
 * emit runnable flags, and attach the validation record for this triple if one exists.
 * @see docs/MATH.md#allocation
 */
export function size(req: SizingRequest): SizingResult {
  const engine = ENGINES[req.engine]
  if (!engine) throw new UnknownEntityError('engine', req.engine, Object.keys(ENGINES))

  const model = resolveModel(req.model)
  const workload: Workload = {
    concurrency: req.concurrency,
    avgSeqLen: req.avgSeqLen ?? req.context,
    maxModelLen: req.context,
    chunkTokens: req.chunkTokens ?? 8192,
    prefixCache: req.prefixCache,
  }
  const input: PlanInput = {
    model,
    gpu: getGpu(req.gpu),
    parallel: { tp: req.tp ?? 1, pp: req.pp ?? 1 },
    workload,
    quant: req.quant ?? 'bf16',
    kvDtype: req.kvDtype ?? 'fp16',
    memoryUtilization: req.memoryUtilization ?? (req.engine === 'sglang' ? 0.9 : 0.9),
    blockSize: req.blockSize ?? 16,
    cudaGraphs: req.cudaGraphs ?? true,
    assumptions: resolveAssumptions(req.assume),
    modelRef: typeof req.model === 'string' ? req.model : model.id,
  }

  const plan = engine.plan(input)
  const throughput = estimateThroughput({
    model, gpu: input.gpu, parallel: input.parallel, workload,
    weightBytesPerDevice: plan.weightBytesPerDevice,
    kvBytesPerTokenPerDevice: plan.kv.perTokenBytesPerDevice,
    kvDtype: input.kvDtype,
    computeDtype: input.quant === 'fp8' ? 'fp8' : 'bf16',
    assumptions: input.assumptions,
  })
  const validation = validationFor(model.id, input.gpu.id, req.engine)
  if (validation) plan.validated = { engineVersion: validation.engineVersion, errors: validation.errors }

  const flags = EMITTERS[req.engine](plan)
  return {
    plan, throughput, flags,
    command: [SERVE[req.engine](input.modelRef!), ...flags].join(' \\\n  '),
    validation,
    label: validation ? 'validated' : 'predicted',
  }
}
