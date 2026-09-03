import { DATA } from './generated.ts'
import { UnknownEntityError } from './errors.ts'
import type { Assumptions, AssumptionOverrides, AssumptionKey, GpuSpec, InstanceSpec, QualityMeasurement, QuantScheme } from './types.ts'

/** All GPUs in data/gpus.json. */
export function listGpus(): GpuSpec[] {
  return DATA.gpus as GpuSpec[]
}

/** Look up one GPU by id, or throw with the list of known ids. */
export function getGpu(id: string): GpuSpec {
  const hit = (DATA.gpus as GpuSpec[]).find((g) => g.id === id)
  if (!hit) throw new UnknownEntityError('gpu', id, (DATA.gpus as GpuSpec[]).map((g) => g.id))
  return hit
}

/** Every rentable machine in data/instances.json. */
export function listInstances(): InstanceSpec[] {
  return DATA.instances as InstanceSpec[]
}

/** Look up one instance type by id, or throw with the list of known ids. */
export function getInstance(id: string): InstanceSpec {
  const all = listInstances()
  const hit = all.find((i) => i.id === id)
  if (!hit) throw new UnknownEntityError('instance', id, all.map((i) => i.id))
  return hit
}

/**
 * Published accuracy measurements for a quantization scheme, or an empty list when nobody has
 * measured it. Empty means unmeasured — it does not mean lossless.
 */
export function qualityFor(scheme: QuantScheme): QualityMeasurement[] {
  return (DATA.quality as QualityMeasurement[]).filter((q) => q.scheme === scheme)
}

/** Every quantization accuracy measurement in data/quality.json. */
export function listQuality(): QualityMeasurement[] {
  return DATA.quality as QualityMeasurement[]
}

/** Relative perplexity cost of a measurement, e.g. 0.0091 for +0.91%. */
export function pplPenalty(q: QualityMeasurement): number {
  return q.ppl / q.baselinePpl - 1
}

/** The shipped empirical constants, unmodified. */
export function defaultAssumptions(): Assumptions {
  return DATA.assumptions as Assumptions
}

/** Merge user overrides over the shipped assumptions, keeping rationale/source intact. */
export function resolveAssumptions(overrides: AssumptionOverrides = {}): Assumptions {
  const base = defaultAssumptions()
  const out = {} as Assumptions
  for (const k of Object.keys(base) as AssumptionKey[]) {
    const o = overrides[k]
    out[k] = o === undefined ? base[k] : { ...base[k], value: o, confidence: 'high', rationale: `user override (was ${base[k].value}: ${base[k].rationale})` }
  }
  for (const k of Object.keys(overrides)) {
    if (!(k in base)) throw new UnknownEntityError('assumption', k, Object.keys(base))
  }
  return out
}

/** Numeric value of one assumption. */
export function assume(a: Assumptions, key: AssumptionKey): number {
  return a[key].value
}

/** Snapshotted HF model ids available offline. */
export function listModels(): string[] {
  return Object.keys(DATA.models)
}

/** One snapshotted HF model record ({ config, measuredWeightBytes, source_url }). */
export function getModelSnapshot(id: string): Record<string, any> {
  const hit = DATA.models[id]
  if (!hit) throw new UnknownEntityError('model', id, listModels())
  return hit
}

/** Quantization tables: GGUF measured bits-per-weight, scheme policies, KV dtype widths. */
export function quantData(): any {
  return DATA.quant
}

/** transformers config-class defaults, keyed by model_type. */
export function archDefaults(): Record<string, any> {
  return DATA.archDefaults
}
