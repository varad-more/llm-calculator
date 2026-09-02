import { DATA } from './generated.ts'
import { UnknownEntityError } from './errors.ts'
import type { Assumptions, AssumptionOverrides, AssumptionKey, GpuSpec } from './types.ts'

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
