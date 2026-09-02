import { listModels, getModelSnapshot, normalizeConfig, type ModelSpec } from '@llmsize/core'

/** URL slug for a model's static page: meta-llama/Llama-3.1-70B-Instruct -> llama-3-1-70b-instruct. */
export function slugFor(id: string): string {
  return id.split('/').pop()!.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

const BY_SLUG = new Map(listModels().map((id) => [slugFor(id), id]))

export function modelForSlug(slug: string): string | undefined {
  return BY_SLUG.get(slug.replace(/-vram$/, ''))
}

export function allSlugs(): string[] {
  return [...BY_SLUG.keys()].map((s) => `${s}-vram`)
}

export function specFor(id: string): ModelSpec {
  const snap = getModelSnapshot(id)
  return normalizeConfig(snap.config, { id, measuredWeightBytes: snap.measuredWeightBytes })
}

/** Human summary of a model's architecture, for page copy and metadata. */
export function describe(spec: ModelSpec): string {
  const kinds = [...new Set(spec.layers.map((l) => l.kind))]
  const attn = kinds.length > 1 ? `hybrid ${kinds.join('+')}` : kinds[0]
  const moe = spec.moe ? `, ${spec.moe.numExperts} experts (top-${spec.moe.topK})` : ''
  return `${spec.numLayers} layers, ${attn}${moe}`
}
