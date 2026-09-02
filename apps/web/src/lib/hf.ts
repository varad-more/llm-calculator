'use client'

import { listModels, getModelSnapshot } from '@llmsize/core'

export type ConfigSource =
  | { kind: 'snapshot'; id: string; config: Record<string, unknown>; measuredWeightBytes?: number }
  | { kind: 'live'; id: string; config: Record<string, unknown> }
  | { kind: 'pasted'; id: string; config: Record<string, unknown> }

/**
 * Resolve a model config, in the order that actually works in a browser.
 *
 * 1. The build-time snapshot in data/models/. Always available, works offline, and covers
 *    the gated repos (meta-llama/*) that return 401 to an unauthenticated fetch — which is
 *    exactly the set of models people search for most.
 * 2. A live fetch from huggingface.co. CORS is open on resolve/main/config.json (verified:
 *    the response echoes Access-Control-Allow-Origin, and so does the CDN redirect), so this
 *    works from the browser with no proxy. An optional user token unlocks gated repos.
 * 3. Raw JSON pasted by the user, for private or unreleased models.
 */
export async function resolveConfig(id: string, token?: string): Promise<ConfigSource> {
  if (listModels().includes(id)) {
    const snap = getModelSnapshot(id)
    return { kind: 'snapshot', id, config: snap.config, measuredWeightBytes: snap.measuredWeightBytes }
  }
  const res = await fetch(`https://huggingface.co/${id}/resolve/main/config.json`, {
    headers: token ? { authorization: `Bearer ${token}` } : undefined,
  })
  if (res.status === 401 || res.status === 403) {
    throw new Error(
      `${id} is a gated repo. Accept its licence on huggingface.co and paste a read token, or paste the config.json directly.`,
    )
  }
  if (!res.ok) throw new Error(`huggingface.co returned ${res.status} for ${id}`)
  return { kind: 'live', id, config: await res.json() }
}

export function parsePasted(id: string, text: string): ConfigSource {
  return { kind: 'pasted', id: id || 'pasted-config', config: JSON.parse(text) }
}
