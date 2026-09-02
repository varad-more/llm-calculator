import { listModels } from './data.ts'
import { size, type SizingRequest } from './plan.ts'
import type { EngineName } from './engines/types.ts'
import type { KvDtype, QuantScheme } from './types.ts'

export interface ReverseQuery {
  gpu: string
  /** GPUs available. Each candidate is tried at every TP that divides this. */
  gpuCount?: number
  engine?: EngineName
  models?: string[]
  quants?: QuantScheme[]
  kvDtypes?: KvDtype[]
  contexts?: number[]
  concurrency?: number
  /** Drop candidates that do not fit. Default true. */
  fittingOnly?: boolean
  limit?: number
}

export interface ReverseCandidate {
  model: string
  quant: QuantScheme
  kvDtype: KvDtype
  context: number
  tp: number
  fits: boolean
  weightBytesPerDevice: number
  kvPoolTokens: number
  freeBytes: number
  decodeTokensPerSecond: number
  command: string
}

const DEFAULT_QUANTS: QuantScheme[] = ['bf16', 'fp8', 'awq-int4']
const DEFAULT_CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072]

/**
 * Reverse lookup: given hardware, enumerate what actually runs on it.
 *
 * A straight cartesian sweep of (model, quant, kv dtype, context, TP) through the same
 * allocator, ranked by useful capacity. Ranking is by fit first, then by
 * \(\text{context} \times \text{tok/s}\) — a config that serves a long context slowly and one
 * that serves a short context fast are both answers, and the caller decides.
 *
 * @see docs/MATH.md#reverse-lookup
 */
export function reverseLookup(q: ReverseQuery): ReverseCandidate[] {
  const gpuCount = q.gpuCount ?? 1
  const engine = q.engine ?? 'vllm'
  const models = q.models ?? listModels()
  const quants = q.quants ?? DEFAULT_QUANTS
  const kvDtypes = q.kvDtypes ?? ['fp16', 'fp8']
  const contexts = q.contexts ?? DEFAULT_CONTEXTS
  const concurrency = q.concurrency ?? 1
  const tps = [1, 2, 4, 8, 16].filter((t) => t <= gpuCount && gpuCount % t === 0)

  const out: ReverseCandidate[] = []
  for (const model of models) {
    for (const quant of quants) {
      for (const kvDtype of kvDtypes) {
        for (const tp of tps) {
          for (const context of contexts) {
            const req: SizingRequest = { model, gpu: q.gpu, engine, tp, quant, kvDtype, context, concurrency }
            let r
            try { r = size(req) } catch { continue }
            if (q.fittingOnly !== false && !r.plan.fits) continue
            out.push({
              model, quant, kvDtype, context, tp,
              fits: r.plan.fits,
              weightBytesPerDevice: r.plan.weightBytesPerDevice,
              kvPoolTokens: r.plan.maxTokens,
              freeBytes: r.plan.freeBytes,
              decodeTokensPerSecond: r.throughput.decode.tokensPerSecond,
              command: r.command,
            })
          }
        }
      }
    }
  }

  out.sort((a, b) => Number(b.fits) - Number(a.fits) ||
    b.context * b.decodeTokensPerSecond - a.context * a.decodeTokensPerSecond)
  return q.limit ? out.slice(0, q.limit) : out
}

/** The largest context each model can serve on this hardware, one row per model. @see docs/MATH.md#reverse-lookup */
export function largestContextPerModel(q: ReverseQuery): ReverseCandidate[] {
  const best = new Map<string, ReverseCandidate>()
  for (const c of reverseLookup({ ...q, limit: undefined })) {
    const key = `${c.model}|${c.quant}`
    const prev = best.get(key)
    if (!prev || c.context > prev.context) best.set(key, c)
  }
  return [...best.values()].sort((a, b) => b.context - a.context)
}
