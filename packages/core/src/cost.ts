import { listInstances } from './data.ts'
import { size, type SizingRequest, type SizingResult } from './plan.ts'
import type { InstanceSpec } from './types.ts'

/** Tensor-parallel degrees worth trying. Anything else is not a real vLLM/SGLang layout. */
const TP_LADDER = [1, 2, 4, 8, 16]

export interface Slo {
  /** Time to first token, seconds. */
  maxTtftSeconds?: number
  /** Inter-token latency, milliseconds. The reciprocal of per-user streaming speed. */
  maxItlMs?: number
  /** Streaming speed one reader sees, tokens/second. */
  minTokensPerSecondPerUser?: number
}

export interface MachineQuery {
  /** The workload. `gpu`, `tp` and `pp` are overwritten per candidate. */
  req: SizingRequest
  /** Defaults to the whole catalogue. */
  instances?: InstanceSpec[]
  slo?: Slo
  /** Replace the catalogue's on-demand rate, by instance id (spot, savings plan, your rate). */
  usdPerHour?: Record<string, number>
}

export interface MachineOption {
  instance: InstanceSpec
  /** GPUs per replica. */
  tp: number
  /** Independent server processes on the machine: `gpuCount / tp`. */
  replicas: number
  concurrencyPerReplica: number
  fits: boolean
  usdPerHour: number
  /** Whole-machine decode throughput: one replica's rate times the replica count. */
  decodeTokensPerSecond: number
  prefillTokensPerSecond: number
  /** Per replica. Latency does not improve by adding replicas. */
  ttftSeconds: number
  itlMs: number
  usdPerMillionOutputTokens: number
  usdPerMillionInputTokens: number
  /** False when the host cannot page-cache one copy of the checkpoint. */
  hostRamHoldsCheckpoint: boolean
  /** Empty when every SLO clause is met. */
  sloMisses: string[]
  /** The full sizing for one replica, so a caller can show memory, flags or warnings. */
  replica: SizingResult
}

/**
 * Rented price converted to a unit price per token.
 *
 * \[ \$_{1M} = \frac{P_{hour}}{T \cdot 3600} \times 10^{6} \]
 *
 * where \(P_{hour}\) is the machine's hourly rate and \(T\) its tokens/second. The machine is
 * billed whether or not it is busy, so this is the price at **full saturation** — the floor.
 * At 50% utilization the real figure is double, which is the same arithmetic as doubling
 * \(P_{hour}\), so a caller that wants a duty cycle can apply it to the rate instead.
 *
 * @see docs/MATH.md#cost-per-token
 */
export function costPerMillionTokens(usdPerHour: number, tokensPerSecond: number): number {
  if (!(tokensPerSecond > 0)) return Infinity
  return (usdPerHour / (tokensPerSecond * 3600)) * 1e6
}

/**
 * Price every machine in the catalogue for one workload, best layout first.
 *
 * An \(N\)-GPU machine can be one \(TP=N\) server or \(R = N / TP\) independent replicas
 * behind a load balancer, each serving \(\lceil C / R \rceil\) of the \(C\) concurrent
 * sequences:
 *
 * \[ T_{machine} = R \cdot T_{replica}, \qquad R = \frac{N}{TP} \]
 *
 * Which layout wins is not folklore, it falls out of the roofline. A decode step streams the
 * weights resident on the device once and serves the whole batch from that one read. Tensor
 * parallel divides that read by \(TP\); replicas do not — each replica re-reads the full
 * weights for its own share of the batch. At fixed total concurrency TP therefore wins on
 * throughput *and* on latency, until the all-reduce term overtakes what it saves. On 8x H100
 * serving an 8B model that crossover sits past ~1k concurrent sequences: at \(C=64\), TP=8 is
 * 4.3x the throughput of 8 replicas, and by \(C=4096\) TP=4 has overtaken TP=8.
 *
 * TTFT and ITL below are one replica's, because a request is served by exactly one replica.
 * Throughput and therefore \(\$/1\text{M}\) are the whole machine's.
 *
 * Every \(TP\) that divides \(N\) is tried, so the crossover is found rather than assumed.
 * The layout kept is the cheapest per output token among those that fit and meet the SLO, or —
 * if none meet it — the cheapest that fits, carrying the clauses it misses.
 *
 * @see docs/MATH.md#machine-search
 */
export function compareMachines(q: MachineQuery): MachineOption[] {
  const out: MachineOption[] = []
  for (const instance of q.instances ?? listInstances()) {
    const best = bestLayout(instance, q)
    if (best) out.push(best)
  }
  out.sort((a, b) =>
    Number(a.sloMisses.length > 0) - Number(b.sloMisses.length > 0) ||
    Number(b.fits) - Number(a.fits) ||
    a.usdPerMillionOutputTokens - b.usdPerMillionOutputTokens)
  return out
}

function bestLayout(instance: InstanceSpec, q: MachineQuery): MachineOption | null {
  const usdPerHour = q.usdPerHour?.[instance.id] ?? instance.usdPerHour
  const candidates: MachineOption[] = []

  for (const tp of TP_LADDER) {
    if (tp > instance.gpuCount || instance.gpuCount % tp !== 0) continue
    const replicas = instance.gpuCount / tp
    const concurrencyPerReplica = Math.max(1, Math.ceil(q.req.concurrency / replicas))
    let replica: SizingResult
    try {
      replica = size({ ...q.req, gpu: instance.gpu, tp, pp: 1, concurrency: concurrencyPerReplica })
    } catch { continue }

    const decode = replica.throughput.decode.tokensPerSecond * replicas
    const prefill = replica.throughput.prefill.tokensPerSecond * replicas
    const ttftSeconds = replica.throughput.prefill.ttftSeconds
    const itlMs = replica.throughput.decode.itlMs
    const model = replica.plan.input.model
    const checkpointBytes = model.measuredWeightBytes ?? replica.plan.weights.totalBytes

    candidates.push({
      instance, tp, replicas, concurrencyPerReplica,
      fits: replica.plan.fits,
      usdPerHour,
      decodeTokensPerSecond: decode,
      prefillTokensPerSecond: prefill,
      ttftSeconds, itlMs,
      usdPerMillionOutputTokens: costPerMillionTokens(usdPerHour, decode),
      usdPerMillionInputTokens: costPerMillionTokens(usdPerHour, prefill),
      hostRamHoldsCheckpoint: instance.hostRamBytes >= checkpointBytes,
      sloMisses: sloMisses(q.slo, { ttftSeconds, itlMs }),
      replica,
    })
  }

  const fitting = candidates.filter((c) => c.fits)
  const pool = fitting.length ? fitting : candidates
  if (!pool.length) return null
  const meeting = pool.filter((c) => c.sloMisses.length === 0)
  return (meeting.length ? meeting : pool)
    .reduce((a, b) => (b.usdPerMillionOutputTokens < a.usdPerMillionOutputTokens ? b : a))
}

function sloMisses(slo: Slo | undefined, m: { ttftSeconds: number; itlMs: number }): string[] {
  if (!slo) return []
  const out: string[] = []
  if (slo.maxTtftSeconds !== undefined && m.ttftSeconds > slo.maxTtftSeconds) {
    out.push(`TTFT ${(m.ttftSeconds * 1000).toFixed(0)} ms > ${(slo.maxTtftSeconds * 1000).toFixed(0)} ms`)
  }
  if (slo.maxItlMs !== undefined && m.itlMs > slo.maxItlMs) {
    out.push(`ITL ${m.itlMs.toFixed(1)} ms > ${slo.maxItlMs} ms`)
  }
  if (slo.minTokensPerSecondPerUser !== undefined && 1000 / m.itlMs < slo.minTokensPerSecondPerUser) {
    out.push(`${(1000 / m.itlMs).toFixed(1)} tok/s per user < ${slo.minTokensPerSecondPerUser}`)
  }
  return out
}
