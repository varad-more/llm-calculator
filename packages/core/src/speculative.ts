import type { ThroughputEstimate } from './throughput.ts'

export interface SpeculativeConfig {
  /** Draft tokens proposed per verification step (vLLM --num-speculative-tokens). */
  numSpeculativeTokens: number
  /** Per-token acceptance rate of the draft, 0..1. Measure it; do not guess. */
  acceptanceRate: number
  /** Decode step time of the draft model on the same hardware, in seconds. */
  draftStepSeconds: number
}

export interface SpeculativeEstimate {
  /** Expected accepted tokens per verification cycle, including the bonus token. */
  tokensPerCycle: number
  cycleSeconds: number
  itlMs: number
  tokensPerSecond: number
  /** Ratio against the same configuration without speculation. */
  speedup: number
  /** Below 1.0 speculation is a net loss; the draft is too slow or too wrong. */
  worthwhile: boolean
}

/**
 * Speculative decoding throughput.
 *
 * With \(k\) proposed tokens and per-token acceptance \(\alpha\), the draft chain is accepted
 * until the first rejection, and the target model contributes one bonus token, so the expected
 * yield per cycle is the truncated geometric series
 *
 * \\[ E[\\text{tokens}] = \\sum_{j=0}^{k} \\alpha^{j} = \\frac{1 - \\alpha^{k+1}}{1 - \\alpha} \\]
 *
 * A cycle costs \(k\) draft steps plus one target forward. The target verifies all \(k+1\)
 * positions in a single pass, and decode is memory-bound, so that pass costs essentially the
 * same as a one-token step:
 *
 * \\[ t_{cycle} = k\\, t_{draft} + t_{target} , \\qquad \\text{ITL} = \\frac{t_{cycle}}{E[\\text{tokens}]} \\]
 *
 * Speculation is a **latency** optimisation. At high batch the target step is already
 * compute-saturated and the extra draft work is pure overhead, which is why the speedup here
 * can and does come out below 1.
 *
 * @see docs/MATH.md#speculative-decoding
 */
export function speculativeThroughput(base: ThroughputEstimate, cfg: SpeculativeConfig): SpeculativeEstimate {
  const k = Math.max(0, cfg.numSpeculativeTokens)
  const a = Math.min(Math.max(cfg.acceptanceRate, 0), 1)
  const tokensPerCycle = a === 1 ? k + 1 : (1 - a ** (k + 1)) / (1 - a)
  const cycleSeconds = k * cfg.draftStepSeconds + base.decode.stepSeconds
  const itl = cycleSeconds / tokensPerCycle
  return {
    tokensPerCycle,
    cycleSeconds,
    itlMs: itl * 1000,
    tokensPerSecond: 1 / itl,
    speedup: base.decode.stepSeconds / itl,
    worthwhile: itl < base.decode.stepSeconds,
  }
}

/** vLLM speculative-decoding flags. @see docs/MATH.md#speculative-decoding */
export function emitSpeculativeFlags(draftModelRef: string, cfg: SpeculativeConfig): string[] {
  return [
    `--speculative-config '${JSON.stringify({ model: draftModelRef, num_speculative_tokens: cfg.numSpeculativeTokens })}'`,
  ]
}
