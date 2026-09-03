import { kvBytesForSequence, kvShards } from './kv.ts'
import { size, type SizingRequest, type SizingResult } from './plan.ts'
import type { Warning } from './types.ts'

export interface PoolShape {
  /** GPUs in this pool. */
  gpus: number
  tp: number
  pp?: number
  /** Defaults to the top-level gpu id; set to size a heterogeneous deployment. */
  gpu?: string
}

export interface DisaggRequest extends Omit<SizingRequest, 'concurrency' | 'context'> {
  prefill: PoolShape
  decode: PoolShape
  context: number
  /** Concurrency each pool is sized for, independently. */
  prefillConcurrency: number
  decodeConcurrency: number
  promptTokens: number
  outputTokens: number
  /** Bytes/s between the pools. NVLink ~900e9, 400G IB ~50e9, 100G Ethernet ~12.5e9. */
  transferBytesPerSec: number
}

export interface DisaggPlan {
  prefill: SizingResult
  decode: SizingResult
  transfer: {
    bytesPerRequest: number
    secondsPerRequest: number
    /** Bandwidth the link must sustain at the balanced request rate. */
    requiredBytesPerSec: number
    /** Fraction of a decode-side TTFT budget spent moving KV. */
    shareOfTtft: number
  }
  balance: {
    prefillRequestsPerSec: number
    decodeRequestsPerSec: number
    /** Decode instances per prefill instance for a balanced pipeline. */
    decodePerPrefill: number
    bottleneck: 'prefill' | 'decode' | 'transfer'
  }
  warnings: Warning[]
}

/**
 * Disaggregated prefill/decode sizing.
 *
 * The two phases have opposite bottlenecks — prefill is compute-bound, decode is
 * bandwidth-bound — so splitting them lets each pool be sized and parallelised on its own.
 * The cost is that every request's KV cache must cross the wire once:
 *
 * \\[ B_{xfer} = b_{seq}(\\ell_{prompt}) \\cdot \\text{shards}(TP_{prefill}) , \\qquad t_{xfer} = \\frac{B_{xfer}}{BW_{link}} \\]
 *
 * \(b_{seq}\) is per-device, so recovering the whole cache multiplies by the number of ways
 * it is actually sharded — which is \(\min(TP, n_{kv})\), and **1** for MLA, whose latent is
 * replicated rather than split. Using \(TP\) blindly overstates a DeepSeek transfer 8-fold.
 *
 * Balance follows from each pool's own rate:
 *
 * \\[ R_{prefill} = \\frac{\\text{tok/s}_{prefill}}{\\ell_{prompt}} , \\qquad
 *    R_{decode} = \\frac{\\text{tok/s}_{decode}}{\\ell_{out}} , \\qquad
 *    \\frac{N_{decode}}{N_{prefill}} = \\frac{R_{prefill}}{R_{decode}} \\]
 *
 * A link that cannot carry \(B_{xfer} R\) bytes per second is the real bottleneck, and on
 * commodity Ethernet it usually is: a 32k-token Llama-70B prompt is 2.5 GiB of KV per request.
 *
 * @see docs/MATH.md#disaggregation
 */
export function planDisaggregated(req: DisaggRequest): DisaggPlan {
  const shared = { ...req } as any
  const context = Math.max(req.context, req.promptTokens + req.outputTokens)
  delete shared.prefill; delete shared.decode; delete shared.prefillConcurrency
  delete shared.decodeConcurrency; delete shared.promptTokens; delete shared.outputTokens
  delete shared.transferBytesPerSec

  const prefill = size({
    ...shared, gpu: req.prefill.gpu ?? req.gpu, tp: req.prefill.tp, pp: req.prefill.pp ?? 1,
    context, concurrency: req.prefillConcurrency, avgSeqLen: req.promptTokens,
  })
  const decode = size({
    ...shared, gpu: req.decode.gpu ?? req.gpu, tp: req.decode.tp, pp: req.decode.pp ?? 1,
    context, concurrency: req.decodeConcurrency,
    avgSeqLen: req.promptTokens + req.outputTokens,
  })

  const bytesPerRequest = kvBytesForSequence(
    prefill.plan.input.model, req.promptTokens, prefill.plan.input.parallel,
    { kvDtype: prefill.plan.input.kvDtype, blockSize: prefill.plan.input.blockSize },
  ) * kvShards(prefill.plan.input.model, req.prefill.tp)
  const secondsPerRequest = bytesPerRequest / req.transferBytesPerSec

  const prefillRequestsPerSec = prefill.throughput.prefill.tokensPerSecond / Math.max(1, req.promptTokens)
  const decodeRequestsPerSec = decode.throughput.decode.tokensPerSecond / Math.max(1, req.outputTokens)
  const transferRequestsPerSec = 1 / Math.max(secondsPerRequest, Number.MIN_VALUE)

  const rates: [number, DisaggPlan['balance']['bottleneck']][] = [
    [prefillRequestsPerSec * req.prefill.gpus / Math.max(1, req.prefill.tp * (req.prefill.pp ?? 1)), 'prefill'],
    [decodeRequestsPerSec * req.decode.gpus / Math.max(1, req.decode.tp * (req.decode.pp ?? 1)), 'decode'],
    [transferRequestsPerSec, 'transfer'],
  ]
  rates.sort((a, b) => a[0] - b[0])
  const bottleneck = rates[0]![1]
  const balancedRate = rates[0]![0]

  const warnings: Warning[] = []
  if (bottleneck === 'transfer') {
    warnings.push({
      code: 'kv_transfer_bound',
      message: `The interconnect is the bottleneck: ${(bytesPerRequest / 2 ** 30).toFixed(2)} GiB of KV per request at ${(req.transferBytesPerSec / 1e9).toFixed(0)} GB/s caps the pipeline at ${transferRequestsPerSec.toFixed(1)} req/s. Use fp8 KV, an MLA model, or a faster fabric.`,
    })
  }
  if (secondsPerRequest > decode.throughput.decode.stepSeconds * 10) {
    warnings.push({
      code: 'kv_transfer_dominates_ttft',
      message: `KV transfer takes ${(secondsPerRequest * 1000).toFixed(0)} ms, ${(secondsPerRequest / prefill.throughput.prefill.ttftSeconds).toFixed(1)}x the prefill compute itself.`,
    })
  }
  if (!prefill.plan.fits) warnings.push({ code: 'prefill_pool_infeasible', message: 'The prefill pool does not fit its own workload.' })
  if (!decode.plan.fits) warnings.push({ code: 'decode_pool_infeasible', message: 'The decode pool does not fit its own workload.' })

  return {
    prefill, decode,
    transfer: {
      bytesPerRequest, secondsPerRequest,
      requiredBytesPerSec: bytesPerRequest * balancedRate,
      shareOfTtft: secondsPerRequest / (prefill.throughput.prefill.ttftSeconds + secondsPerRequest),
    },
    balance: {
      prefillRequestsPerSec, decodeRequestsPerSec,
      decodePerPrefill: decodeRequestsPerSec > 0 ? prefillRequestsPerSec / decodeRequestsPerSec : Infinity,
      bottleneck,
    },
    warnings,
  }
}
