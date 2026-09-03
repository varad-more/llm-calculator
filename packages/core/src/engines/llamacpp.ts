import type { Warning } from '../types.ts'
import { gpuWarnings, baseAllocation, feasible, poolTokens, solveLargestFit, usableVram } from './common.ts'
import type { AllocationPlan, EngineAdapter, PlanInput } from './types.ts'

/**
 * llama.cpp allocates one contiguous KV buffer for `n_ctx` tokens shared across `n_parallel`
 * slots — there is no paging, so a slot holds its full share whether or not it is used:
 * \\[ B_{KV} = n_{ctx}\\, b_{tok}, \\qquad \\ell_{slot} = n_{ctx} / n_{parallel} \\]
 * Weight bytes come from the measured GGUF bits-per-weight table, not the nominal bit width.
 * @see docs/MATH.md#llamacpp
 */
export function planLlamacpp(i: PlanInput): AllocationPlan {
  const { weights, perDevice, kv, overhead } = baseAllocation(i)
  const warnings: Warning[] = [...gpuWarnings(i), ...i.model.warnings, ...weights.warnings, ...kv.warnings, ...overhead.warnings]
  if (!i.quant.startsWith('gguf:')) {
    warnings.push({ code: 'llamacpp_needs_gguf', message: `llama.cpp serves GGUF; quant "${i.quant}" is not a GGUF scheme. Use e.g. gguf:Q4_K_M.` })
  }

  const usable = usableVram(i)
  const budget = usable * i.memoryUtilization
  const availableKv = budget - perDevice - overhead.cudaContextBytes - overhead.activationBytes
  const perTok = kv.perTokenBytesPerDevice
  const { maxTokens } = poolTokens(availableKv, perTok, i.blockSize)
  // n_ctx is one contiguous unpaged buffer shared by all slots, so the requirement is
  // the whole batch's tokens at full width, not the paged per-sequence cost.
  const needTokens = i.workload.concurrency * i.workload.avgSeqLen
  const fits = feasible(i, availableKv, needTokens * perTok)

  const plan: AllocationPlan = {
    engine: 'llamacpp', input: i, weights, weightBytesPerDevice: perDevice, kv, overhead,
    usableVramBytes: usable, budgetBytes: budget, availableKvBytes: availableKv,
    numBlocks: 0, maxTokens, requiredKvBytes: needTokens * perTok, fits,
    freeBytes: usable - perDevice - needTokens * perTok - overhead.cudaContextBytes - overhead.activationBytes,
    warnings, validated: false,
  }
  if (!fits) plan.autofix = solveLargestFit(i, Math.max(0, availableKv))
  return plan
}

/** `llama-server` flags. @see docs/MATH.md#llamacpp */
export function emitLlamacppFlags(plan: AllocationPlan): string[] {
  const i = plan.input
  const seqs = plan.fits ? i.workload.concurrency : (plan.autofix?.maxNumSeqs ?? 1)
  const perSlot = plan.fits ? i.workload.maxModelLen : (plan.autofix?.maxModelLen ?? 0)
  const flags = [`-c ${perSlot * Math.max(1, seqs)}`, `-np ${seqs}`, '-ngl 99', `-b ${i.workload.chunkTokens}`]
  if (i.kvDtype.startsWith('fp8') || i.kvDtype === 'int8') flags.push('-ctk q8_0', '-ctv q8_0')
  if (i.parallel.tp > 1) flags.push(`-ts ${Array(i.parallel.tp).fill(1).join(',')}`)
  return flags
}

export const llamacpp: EngineAdapter = { name: 'llamacpp', plan: planLlamacpp, emitFlags: emitLlamacppFlags }
