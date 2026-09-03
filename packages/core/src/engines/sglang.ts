import type { Warning } from '../types.ts'
import { gpuWarnings, baseAllocation, feasible, poolTokens, solveLargestFit, usableVram } from './common.ts'
import type { AllocationPlan, EngineAdapter, PlanInput } from './types.ts'

/**
 * SGLang reserves a *static* pool up front rather than measuring at runtime:
 * `--mem-fraction-static` covers weights **and** the KV pool, and whatever is left
 * must absorb activations, CUDA graphs and the CUDA context.
 *
 * \\[ B_{KV} = V_{usable}\\, f_{static} - B_{w}, \\qquad
 *    f_{static}^{*} = \\frac{B_{w} + B_{KV}^{req}}{V_{usable}} \\]
 * Feasible iff \\( (1-f_{static})V_{usable} \\ge B_{act} + B_{graph} + B_{ctx} \\).
 *
 * RadixAttention makes prefix reuse the default, so the prefix-cache term is on unless
 * the caller disables it.
 *
 * @see docs/MATH.md#sglang
 */
export function planSglang(input: PlanInput): AllocationPlan {
  const i: PlanInput = {
    ...input,
    workload: {
      ...input.workload,
      prefixCache: input.workload.prefixCache ?? { enabled: true, hitRate: 0.5, sharedPrefixTokens: 0 },
    },
  }
  const { weights, perDevice, kv, overhead } = baseAllocation(i)
  const warnings: Warning[] = [...gpuWarnings(i), ...i.model.warnings, ...weights.warnings, ...kv.warnings, ...overhead.warnings]

  const usable = usableVram(i)
  const budget = usable * i.memoryUtilization
  const availableKv = budget - perDevice
  const dynamicNeed = overhead.activationBytes + overhead.graphBytes + overhead.cudaContextBytes + overhead.commBytes + overhead.logitsBytes
  const dynamicHave = usable - budget
  if (dynamicHave < dynamicNeed) {
    warnings.push({
      code: 'mem_fraction_static_too_high',
      message: `--mem-fraction-static ${i.memoryUtilization.toFixed(2)} leaves ${(dynamicHave / 2 ** 30).toFixed(1)} GiB for activations but ${(dynamicNeed / 2 ** 30).toFixed(1)} GiB is needed. Lower it to ${(1 - dynamicNeed / usable).toFixed(2)} or below.`,
    })
  }

  const perTok = kv.perTokenBytesPerDevice
  const { numBlocks, maxTokens } = poolTokens(availableKv, perTok, i.blockSize)
  const fits = dynamicHave >= dynamicNeed && feasible(i, availableKv, kv.totalBytesPerDevice)

  const plan: AllocationPlan = {
    engine: 'sglang', input: i, weights, weightBytesPerDevice: perDevice, kv, overhead,
    usableVramBytes: usable, budgetBytes: budget, availableKvBytes: availableKv,
    numBlocks, maxTokens,
    requiredKvBytes: kv.totalBytesPerDevice, fits,
    freeBytes: usable - perDevice - kv.totalBytesPerDevice - overhead.totalBytes,
    warnings, validated: false,
  }
  if (!fits) plan.autofix = solveLargestFit(i, Math.max(0, availableKv))
  return plan
}

/** `sglang.launch_server` flags, including the solved `--mem-fraction-static`. @see docs/MATH.md#sglang */
export function emitSglangFlags(plan: AllocationPlan): string[] {
  const i = plan.input
  const len = plan.fits ? i.workload.maxModelLen : (plan.autofix?.maxModelLen ?? 0)
  const seqs = plan.fits ? i.workload.concurrency : (plan.autofix?.maxNumSeqs ?? 0)
  const staticFraction = Math.min(
    i.memoryUtilization,
    (plan.weightBytesPerDevice + plan.requiredKvBytes) / plan.usableVramBytes,
  )
  const flags = [
    `--tp ${i.parallel.tp}`,
    ...(i.parallel.pp > 1 ? [`--pp ${i.parallel.pp}`] : []),
    `--context-length ${len}`,
    `--max-running-requests ${seqs}`,
    `--mem-fraction-static ${staticFraction.toFixed(2)}`,
    `--chunked-prefill-size ${i.workload.chunkTokens}`,
  ]
  if (i.kvDtype.startsWith('fp8')) flags.push('--kv-cache-dtype fp8_e5m2')
  if (i.quant.startsWith('awq')) flags.push('--quantization awq')
  else if (i.quant.startsWith('gptq')) flags.push('--quantization gptq')
  else if (i.quant === 'fp8') flags.push('--quantization fp8')
  if (i.workload.prefixCache?.enabled === false) flags.push('--disable-radix-cache')
  if (!i.cudaGraphs) flags.push('--disable-cuda-graph')
  return flags
}

export const sglang: EngineAdapter = { name: 'sglang', plan: planSglang, emitFlags: emitSglangFlags }
