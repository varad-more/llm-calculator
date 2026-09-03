import { InfeasibleError } from '../errors.ts'
import { kvBytesForSequence } from '../kv.ts'
import type { Warning } from '../types.ts'
import { gpuWarnings, baseAllocation, feasible, poolTokens, solveLargestFit, usableVram } from './common.ts'
import type { AllocationPlan, EngineAdapter, PlanInput } from './types.ts'

/**
 * vLLM V1 allocation, matching what the engine does at startup.
 *
 * vLLM profiles a dummy forward pass, subtracts everything it saw from its utilisation
 * budget, and turns the remainder into a fixed number of KV blocks:
 * \\[ B_{KV} = V_{usable}\\,u - (B_{w} + B_{nontorch} + B_{act} + B_{graph}) \\]
 * \\[ N_{blocks} = \\left\\lfloor \\frac{B_{KV}}{B\\, b_{tok}} \\right\\rfloor, \\quad
 *     T_{max} = N_{blocks}\\, B \\]
 * where \\(B\\) is `--block-size` (default 16) and \\(b_{tok}\\) the per-token KV bytes on
 * one device. The config is feasible iff
 * \\( L \\le T_{max} \\) and \\( S \\cdot \\bar{\\ell} \\le T_{max} \\).
 *
 * @see docs/MATH.md#vllm
 */
export function planVllm(i: PlanInput): AllocationPlan {
  const { weights, perDevice, kv, overhead } = baseAllocation(i)
  const warnings: Warning[] = [...gpuWarnings(i), ...i.model.warnings, ...weights.warnings, ...kv.warnings, ...overhead.warnings]

  const usable = usableVram(i)
  const budget = usable * i.memoryUtilization
  const reserved = perDevice + overhead.nonTorchBytes + overhead.activationBytes + overhead.logitsBytes + overhead.graphBytes
  const availableKv = budget - reserved

  if (availableKv <= 0) {
    warnings.push({
      code: 'no_kv_headroom',
      message: `Weights (${(perDevice / 2 ** 30).toFixed(1)} GiB) plus overhead (${((reserved - perDevice) / 2 ** 30).toFixed(1)} GiB) exceed the ${(budget / 2 ** 30).toFixed(1)} GiB budget: vLLM will abort with "No available memory for the cache blocks".`,
    })
  }

  const perTok = kv.perTokenBytesPerDevice
  const { numBlocks, maxTokens } = poolTokens(availableKv, perTok, i.blockSize)
  const requiredKv = kv.totalBytesPerDevice

  const needTokens = i.workload.concurrency * i.workload.avgSeqLen
  const fits = feasible(i, availableKv, requiredKv)

  const plan: AllocationPlan = {
    engine: 'vllm', input: i, weights, weightBytesPerDevice: perDevice, kv, overhead,
    usableVramBytes: usable, budgetBytes: budget, availableKvBytes: availableKv,
    numBlocks, maxTokens, requiredKvBytes: requiredKv,
    fits, freeBytes: usable - reserved - requiredKv, warnings, validated: false,
  }

  if (!fits) {
    plan.autofix = refineAutofix(i, availableKv)
    warnings.push({
      code: 'infeasible',
      message: i.workload.maxModelLen > maxTokens
        ? `--max-model-len ${i.workload.maxModelLen} needs ${(kvBytesForSequence(i.model, i.workload.maxModelLen, i.parallel, { kvDtype: i.kvDtype, blockSize: i.blockSize }) / 2 ** 30).toFixed(1)} GiB of KV but only ${(Math.max(0, availableKv) / 2 ** 30).toFixed(1)} GiB is free (KV pool holds ${maxTokens} tokens).`
        : `${i.workload.concurrency} sequences x ${i.workload.avgSeqLen} tokens = ${needTokens} tokens exceeds the ${maxTokens}-token KV pool.`,
    })
  }
  return plan
}

/**
 * Re-solve the fit after shrinking `max_num_seqs`, because CUDA-graph capture and the
 * sampling logits both scale with it: fewer sequences frees memory that becomes more KV.
 * Three fixed-point passes converge for every configuration we have tested.
 * @see docs/MATH.md#autofix
 */
function refineAutofix(i: PlanInput, availableKv: number): { maxModelLen: number; maxNumSeqs: number } {
  let fix = solveLargestFit(i, Math.max(0, availableKv))
  for (let pass = 0; pass < 3; pass++) {
    const seqs = Math.max(1, fix.maxNumSeqs)
    const probe: PlanInput = { ...i, workload: { ...i.workload, concurrency: seqs, maxModelLen: Math.max(1, fix.maxModelLen) } }
    const { perDevice, overhead } = baseAllocation(probe)
    const avail = usableVram(probe) * probe.memoryUtilization -
      (perDevice + overhead.nonTorchBytes + overhead.activationBytes + overhead.logitsBytes + overhead.graphBytes)
    fix = solveLargestFit(probe, Math.max(0, avail))
  }
  return fix
}

/**
 * The `vllm serve` command line for this plan. When the request does not fit, the
 * auto-fixed context and concurrency are emitted instead of the impossible ones.
 * @see docs/MATH.md#vllm
 */
export function emitVllmFlags(plan: AllocationPlan): string[] {
  const i = plan.input
  const len = plan.fits ? i.workload.maxModelLen : (plan.autofix?.maxModelLen ?? 0)
  const seqs = plan.fits ? i.workload.concurrency : (plan.autofix?.maxNumSeqs ?? 0)
  const flags = [
    `--tensor-parallel-size ${i.parallel.tp}`,
    ...(i.parallel.pp > 1 ? [`--pipeline-parallel-size ${i.parallel.pp}`] : []),
    `--max-model-len ${len}`,
    `--max-num-seqs ${seqs}`,
    `--gpu-memory-utilization ${i.memoryUtilization.toFixed(2)}`,
    `--max-num-batched-tokens ${i.workload.chunkTokens}`,
  ]
  if (i.blockSize !== 16) flags.push(`--block-size ${i.blockSize}`)
  if (i.kvDtype.startsWith('fp8')) flags.push(`--kv-cache-dtype ${i.kvDtype === 'fp8' ? 'fp8' : i.kvDtype}`)
  if (i.quant.startsWith('awq')) flags.push('--quantization awq')
  else if (i.quant.startsWith('gptq')) flags.push('--quantization gptq')
  else if (i.quant === 'fp8') flags.push('--quantization fp8')
  if (!i.cudaGraphs) flags.push('--enforce-eager')
  if (i.workload.prefixCache?.enabled) flags.push('--enable-prefix-caching')
  else flags.push('--no-enable-prefix-caching')
  return flags
}

/** Full copy-pasteable server command. @see docs/MATH.md#vllm */
export function emitVllmCommand(plan: AllocationPlan): string {
  const ref = plan.input.modelRef ?? plan.input.model.id
  return ['vllm serve ' + ref, ...emitVllmFlags(plan)].join(' \\\n  ')
}

export const vllm: EngineAdapter = { name: 'vllm', plan: planVllm, emitFlags: emitVllmFlags }
export { InfeasibleError }
