import type { Warning } from '../types.ts'
import { gpuWarnings, baseAllocation, feasible, poolTokens, solveLargestFit, usableVram } from './common.ts'
import type { AllocationPlan, EngineAdapter, PlanInput } from './types.ts'

/**
 * TensorRT-LLM differs structurally: the activation budget is **frozen at engine build
 * time**. `trtllm-build --max_batch_size/--max_num_tokens` bakes fixed workspace buffers
 * into the plan file, so at serve time only the KV pool is elastic:
 *
 * \\[ B_{act} = \\text{const}(\\text{build max\\_num\\_tokens}), \\qquad
 *    B_{KV} = f_{free}\\,(V_{usable} - B_{w} - B_{act}^{build}) \\]
 *
 * Running with a smaller runtime batch does **not** give the KV pool more room, which is
 * the usual surprise when moving a config over from vLLM.
 *
 * @see docs/MATH.md#tensorrt-llm
 */
export function planTrtllm(i: PlanInput): AllocationPlan {
  const { weights, perDevice, kv, overhead } = baseAllocation(i)
  const warnings: Warning[] = [...gpuWarnings(i), ...i.model.warnings, ...weights.warnings, ...kv.warnings, ...overhead.warnings, {
    code: 'trtllm_build_time_budget',
    message: 'TensorRT-LLM freezes activation workspace at build time; these numbers assume the engine was built with the same max_num_tokens/max_batch_size shown in the emitted flags.',
  }]

  const usable = usableVram(i)
  const buildActivation = overhead.activationBytes + overhead.logitsBytes
  const budget = usable * i.memoryUtilization
  const availableKv = i.memoryUtilization * (usable - perDevice - buildActivation - overhead.nonTorchBytes)

  const perTok = kv.perTokenBytesPerDevice
  const { numBlocks, maxTokens } = poolTokens(availableKv, perTok, i.blockSize)
  const fits = feasible(i, availableKv, kv.totalBytesPerDevice)

  const plan: AllocationPlan = {
    engine: 'trtllm', input: i, weights, weightBytesPerDevice: perDevice, kv, overhead,
    usableVramBytes: usable, budgetBytes: budget, availableKvBytes: availableKv,
    numBlocks, maxTokens,
    requiredKvBytes: kv.totalBytesPerDevice, fits,
    freeBytes: usable - perDevice - kv.totalBytesPerDevice - buildActivation - overhead.nonTorchBytes,
    warnings, validated: false,
  }
  if (!fits) plan.autofix = solveLargestFit(i, Math.max(0, availableKv))
  return plan
}

/** Build-time flags first, then serve-time. @see docs/MATH.md#tensorrt-llm */
export function emitTrtllmFlags(plan: AllocationPlan): string[] {
  const i = plan.input
  const len = plan.fits ? i.workload.maxModelLen : (plan.autofix?.maxModelLen ?? 0)
  const seqs = plan.fits ? i.workload.concurrency : (plan.autofix?.maxNumSeqs ?? 0)
  return [
    `--max_batch_size ${seqs}`,
    `--max_seq_len ${len}`,
    `--max_num_tokens ${i.workload.chunkTokens}`,
    `--tp_size ${i.parallel.tp}`,
    ...(i.parallel.pp > 1 ? [`--pp_size ${i.parallel.pp}`] : []),
    ...(i.kvDtype.startsWith('fp8') ? ['--kv_cache_dtype fp8'] : []),
    `--kv_cache_free_gpu_memory_fraction ${i.memoryUtilization.toFixed(2)}`,
  ]
}

export const trtllm: EngineAdapter = { name: 'trtllm', plan: planTrtllm, emitFlags: emitTrtllmFlags }
