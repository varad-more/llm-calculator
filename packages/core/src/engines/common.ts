import { assume } from '../data.ts'
import { kvBytesForSequence, kvCacheBytes } from '../kv.ts'
import { overheadBytes, type OverheadInput } from '../overhead.ts'
import { weightBytes, weightBytesPerDevice } from '../weights.ts'
import type { PlanInput } from './types.ts'

/**
 * Device memory the engine may actually touch:
 * \\[ V_{usable} = V_{nominal}\\,(1 - r_{driver}) \\]
 * The driver reserves a slice of the nominal capacity (ECC, page tables), which is why
 * an "80GB" H100 reports ~79.6 GiB to torch.
 * @see docs/MATH.md#usable-vram
 */
export function usableVram(i: PlanInput): number {
  return i.gpu.vramBytes * (1 - assume(i.assumptions, 'driver_reserved_vram_fraction'))
}

/** Shared weight/KV/overhead computation every engine adapter starts from. @see docs/MATH.md#allocation */
export function baseAllocation(i: PlanInput, o: { logitsForAllPositions?: boolean } = {}) {
  const weights = weightBytes(i.model, { quant: i.quant })
  const perDevice = weightBytesPerDevice(weights.totalBytes, i.parallel)
  const kv = kvCacheBytes(i.model, i.workload, i.parallel, { kvDtype: i.kvDtype, blockSize: i.blockSize })
  const oi: OverheadInput = {
    model: i.model, workload: i.workload, parallel: i.parallel, actDtypeBytes: 2,
    cudaGraphs: i.cudaGraphs, logitsForAllPositions: o.logitsForAllPositions, assumptions: i.assumptions,
  }
  const overhead = overheadBytes(oi, perDevice)
  return { weights, perDevice, kv, overhead }
}

/**
 * Largest (max_model_len, max_num_seqs) that fits a KV pool of `maxTokens` tokens.
 *
 * \\[ L^{*} = \\min(\\text{maxTokens},\\, L_{model}) , \\quad S^{*} = \\lfloor \\text{maxTokens} / \\bar{\\ell} \\rfloor \\]
 *
 * Sliding-window layers make token capacity non-linear in context length, so the context
 * solve is a bisection on the true per-sequence KV cost rather than a division.
 *
 * @see docs/MATH.md#autofix
 */
export function solveLargestFit(i: PlanInput, availableKvBytes: number): { maxModelLen: number; maxNumSeqs: number } {
  const cost = (tokens: number) =>
    kvBytesForSequence(i.model, tokens, i.parallel, { kvDtype: i.kvDtype, blockSize: i.blockSize })

  let lo = 0
  let hi = i.model.maxPositionEmbeddings
  if (cost(hi) <= availableKvBytes) lo = hi
  else {
    while (hi - lo > i.blockSize) {
      const mid = Math.floor((lo + hi) / 2)
      if (cost(mid) <= availableKvBytes) lo = mid
      else hi = mid
    }
  }
  const perSeq = cost(Math.min(i.workload.avgSeqLen, i.model.maxPositionEmbeddings))
  return {
    maxModelLen: Math.floor(lo / i.blockSize) * i.blockSize,
    maxNumSeqs: perSeq > 0 ? Math.floor(availableKvBytes / perSeq) : 0,
  }
}

/**
 * Feasibility, decided in **bytes** rather than tokens.
 *
 * The token form \\( L \\le T_{max} \\wedge S\\bar{\\ell} \\le T_{max} \\) is only valid when every
 * layer caches every token. A sliding-window model caches \\(\\min(\\ell, W)\\) per windowed layer,
 * so its KV pool holds far more *sequences* than \\(T_{max}/\\bar{\\ell}\\) suggests. Comparing
 * bytes is exactly equivalent for homogeneous models and correct for hybrid ones.
 *
 * \\[ \\text{fits} \\iff B_{KV}^{avail} > 0 \;\\wedge\; B_{KV}^{req} \\le B_{KV}^{avail}
 *    \;\\wedge\; b_{seq}(L) \\le B_{KV}^{avail} \\]
 *
 * @see docs/MATH.md#feasibility
 */
export function feasible(i: PlanInput, availableKvBytes: number, requiredKvBytes: number): boolean {
  if (availableKvBytes <= 0) return false
  if (requiredKvBytes > availableKvBytes) return false
  const longest = kvBytesForSequence(i.model, i.workload.maxModelLen, i.parallel, {
    kvDtype: i.kvDtype, blockSize: i.blockSize,
  })
  return longest <= availableKvBytes
}

/** Token capacity of a KV pool, as engines report it. Clamped at zero. @see docs/MATH.md#feasibility */
export function poolTokens(availableKvBytes: number, perTokenBytes: number, blockSize: number): { numBlocks: number; maxTokens: number } {
  if (perTokenBytes <= 0 || availableKvBytes <= 0) return { numBlocks: 0, maxTokens: 0 }
  const numBlocks = Math.max(0, Math.floor(availableKvBytes / (blockSize * perTokenBytes)))
  return { numBlocks, maxTokens: numBlocks * blockSize }
}
