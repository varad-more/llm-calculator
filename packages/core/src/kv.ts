import { quantData } from './data.ts'
import { UnknownEntityError } from './errors.ts'
import { layersOnStage } from './model.ts'
import type { KvBreakdown, KvDtype, LayerSpec, ModelSpec, Parallelism, Warning, Workload } from './types.ts'

/** Element width of a KV dtype, in bytes. @see docs/MATH.md#kv-dtype */
export function kvDtypeBytes(dtype: KvDtype): number {
  const bits = quantData().kvDtypeBits[dtype]
  if (bits === undefined) {
    throw new UnknownEntityError('kv dtype', dtype, Object.keys(quantData().kvDtypeBits).filter((k) => k[0] !== '_'))
  }
  return bits / 8
}

/**
 * KV bytes for ONE token in ONE layer, dispatched on the layer's attention kind.
 *
 * MHA / GQA / MQA / sliding-window (a windowed GQA layer stores the same per-token bytes,
 * it just stores fewer tokens):
 * \\[ b_{tok} = 2\\, n_{kv}\\, d_h\\, s \\]
 * MLA (DeepSeek V2/V3) caches one compressed latent plus the decoupled RoPE key —
 * there is no factor of 2 and no \\(n_{kv}\\):
 * \\[ b_{tok} = (r_{kv} + d_{rope})\\, s \\]
 * Mamba / linear-attention layers hold a fixed state per *sequence*, so their per-token cost is 0.
 *
 * @see docs/MATH.md#kv-per-token
 */
export function kvBytesPerTokenPerLayer(layer: LayerSpec, model: ModelSpec, dtypeBytes: number): number {
  switch (layer.kind) {
    case 'mla': {
      if (!model.mla) throw new Error('layer kind "mla" but model has no mla spec')
      return (model.mla.kvLoraRank + model.mla.qkRopeHeadDim) * dtypeBytes
    }
    case 'mamba':
    case 'linear':
      return 0
    default:
      return 2 * model.numKeyValueHeads * model.headDim * dtypeBytes
  }
}

/**
 * Fixed recurrent state for one sequence in one SSM/linear layer:
 * \\[ b_{seq} = (d_{inner} d_{state} + d_{conv} d_{inner})\\, s \\]
 * @see docs/MATH.md#ssm-state
 */
export function ssmStateBytesPerSequencePerLayer(model: ModelSpec, dtypeBytes: number): number {
  if (!model.ssm) return 0
  const { dInner, dState, dConv } = model.ssm
  return (dInner * dState + dConv * dInner) * dtypeBytes
}

const roundUp = (n: number, block: number) => Math.ceil(n / block) * block

/**
 * How many ways the KV cache is actually split across tensor-parallel ranks.
 *
 * \[ \text{shards} = \begin{cases} 1 & \text{MLA (latent is replicated)} \\
 *    \min(TP, n_{kv}) & \text{otherwise} \end{cases} \]
 *
 * Multiply a per-device figure by this to recover the whole cache — which is what crosses
 * the wire in a disaggregated deployment.
 *
 * @see docs/MATH.md#kv-cache
 */
export function kvShards(model: ModelSpec, tp: number): number {
  return model.mla ? 1 : Math.min(Math.max(1, tp), model.numKeyValueHeads)
}

/**
 * Tokens actually resident for one layer, after paged-attention round-up and prefix sharing.
 *
 * Paged allocation charges whole blocks: \\( \\lceil \\ell / B \\rceil B \\) per sequence.
 * With a prefix cache of hit rate \\(h\\) over a shared prefix of \\(p\\) tokens, the prefix is
 * stored once and each sequence pays only for what it does not share:
 * \\[ T = \\lceil p/B \\rceil B + \\sum_i \\lceil (\\ell_i - h p)/B \\rceil B \\]
 * A sliding-window layer caps \\(\\ell\\) (and \\(p\\)) at the window \\(W\\), and no sequence can
 * exceed `max_model_len` — a server rejects longer requests rather than caching them.
 *
 * @see docs/MATH.md#kv-token-count
 */
export function effectiveTokensForLayer(w: Workload, blockSize: number, windowSize?: number): number {
  const cap = windowSize ?? Number.POSITIVE_INFINITY
  const perSeq = Math.min(w.avgSeqLen, w.maxModelLen, cap)
  const pc = w.prefixCache
  if (!pc?.enabled || pc.sharedPrefixTokens <= 0 || pc.hitRate <= 0) {
    return w.concurrency * roundUp(perSeq, blockSize)
  }
  const p = Math.min(pc.sharedPrefixTokens, cap, perSeq)
  const unique = Math.max(0, perSeq - pc.hitRate * p)
  return roundUp(p, blockSize) + w.concurrency * roundUp(unique, blockSize)
}

/**
 * Total KV bytes that must fit on ONE device.
 *
 * \\[ B_{KV} = \\frac{1}{\\min(TP, n_{kv})} \\sum_{\\ell \\in \\text{stage}} b_{tok}(\\ell)\\, T(\\ell) \\]
 *
 * Two footguns are surfaced as warnings rather than silently absorbed:
 * - **\\(TP > n_{kv}\\)**: KV heads replicate across ranks, so per-GPU KV stops shrinking.
 *   Going from TP=8 to TP=16 on an 8-KV-head model buys zero KV headroom.
 * - **MLA under TP**: the compressed latent is not head-sharded, so every rank holds the
 *   full KV cache. TP buys weight capacity, not KV capacity.
 *
 * @see docs/MATH.md#kv-cache
 */
export function kvCacheBytes(
  model: ModelSpec,
  workload: Workload,
  parallel: Parallelism,
  opts: { kvDtype: KvDtype; blockSize: number },
): KvBreakdown {
  const s = kvDtypeBytes(opts.kvDtype)
  const warnings: Warning[] = []
  const tp = Math.max(1, parallel.tp)
  const pp = Math.max(1, parallel.pp)

  const shards = kvShards(model, tp)
  if (model.mla && tp > 1) {
    warnings.push({
      code: 'mla_kv_replicated',
      message: `MLA caches a compressed latent that is not head-sharded: all ${tp} ranks hold the full KV cache. TP adds weight capacity, not KV capacity.`,
    })
  } else if (tp > model.numKeyValueHeads) {
    warnings.push({
      code: 'tp_exceeds_kv_heads',
      message: `TP=${tp} exceeds num_key_value_heads=${model.numKeyValueHeads}: KV heads replicate, so per-GPU KV stays at 1/${model.numKeyValueHeads} of the total no matter how far you scale TP.`,
    })
  }

  const stage = layersOnStage(model, pp, 0)
  let perTokenBytesPerDevice = 0
  let totalBytes = 0
  let ssmStateBytesPerDevice = 0
  for (const layer of stage) {
    const perToken = kvBytesPerTokenPerLayer(layer, model, s)
    perTokenBytesPerDevice += perToken / shards
    totalBytes += (perToken * effectiveTokensForLayer(workload, opts.blockSize, layer.windowSize)) / shards
    if (layer.kind === 'mamba' || layer.kind === 'linear') {
      ssmStateBytesPerDevice += (ssmStateBytesPerSequencePerLayer(model, s) * workload.concurrency) / shards
    }
  }

  if (workload.avgSeqLen > workload.maxModelLen) {
    warnings.push({
      code: 'seqlen_exceeds_max_model_len',
      message: `avgSeqLen=${workload.avgSeqLen} exceeds maxModelLen=${workload.maxModelLen}; sequences will be truncated or rejected by the server.`,
    })
  }

  return {
    perTokenBytesPerDevice,
    effectiveTokens: effectiveTokensForLayer(workload, opts.blockSize),
    ssmStateBytesPerDevice,
    totalBytesPerDevice: totalBytes + ssmStateBytesPerDevice,
    warnings,
  }
}

/**
 * KV bytes for a single sequence of `tokens` tokens, on one device — the quantity a
 * serving engine multiplies by its block count. Sliding-window layers cap at their window.
 * @see docs/MATH.md#kv-cache
 */
export function kvBytesForSequence(
  model: ModelSpec, tokens: number, parallel: Parallelism, opts: { kvDtype: KvDtype; blockSize: number },
): number {
  const s = kvDtypeBytes(opts.kvDtype)
  const shards = kvShards(model, parallel.tp)
  let bytes = 0
  for (const layer of layersOnStage(model, Math.max(1, parallel.pp), 0)) {
    const t = roundUp(Math.min(tokens, layer.windowSize ?? tokens), opts.blockSize)
    bytes += (kvBytesPerTokenPerLayer(layer, model, s) * t) / shards
  }
  return bytes
}
