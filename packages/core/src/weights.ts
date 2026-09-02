import { quantData } from './data.ts'
import { UnknownEntityError } from './errors.ts'
import type { ModelSpec, QuantScheme, Warning } from './types.ts'

export type TensorClass = 'embedding' | 'lm_head' | 'norm' | 'attn' | 'mlp' | 'router'
export type ParamCounts = Record<TensorClass, number> & { total: number }

export interface WeightResult {
  totalBytes: number
  bytesByClass: Record<TensorClass, number>
  params: ParamCounts
  /** 'measured' = read from model.safetensors.index.json; 'derived' = counted analytically. */
  method: 'measured' | 'derived'
  warnings: Warning[]
}

/**
 * Analytic parameter count, by tensor class.
 *
 * Per layer (GQA):
 * \\[ P_{attn} = d\\,(n_h d_h) + 2\\,d\\,(n_{kv} d_h) + (n_h d_h)\\,d \\]
 * Per layer (MLA, DeepSeek V2/V3), with optional query LoRA of rank \\(r_q\\):
 * \\[ P_{attn} = \\underbrace{d r_q + r_q n_h (d_{nope}+d_{rope})}_{q} + \\underbrace{d (r_{kv}+d_{rope})}_{kv_a}
 *   + \\underbrace{r_{kv} n_h (d_{nope}+d_v)}_{kv_b} + n_h d_v d \\]
 * Gated MLP: \\( P_{mlp} = 3 d d_{ff} \\).
 * MoE layer: \\( P_{moe} = E \\cdot 3 d d_{ff}^{exp} + d E + S \\cdot 3 d d_{ff}^{exp} \\) — **all**
 * experts are resident; \\(k\\) (top-k) only affects FLOPs, never memory.
 *
 * @see docs/MATH.md#weight-parameters
 */
export function parameterCounts(model: ModelSpec): ParamCounts {
  const { hiddenSize: d, numAttentionHeads: nh, numKeyValueHeads: nkv, headDim: dh } = model
  const p: ParamCounts = { embedding: 0, lm_head: 0, norm: 0, attn: 0, mlp: 0, router: 0, total: 0 }

  p.embedding = model.vocabSize * d
  p.lm_head = model.tieWordEmbeddings ? 0 : model.vocabSize * d
  p.norm = d // final norm

  for (const layer of model.layers) {
    if (model.mla && layer.kind === 'mla') {
      const { kvLoraRank: rkv, qkRopeHeadDim: dr, qkNopeHeadDim: dn, vHeadDim: dv, qLoraRank: rq } = model.mla
      const q = rq === null ? d * nh * (dn + dr) : d * rq + rq * nh * (dn + dr)
      p.attn += q + d * (rkv + dr) + rkv * nh * (dn + dv) + nh * dv * d
    } else if (layer.kind === 'mamba' || layer.kind === 'linear') {
      // Mixer params are architecture-specific; counted as 0 and flagged by weightBytes().
    } else {
      p.attn += d * (nh * dh) + 2 * d * (nkv * dh) + nh * dh * d
    }

    if (layer.mlp === 'moe' && model.moe) {
      const { numExperts: E, expertIntermediate: dff, sharedExperts: S } = model.moe
      p.mlp += (E + S) * 3 * d * dff
      p.router += d * E
    } else if (layer.mlp === 'dense') {
      p.mlp += 3 * d * model.intermediateSize
    }
    p.norm += 2 * d
  }

  // Multi-Token-Prediction modules (DeepSeek V3/R1): each is a full transformer
  // layer plus its own copy of the embedding and output head, and an eh_proj of
  // shape (2d, d). They live in the checkpoint but not in num_hidden_layers.
  for (let i = 0; i < (model.mtpLayers ?? 0); i++) {
    const last = model.layers[model.layers.length - 1]!
    const perLayerAttn = p.attn / Math.max(1, model.layers.filter((l) => l.kind !== 'mamba' && l.kind !== 'linear').length)
    p.attn += perLayerAttn
    if (last.mlp === 'moe' && model.moe) {
      p.mlp += (model.moe.numExperts + model.moe.sharedExperts) * 3 * d * model.moe.expertIntermediate
      p.router += d * model.moe.numExperts
    } else {
      p.mlp += 3 * d * model.intermediateSize
    }
    p.embedding += model.vocabSize * d
    p.lm_head += model.vocabSize * d
    p.attn += 2 * d * d // eh_proj
    p.norm += 4 * d
  }

  p.total = p.embedding + p.lm_head + p.norm + p.attn + p.mlp + p.router
  return p
}

/**
 * Bits used to store one weight of `cls`. The scheme's own tensorPolicy applies, plus
 * any classes the specific checkpoint declares it did not convert.
 */
function bitsFor(scheme: any, cls: TensorClass, unquantized?: string[]): number {
  if (unquantized?.includes(cls)) return 16
  return scheme.tensorPolicy?.[cls] ?? scheme.bits
}

/**
 * Storage bytes for the weights.
 *
 * Derived path, per tensor class \\(c\\):
 * \\[ B_c = P_c \\frac{b_c}{8} + \\underbrace{\\left\\lceil \\frac{P_c}{g} \\right\\rceil \\frac{b_{scale}+b_{zero}}{8}}_{\\text{grouped schemes only}} \\]
 * GGUF path uses a **measured** bits-per-weight for the whole file (Q4\\_K\\_M = 4.85 bpw, not 4),
 * because k-quants mix block formats across tensors.
 *
 * The measured path (`model.safetensors.index.json` → `metadata.total_size`) is preferred when
 * the requested scheme matches the checkpoint's own, since it sidesteps param counting entirely.
 *
 * @see docs/MATH.md#weight-bytes
 */
export function weightBytes(
  model: ModelSpec,
  opts: { quant: QuantScheme; preferMeasured?: boolean } ,
): WeightResult {
  const q = quantData()
  const params = parameterCounts(model)
  const warnings: Warning[] = []
  if (model.layers.some((l) => l.kind === 'mamba' || l.kind === 'linear')) {
    warnings.push({ code: 'ssm_weights_unmodelled', message: 'SSM/linear-attention mixer weights are not counted; weight total is a lower bound.' })
  }

  const zero: Record<TensorClass, number> = { embedding: 0, lm_head: 0, norm: 0, attn: 0, mlp: 0, router: 0 }

  if (opts.quant.startsWith('gguf:')) {
    const name = opts.quant.slice(5)
    const bpw = q.gguf.bpw[name]
    if (bpw === undefined) throw new UnknownEntityError('gguf quant', name, Object.keys(q.gguf.bpw))
    const totalBytes = (params.total * bpw) / 8
    return { totalBytes, bytesByClass: { ...zero, mlp: totalBytes }, params, method: 'derived', warnings }
  }

  const scheme = q.schemes[opts.quant]
  if (!scheme) throw new UnknownEntityError('quant scheme', opts.quant, Object.keys(q.schemes).concat('gguf:<name>'))

  const unquantized = model.checkpointQuant === opts.quant ? model.unquantizedClasses : undefined
  const bytesByClass = { ...zero }
  for (const cls of Object.keys(zero) as TensorClass[]) {
    const bits = bitsFor(scheme, cls, unquantized)
    let bytes = (params[cls] * bits) / 8
    const quantized = scheme.tensorPolicy?.[cls] === undefined && !unquantized?.includes(cls)
    if (scheme.kind === 'grouped' && quantized && params[cls] > 0) {
      bytes += Math.ceil(params[cls] / scheme.groupSize) * ((scheme.scaleBits + scheme.zeroBits) / 8)
    }
    bytesByClass[cls] = bytes
  }
  const derived = Object.values(bytesByClass).reduce((a, b) => a + b, 0)

  const hasUnmodelledTensors = model.warnings.some((w) => w.code === 'vision_tower_excluded')
  const matchesCheckpoint =
    !hasUnmodelledTensors &&
    model.measuredWeightBytes !== undefined &&
    (model.checkpointQuant === undefined
      ? opts.quant === 'bf16' || opts.quant === 'fp16'
      : opts.quant.startsWith(model.checkpointQuant.replace('compressed-tensors', '')))
  if (hasUnmodelledTensors && model.measuredWeightBytes !== undefined) {
    warnings.push({
      code: 'measured_weights_unusable',
      message: 'metadata.total_size covers the vision tower too, so the derived text-only count is used instead.',
    })
  }
  // A checkpoint's own metadata can be wrong (DeepSeek-V3's safetensors index reports
  // 1369 GB for files that sum to 688 GB). Trust the measurement only when it agrees with
  // the analytic count; otherwise say so and use the count.
  const disagrees = matchesCheckpoint && Math.abs(model.measuredWeightBytes! - derived) / derived > 0.05
  if (disagrees) {
    warnings.push({
      code: 'measured_weights_disagree',
      message: `Checkpoint metadata says ${(model.measuredWeightBytes! / 2 ** 30).toFixed(1)} GiB but the parameter count gives ${(derived / 2 ** 30).toFixed(1)} GiB (${((model.measuredWeightBytes! / derived - 1) * 100).toFixed(0)}% apart). Using the derived figure.`,
    })
  }
  if (opts.preferMeasured !== false && matchesCheckpoint && !disagrees) {
    const measured = model.measuredWeightBytes!
    const scale = measured / derived
    return {
      totalBytes: measured,
      bytesByClass: Object.fromEntries(
        Object.entries(bytesByClass).map(([k, v]) => [k, v * scale]),
      ) as Record<TensorClass, number>,
      params, method: 'measured', warnings,
    }
  }
  return { totalBytes: derived, bytesByClass, params, method: 'derived', warnings }
}

/**
 * Parameters actually multiplied per token: all of a dense model, but only the
 * top-\\(k\\) experts (plus shared experts) of an MoE layer.
 * \\[ P_{active} = P_{total} - \\sum_{\\ell \\in MoE} (E - k - S)\\, 3 d d_{ff}^{exp} \\]
 * @see docs/MATH.md#active-parameters
 */
export function activeParameters(model: ModelSpec): number {
  const p = parameterCounts(model)
  if (!model.moe) return p.total
  const { numExperts: E, topK: k, expertIntermediate: dff, sharedExperts: S } = model.moe
  const moeLayers = model.layers.filter((l) => l.mlp === 'moe').length
  const inactive = moeLayers * Math.max(0, E - k) * 3 * model.hiddenSize * dff
  return p.total - inactive
}

/**
 * Weight bytes resident on one device. Linear and embedding tensors are sharded by
 * tensor parallelism and layers are split by pipeline parallelism; norms are replicated
 * but are ~0.01% of the total.
 * \\[ B_{dev} = B_{total} / (TP \\cdot PP) \\]
 * @see docs/MATH.md#weight-sharding
 */
export function weightBytesPerDevice(totalBytes: number, parallel: { tp: number; pp: number }): number {
  return totalBytes / (Math.max(1, parallel.tp) * Math.max(1, parallel.pp))
}
