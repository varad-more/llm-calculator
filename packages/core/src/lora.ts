import { layersOnStage } from './model.ts'
import type { ModelSpec, Parallelism } from './types.ts'

export interface LoraConfig {
  /** Adapter slots kept resident on the GPU (vLLM --max-loras). */
  maxLoras: number
  /** Largest rank any resident adapter may have (vLLM --max-lora-rank). */
  maxLoraRank: number
  /** Which projections the adapters target. vLLM's default set is all of them. */
  targets?: ('qkv' | 'o' | 'gate_up' | 'down')[]
  /** Adapter weight dtype width in bytes. */
  dtypeBytes?: number
}

export interface LoraSizing {
  bytesPerAdapter: number
  totalBytesPerDevice: number
  paramsPerAdapter: number
  flags: string[]
}

/**
 * Resident memory for multi-LoRA serving.
 *
 * A rank-\(r\) adapter on a projection \(W \in \mathbb{R}^{m \times n}\) stores \(A \in
 * \mathbb{R}^{m \times r}\) and \(B \in \mathbb{R}^{r \times n}\), so \(r(m+n)\) parameters —
 * not \(r \cdot m \cdot n\). Per layer, over vLLM's default target set:
 *
 * \\[ P_{lora} = r\\big[\\underbrace{(d + n_h d_h) + 2(d + n_{kv} d_h)}_{qkv} + \\underbrace{(n_h d_h + d)}_{o}
 *   + \\underbrace{2(d + d_{ff})}_{gate,up} + \\underbrace{(d_{ff} + d)}_{down}\\big] \\]
 *
 * Every slot is sized for `maxLoraRank` whether or not the loaded adapter uses it — vLLM
 * pre-allocates uniform slots — so the resident cost is \(S \cdot P_{lora} \cdot s\).
 *
 * @see docs/MATH.md#multi-lora
 */
export function loraSizing(model: ModelSpec, cfg: LoraConfig, parallel: Parallelism): LoraSizing {
  const r = cfg.maxLoraRank
  const targets = cfg.targets ?? ['qkv', 'o', 'gate_up', 'down']
  const s = cfg.dtypeBytes ?? 2
  const { hiddenSize: d, numAttentionHeads: nh, numKeyValueHeads: nkv, headDim: dh } = model

  let perLayer = 0
  if (targets.includes('qkv')) perLayer += r * ((d + nh * dh) + 2 * (d + nkv * dh))
  if (targets.includes('o')) perLayer += r * (nh * dh + d)

  const layers = layersOnStage(model, Math.max(1, parallel.pp), 0)
  let params = 0
  for (const layer of layers) {
    let ffn = 0
    const dff = layer.mlp === 'moe' && model.moe ? model.moe.expertIntermediate : model.intermediateSize
    if (targets.includes('gate_up')) ffn += r * 2 * (d + dff)
    if (targets.includes('down')) ffn += r * (dff + d)
    params += perLayer + ffn
  }

  const bytesPerAdapter = params * s
  return {
    bytesPerAdapter,
    paramsPerAdapter: params,
    totalBytesPerDevice: (cfg.maxLoras * bytesPerAdapter) / Math.max(1, parallel.tp),
    flags: ['--enable-lora', `--max-loras ${cfg.maxLoras}`, `--max-lora-rank ${r}`],
  }
}
