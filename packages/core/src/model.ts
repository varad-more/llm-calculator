import { IncompleteConfigError } from './errors.ts'
import { archDefaults } from './data.ts'
import type { AttentionKind, LayerSpec, ModelSpec, MlpKind, Warning } from './types.ts'

type Raw = Record<string, any>

const REQUIRED = [
  'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads',
  'intermediate_size', 'vocab_size', 'max_position_embeddings', 'tie_word_embeddings',
] as const

/**
 * Normalize a raw HF `config.json` into a {@link ModelSpec}.
 *
 * The layer array is built per layer, never assumed homogeneous: Gemma 3 alternates
 * 5 sliding-window layers to 1 global, DeepSeek replaces the first `first_k_dense_replace`
 * MoE blocks with dense MLPs, and gpt-oss ships an explicit `layer_types` array.
 *
 * Fields absent from the config are taken from `data/arch-defaults.json` (the transformers
 * config-class defaults) and listed in `spec.assumed`; if there is no sourced default we
 * throw {@link IncompleteConfigError} rather than guessing.
 *
 * \\[ d_{head} = \\texttt{config.head\\_dim} \;?\; \\texttt{config.head\\_dim} : d_{model} / n_{heads} \\]
 *
 * @see docs/MATH.md#model-normalization
 */
export function normalizeConfig(raw: Raw, opts: { id?: string; measuredWeightBytes?: number } = {}): ModelSpec {
  const warnings: Warning[] = []
  const text: Raw = raw.text_config ?? raw
  const modelType: string = text.model_type ?? raw.model_type ?? 'unknown'
  if (raw.vision_config) {
    warnings.push({
      code: 'vision_tower_excluded',
      message: `${modelType} is multimodal; the vision tower is excluded from weight and KV estimates.`,
    })
  }

  const assumed: string[] = []
  const defaults = archDefaults()[modelType]?.fields ?? {}
  const field = (name: string, optional = false): any => {
    if (text[name] !== undefined && text[name] !== null) return text[name]
    if (raw[name] !== undefined && raw[name] !== null) return raw[name]
    if (name in defaults) {
      assumed.push(`${name}=${defaults[name]} (transformers default for ${modelType})`)
      return defaults[name]
    }
    if (optional) return undefined
    throw new IncompleteConfigError(name, modelType)
  }

  for (const r of REQUIRED) field(r)

  const numLayers: number = field('num_hidden_layers')
  const hiddenSize: number = field('hidden_size')
  const numAttentionHeads: number = field('num_attention_heads')
  const numKeyValueHeads: number = field('num_key_value_heads')
  const headDim: number = field('head_dim', true) ?? hiddenSize / numAttentionHeads
  const intermediateSize: number = field('intermediate_size')
  const vocabSize: number = field('vocab_size')
  const tieWordEmbeddings: boolean = field('tie_word_embeddings')
  const maxPositionEmbeddings: number = field('max_position_embeddings')

  const mla = text.kv_lora_rank
    ? {
        kvLoraRank: text.kv_lora_rank,
        qkRopeHeadDim: field('qk_rope_head_dim'),
        qkNopeHeadDim: field('qk_nope_head_dim'),
        vHeadDim: field('v_head_dim'),
        qLoraRank: text.q_lora_rank ?? null,
      }
    : undefined

  const numExperts = text.n_routed_experts ?? text.num_local_experts ?? text.num_experts
  const moe = numExperts
    ? {
        numExperts,
        topK: text.num_experts_per_tok ?? text.experts_per_token ?? 1,
        expertIntermediate: text.moe_intermediate_size ?? intermediateSize,
        sharedExperts: text.n_shared_experts ?? 0,
      }
    : undefined

  const ssm = text.mamba_d_state
    ? {
        dInner: text.mamba_expand ? text.mamba_expand * hiddenSize : (text.mamba_d_inner ?? 2 * hiddenSize),
        dState: text.mamba_d_state,
        dConv: text.mamba_d_conv ?? 4,
      }
    : undefined

  const baseKind: AttentionKind = mla
    ? 'mla'
    : numKeyValueHeads === numAttentionHeads ? 'mha'
    : numKeyValueHeads === 1 ? 'mqa'
    : 'gqa'

  const layers = buildLayers({
    numLayers, text, baseKind, moe: !!moe, assumed, defaults, modelType, maxPositionEmbeddings,
  })

  return {
    id: opts.id ?? modelType,
    numLayers, layers, hiddenSize, numAttentionHeads, numKeyValueHeads, headDim,
    intermediateSize, vocabSize, tieWordEmbeddings, maxPositionEmbeddings,
    moe, mla, ssm, assumed, warnings,
    mtpLayers: text.num_nextn_predict_layers || undefined,
    unquantizedClasses: unquantizedClasses(raw.quantization_config?.modules_to_not_convert),
    measuredWeightBytes: opts.measuredWeightBytes,
    checkpointQuant: raw.quantization_config?.quant_method,
  }
}

/**
 * Map a checkpoint's `modules_to_not_convert` globs onto our tensor classes, so a
 * partially-quantized checkpoint (gpt-oss keeps attention, router, embeddings and
 * lm_head in bf16 while the experts are MXFP4) is sized correctly.
 * @see docs/MATH.md#weight-bytes
 */
function unquantizedClasses(modules?: string[]): string[] | undefined {
  if (!modules?.length) return undefined
  const out = new Set<string>(['norm'])
  for (const m of modules) {
    if (m.includes('attn')) out.add('attn')
    if (m.includes('router') || m.includes('gate')) out.add('router')
    if (m.includes('embed')) out.add('embedding')
    if (m.includes('lm_head')) out.add('lm_head')
    if (m.includes('mlp') && !m.includes('router')) out.add('mlp')
  }
  return [...out]
}

/**
 * Per-layer attention/MLP dispatch table.
 *
 * Precedence: an explicit `layer_types` array wins; then a `sliding_window_pattern`
 * (layer \\(i\\) is global iff \\((i+1) \\bmod P = 0\\), matching transformers'
 * `Gemma3` and `Gemma2` implementations); then a global `sliding_window` that is
 * actually shorter than the context; otherwise all layers take the base kind.
 *
 * @see docs/MATH.md#per-layer-dispatch
 */
function buildLayers(a: {
  numLayers: number; text: Raw; baseKind: AttentionKind; moe: boolean
  assumed: string[]; defaults: Raw; modelType: string; maxPositionEmbeddings: number
}): LayerSpec[] {
  const { numLayers, text, baseKind } = a
  const window: number | undefined = text.sliding_window ?? undefined
  const windowUsed = text.use_sliding_window === false ? undefined
    : window && window < a.maxPositionEmbeddings ? window : undefined
  let pattern: number | undefined = text.sliding_window_pattern
  if (pattern === undefined && a.defaults.sliding_window_pattern !== undefined && windowUsed) {
    pattern = a.defaults.sliding_window_pattern
    a.assumed.push(`sliding_window_pattern=${pattern} (transformers default for ${a.modelType})`)
  }
  const explicit: string[] | undefined = text.layer_types

  const attnAt = (i: number): { kind: AttentionKind; windowSize?: number } => {
    if (explicit?.[i]) {
      const t = explicit[i]!
      if (t.includes('sliding')) return { kind: 'sliding_window', windowSize: windowUsed ?? window ?? 0 }
      if (t.includes('mamba')) return { kind: 'mamba' }
      if (t.includes('linear') || t.includes('delta')) return { kind: 'linear' }
      return { kind: baseKind }
    }
    if (windowUsed && pattern) {
      return (i + 1) % pattern === 0 ? { kind: baseKind } : { kind: 'sliding_window', windowSize: windowUsed }
    }
    if (windowUsed) return { kind: 'sliding_window', windowSize: windowUsed }
    return { kind: baseKind }
  }

  const denseFirstK: number = text.first_k_dense_replace ?? 0
  const mlpOnly: number[] = text.mlp_only_layers ?? []
  const sparseStep: number = text.decoder_sparse_step ?? 1
  const mlpAt = (i: number): MlpKind => {
    if (!a.moe) return 'dense'
    if (i < denseFirstK) return 'dense'
    if (mlpOnly.includes(i)) return 'dense'
    if (sparseStep > 1 && i % sparseStep !== 0) return 'dense'
    return 'moe'
  }

  return Array.from({ length: numLayers }, (_, i) => {
    const { kind, windowSize } = attnAt(i)
    const layer: LayerSpec = { kind, mlp: kind === 'mamba' || kind === 'linear' ? mlpAt(i) : mlpAt(i) }
    if (windowSize) layer.windowSize = windowSize
    return layer
  })
}

/** Count of layers on one pipeline stage: \\( \\lceil L / PP \\rceil \\). @see docs/MATH.md#pipeline-parallel */
export function layersPerStage(numLayers: number, pp: number): number {
  return Math.ceil(numLayers / pp)
}

/** The layers resident on pipeline stage `stage` (0-indexed); the widest stage sets the memory budget. @see docs/MATH.md#pipeline-parallel */
export function layersOnStage(model: ModelSpec, pp: number, stage = 0): LayerSpec[] {
  const per = layersPerStage(model.numLayers, pp)
  return model.layers.slice(stage * per, Math.min((stage + 1) * per, model.numLayers))
}
