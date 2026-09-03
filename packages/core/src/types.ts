// All memory quantities in this package are BYTES. Formatting to GiB happens
// only in packages/core/src/format.ts, at the presentation edge.

export type AttentionKind = 'mha' | 'gqa' | 'mqa' | 'mla' | 'sliding_window' | 'linear' | 'mamba'
export type MlpKind = 'dense' | 'moe' | 'none'

export interface LayerSpec {
  kind: AttentionKind
  /** Attention window in tokens. Only set for kind === 'sliding_window'. */
  windowSize?: number
  mlp: MlpKind
}

export interface MoeSpec {
  numExperts: number
  topK: number
  expertIntermediate: number
  sharedExperts: number
}

/** Multi-head Latent Attention (DeepSeek V2/V3). */
export interface MlaSpec {
  kvLoraRank: number
  qkRopeHeadDim: number
  qkNopeHeadDim: number
  vHeadDim: number
  /** null when q is a single dense projection (DeepSeek-V2-Lite). */
  qLoraRank: number | null
}

/** State-space / linear-attention layers: state is per SEQUENCE, not per token. */
export interface SsmSpec {
  dInner: number
  dState: number
  dConv: number
}

export interface ModelSpec {
  id: string
  numLayers: number
  layers: LayerSpec[]
  hiddenSize: number
  numAttentionHeads: number
  numKeyValueHeads: number
  headDim: number
  intermediateSize: number
  vocabSize: number
  tieWordEmbeddings: boolean
  maxPositionEmbeddings: number
  moe?: MoeSpec
  mla?: MlaSpec
  ssm?: SsmSpec
  /** Multi-Token-Prediction modules shipped in the checkpoint but not counted in num_hidden_layers (DeepSeek V3/R1). */
  mtpLayers?: number
  /** Tensor classes the checkpoint leaves unquantized (config.quantization_config.modules_to_not_convert). */
  unquantizedClasses?: string[]
  /** Fields not present in config.json, taken from data/arch-defaults.json. Always surfaced. */
  assumed: string[]
  /** Bytes on disk from model.safetensors.index.json, when known. Preferred over derived. */
  measuredWeightBytes?: number
  /** Quant scheme the checkpoint ships in, from config.quantization_config. */
  checkpointQuant?: string
  /** Disclosures raised while normalizing (e.g. vision tower excluded). */
  warnings: Warning[]
}

export interface GpuSpec {
  id: string
  name: string
  vendor: string
  vramBytes: number
  memoryType: string
  memBandwidthBytesPerSec: number
  tflopsDense: Partial<Record<'fp32' | 'tf32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'fp4', number>>
  interconnect: { kind: string; bidirectionalBytesPerSec: number }
  /**
   * Fraction of `vramBytes` the platform keeps out of reach, when the device differs from the
   * usual driver reservation. Apple silicon caps a Metal process at ~75% of unified memory.
   * Omitted means the global `driver_reserved_vram_fraction` assumption applies.
   */
  reservedVramFraction?: number
  sparsity_excluded: true
  notes?: string
  source_url: string
}

export interface Parallelism {
  tp: number
  pp: number
  /** Experts sharded across ranks instead of replicated. */
  ep?: boolean
}

export interface PrefixCache {
  enabled: boolean
  /** 0..1 fraction of the shared prefix already resident when a request arrives. */
  hitRate: number
  sharedPrefixTokens: number
}

export interface Workload {
  /** Concurrent sequences in flight (maps to vLLM --max-num-seqs). */
  concurrency: number
  /** Mean total tokens (prompt + generated) per in-flight sequence. */
  avgSeqLen: number
  /** Longest sequence the server must admit (maps to --max-model-len). */
  maxModelLen: number
  /** Tokens per prefill chunk (vLLM --max-num-batched-tokens). */
  chunkTokens: number
  prefixCache?: PrefixCache
}

export type QuantScheme =
  | 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'int8' | 'awq-int4' | 'gptq-int4' | 'mxfp4'
  | `gguf:${string}`

export type KvDtype = 'fp32' | 'fp16' | 'bf16' | 'fp8' | 'fp8_e5m2' | 'fp8_e4m3' | 'int8'

export interface Assumption {
  value: number
  unit: string
  rationale: string
  source_url: string
  confidence: 'low' | 'medium' | 'high'
}
export type AssumptionKey =
  | 'driver_reserved_vram_fraction' | 'cuda_context_bytes' | 'nccl_buffer_bytes_per_rank'
  | 'cudagraph_base_bytes' | 'cudagraph_bytes_per_captured_size' | 'prefill_activation_multiplier'
  | 'allocator_fragmentation_fraction' | 'mbu_decode' | 'mfu_prefill' | 'quant_group_size'
  | 'interconnect_efficiency'
export type Assumptions = Record<AssumptionKey, Assumption>
/** Numeric overrides supplied by the user (CLI --assume, UI panel). */
export type AssumptionOverrides = Partial<Record<AssumptionKey, number>>

export interface Warning {
  code: string
  message: string
}

export interface MemoryBreakdown {
  weightBytes: number
  kvBytes: number
  activationBytes: number
  overheadBytes: number
  /** Per-device totals; `perDevice` is what must fit in one GPU's VRAM. */
  perDeviceTotalBytes: number
}

export interface KvBreakdown {
  /** Bytes for one token of KV, summed over the layers resident on ONE device. */
  perTokenBytesPerDevice: number
  /** Tokens actually stored after paging round-up and prefix-cache sharing. */
  effectiveTokens: number
  /** Fixed per-sequence SSM/Mamba state on one device (0 for pure transformers). */
  ssmStateBytesPerDevice: number
  totalBytesPerDevice: number
  warnings: Warning[]
}
