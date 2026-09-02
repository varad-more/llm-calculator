"""Model normalization. Mirrors packages/core/src/model.ts expression for expression;
the shared golden fixtures in fixtures/golden/ fail if the two ever drift."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._generated import DATA

REQUIRED = [
    "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
    "intermediate_size", "vocab_size", "max_position_embeddings", "tie_word_embeddings",
]


class IncompleteConfigError(Exception):
    """A HF config omits a field we refuse to guess. See docs/MATH.md#model-normalization."""

    def __init__(self, field_name: str, model_type: str):
        self.field = field_name
        self.model_type = model_type
        super().__init__(
            'IncompleteConfigError: missing "%s" (model_type=%s). '
            "Add it to data/arch-defaults.json with a source_url, or pass it explicitly."
            % (field_name, model_type)
        )


class UnknownEntityError(Exception):
    """An id is not in the shipped data."""

    def __init__(self, kind: str, ident: str, known):
        super().__init__('Unknown %s "%s". Known: %s' % (kind, ident, ", ".join(known)))


@dataclass
class LayerSpec:
    kind: str
    mlp: str
    window_size: Optional[int] = None


@dataclass
class ModelSpec:
    id: str
    num_layers: int
    layers: List[LayerSpec]
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    moe: Optional[Dict[str, int]] = None
    mla: Optional[Dict[str, Any]] = None
    ssm: Optional[Dict[str, int]] = None
    mtp_layers: Optional[int] = None
    unquantized_classes: Optional[List[str]] = None
    assumed: List[str] = field(default_factory=list)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    measured_weight_bytes: Optional[int] = None
    checkpoint_quant: Optional[str] = None


def _unquantized_classes(modules):
    """Map a checkpoint's modules_to_not_convert globs onto tensor classes."""
    if not modules:
        return None
    out = {"norm"}
    for m in modules:
        if "attn" in m:
            out.add("attn")
        if "router" in m or "gate" in m:
            out.add("router")
        if "embed" in m:
            out.add("embedding")
        if "lm_head" in m:
            out.add("lm_head")
        if "mlp" in m and "router" not in m:
            out.add("mlp")
    return sorted(out)


def normalize_config(raw: Dict[str, Any], model_id=None, measured_weight_bytes=None) -> ModelSpec:
    """Raw HF config.json -> ModelSpec. See docs/MATH.md#model-normalization."""
    warnings: List[Dict[str, str]] = []
    text = raw.get("text_config") or raw
    model_type = text.get("model_type") or raw.get("model_type") or "unknown"
    if raw.get("vision_config"):
        warnings.append({
            "code": "vision_tower_excluded",
            "message": "%s is multimodal; the vision tower is excluded from weight and KV estimates." % model_type,
        })

    assumed: List[str] = []
    defaults = (DATA["archDefaults"].get(model_type) or {}).get("fields") or {}

    def get(name, optional=False):
        if text.get(name) is not None:
            return text[name]
        if raw.get(name) is not None:
            return raw[name]
        if name in defaults:
            assumed.append("%s=%s (transformers default for %s)" % (name, _js(defaults[name]), model_type))
            return defaults[name]
        if optional:
            return None
        raise IncompleteConfigError(name, model_type)

    for r in REQUIRED:
        get(r)

    num_layers = get("num_hidden_layers")
    hidden = get("hidden_size")
    n_heads = get("num_attention_heads")
    n_kv = get("num_key_value_heads")
    head_dim = get("head_dim", True)
    if head_dim is None:
        head_dim = hidden / n_heads
    intermediate = get("intermediate_size")
    vocab = get("vocab_size")
    tie = get("tie_word_embeddings")
    max_pos = get("max_position_embeddings")

    mla = None
    if text.get("kv_lora_rank"):
        mla = {
            "kv_lora_rank": text["kv_lora_rank"],
            "qk_rope_head_dim": get("qk_rope_head_dim"),
            "qk_nope_head_dim": get("qk_nope_head_dim"),
            "v_head_dim": get("v_head_dim"),
            "q_lora_rank": text.get("q_lora_rank"),
        }

    num_experts = text.get("n_routed_experts") or text.get("num_local_experts") or text.get("num_experts")
    moe = None
    if num_experts:
        moe = {
            "num_experts": num_experts,
            "top_k": text.get("num_experts_per_tok") or text.get("experts_per_token") or 1,
            "expert_intermediate": text.get("moe_intermediate_size") or intermediate,
            "shared_experts": text.get("n_shared_experts") or 0,
        }

    ssm = None
    if text.get("mamba_d_state"):
        d_inner = text["mamba_expand"] * hidden if text.get("mamba_expand") else (text.get("mamba_d_inner") or 2 * hidden)
        ssm = {"d_inner": d_inner, "d_state": text["mamba_d_state"], "d_conv": text.get("mamba_d_conv") or 4}

    if mla:
        base_kind = "mla"
    elif n_kv == n_heads:
        base_kind = "mha"
    elif n_kv == 1:
        base_kind = "mqa"
    else:
        base_kind = "gqa"

    layers = _build_layers(num_layers, text, base_kind, moe is not None, assumed, defaults, model_type, max_pos)

    return ModelSpec(
        id=model_id or model_type, num_layers=num_layers, layers=layers, hidden_size=hidden,
        num_attention_heads=n_heads, num_key_value_heads=n_kv, head_dim=head_dim,
        intermediate_size=intermediate, vocab_size=vocab, tie_word_embeddings=tie,
        max_position_embeddings=max_pos, moe=moe, mla=mla, ssm=ssm,
        mtp_layers=text.get("num_nextn_predict_layers") or None,
        unquantized_classes=_unquantized_classes((raw.get("quantization_config") or {}).get("modules_to_not_convert")),
        assumed=assumed, warnings=warnings, measured_weight_bytes=measured_weight_bytes,
        checkpoint_quant=(raw.get("quantization_config") or {}).get("quant_method"),
    )


def _js(v):
    """Render a value the way JSON/JS would, so `assumed` strings match the TS port."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    return v


def _build_layers(num_layers, text, base_kind, has_moe, assumed, defaults, model_type, max_pos):
    """Per-layer attention/MLP dispatch. See docs/MATH.md#per-layer-dispatch."""
    window = text.get("sliding_window")
    if text.get("use_sliding_window") is False:
        window_used = None
    elif window and window < max_pos:
        window_used = window
    else:
        window_used = None

    pattern = text.get("sliding_window_pattern")
    if pattern is None and defaults.get("sliding_window_pattern") is not None and window_used:
        pattern = defaults["sliding_window_pattern"]
        assumed.append("sliding_window_pattern=%s (transformers default for %s)" % (pattern, model_type))
    explicit = text.get("layer_types")

    def attn_at(i):
        if explicit and i < len(explicit) and explicit[i]:
            t = explicit[i]
            if "sliding" in t:
                return "sliding_window", (window_used if window_used is not None else (window or 0))
            if "mamba" in t:
                return "mamba", None
            if "linear" in t or "delta" in t:
                return "linear", None
            return base_kind, None
        if window_used and pattern:
            if (i + 1) % pattern == 0:
                return base_kind, None
            return "sliding_window", window_used
        if window_used:
            return "sliding_window", window_used
        return base_kind, None

    dense_first_k = text.get("first_k_dense_replace") or 0
    mlp_only = text.get("mlp_only_layers") or []
    sparse_step = text.get("decoder_sparse_step") or 1

    def mlp_at(i):
        if not has_moe:
            return "dense"
        if i < dense_first_k:
            return "dense"
        if i in mlp_only:
            return "dense"
        if sparse_step > 1 and i % sparse_step != 0:
            return "dense"
        return "moe"

    layers = []
    for i in range(num_layers):
        kind, win = attn_at(i)
        layers.append(LayerSpec(kind=kind, mlp=mlp_at(i), window_size=win if win else None))
    return layers


def layers_per_stage(num_layers: int, pp: int) -> int:
    """ceil(L / PP). See docs/MATH.md#pipeline-parallel."""
    import math
    return math.ceil(num_layers / pp)


def layers_on_stage(model: ModelSpec, pp: int, stage: int = 0) -> List[LayerSpec]:
    """Layers resident on one pipeline stage. See docs/MATH.md#pipeline-parallel."""
    per = layers_per_stage(model.num_layers, pp)
    return model.layers[stage * per: min((stage + 1) * per, model.num_layers)]
