"""Weights, KV cache and overhead. Mirrors weights.ts / kv.ts / overhead.ts."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ._generated import DATA
from .model import ModelSpec, UnknownEntityError, layers_on_stage

TENSOR_CLASSES = ["embedding", "lm_head", "norm", "attn", "mlp", "router"]


def resolve_assumptions(overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Merge user overrides over the shipped constants. See docs/MATH.md#overhead."""
    base = DATA["assumptions"]
    overrides = overrides or {}
    for k in overrides:
        if k not in base:
            raise UnknownEntityError("assumption", k, sorted(base))
    out = {}
    for k, v in base.items():
        if k in overrides:
            out[k] = dict(v, value=overrides[k], confidence="high",
                          rationale="user override (was %s: %s)" % (v["value"], v["rationale"]))
        else:
            out[k] = v
    return out


def assume(a: Dict[str, Any], key: str) -> float:
    return a[key]["value"]


def get_gpu(gpu_id: str) -> Dict[str, Any]:
    for g in DATA["gpus"]:
        if g["id"] == gpu_id:
            return g
    raise UnknownEntityError("gpu", gpu_id, [g["id"] for g in DATA["gpus"]])


def parameter_counts(model: ModelSpec) -> Dict[str, float]:
    """Analytic parameter count by tensor class. See docs/MATH.md#weight-parameters."""
    d = model.hidden_size
    nh = model.num_attention_heads
    nkv = model.num_key_value_heads
    dh = model.head_dim
    p = {c: 0 for c in TENSOR_CLASSES}

    p["embedding"] = model.vocab_size * d
    p["lm_head"] = 0 if model.tie_word_embeddings else model.vocab_size * d
    p["norm"] = d

    for layer in model.layers:
        if model.mla and layer.kind == "mla":
            m = model.mla
            rkv, dr, dn, dv, rq = (m["kv_lora_rank"], m["qk_rope_head_dim"],
                                   m["qk_nope_head_dim"], m["v_head_dim"], m["q_lora_rank"])
            q = d * nh * (dn + dr) if rq is None else d * rq + rq * nh * (dn + dr)
            p["attn"] += q + d * (rkv + dr) + rkv * nh * (dn + dv) + nh * dv * d
        elif layer.kind in ("mamba", "linear"):
            pass  # mixer params are architecture-specific; flagged by weight_bytes()
        else:
            p["attn"] += d * (nh * dh) + 2 * d * (nkv * dh) + nh * dh * d

        if layer.mlp == "moe" and model.moe:
            e, dff, s = model.moe["num_experts"], model.moe["expert_intermediate"], model.moe["shared_experts"]
            p["mlp"] += (e + s) * 3 * d * dff
            p["router"] += d * e
        elif layer.mlp == "dense":
            p["mlp"] += 3 * d * model.intermediate_size
        p["norm"] += 2 * d

    for _ in range(model.mtp_layers or 0):
        last = model.layers[-1]
        attn_layers = len([l for l in model.layers if l.kind not in ("mamba", "linear")])
        per_layer_attn = p["attn"] / max(1, attn_layers)
        p["attn"] += per_layer_attn
        if last.mlp == "moe" and model.moe:
            p["mlp"] += (model.moe["num_experts"] + model.moe["shared_experts"]) * 3 * d * model.moe["expert_intermediate"]
            p["router"] += d * model.moe["num_experts"]
        else:
            p["mlp"] += 3 * d * model.intermediate_size
        p["embedding"] += model.vocab_size * d
        p["lm_head"] += model.vocab_size * d
        p["attn"] += 2 * d * d
        p["norm"] += 4 * d

    p["total"] = p["embedding"] + p["lm_head"] + p["norm"] + p["attn"] + p["mlp"] + p["router"]
    return p


def _bits_for(scheme, cls, unquantized):
    if unquantized and cls in unquantized:
        return 16
    return (scheme.get("tensorPolicy") or {}).get(cls, scheme["bits"])


def weight_bytes(model: ModelSpec, quant: str, prefer_measured: bool = True) -> Dict[str, Any]:
    """Storage bytes for the weights. See docs/MATH.md#weight-bytes."""
    q = DATA["quant"]
    params = parameter_counts(model)
    warnings: List[Dict[str, str]] = []
    if any(l.kind in ("mamba", "linear") for l in model.layers):
        warnings.append({"code": "ssm_weights_unmodelled",
                         "message": "SSM/linear-attention mixer weights are not counted; weight total is a lower bound."})
    zero = {c: 0 for c in TENSOR_CLASSES}

    if quant.startswith("gguf:"):
        name = quant[5:]
        bpw = q["gguf"]["bpw"].get(name)
        if bpw is None:
            raise UnknownEntityError("gguf quant", name, sorted(q["gguf"]["bpw"]))
        total = params["total"] * bpw / 8
        by = dict(zero, mlp=total)
        return {"totalBytes": total, "bytesByClass": by, "params": params, "method": "derived", "warnings": warnings}

    scheme = q["schemes"].get(quant)
    if scheme is None:
        raise UnknownEntityError("quant scheme", quant, sorted(q["schemes"]) + ["gguf:<name>"])

    unquantized = model.unquantized_classes if model.checkpoint_quant == quant else None
    by = dict(zero)
    for cls in TENSOR_CLASSES:
        bits = _bits_for(scheme, cls, unquantized)
        b = params[cls] * bits / 8
        quantized = (scheme.get("tensorPolicy") or {}).get(cls) is None and not (unquantized and cls in unquantized)
        if scheme["kind"] == "grouped" and quantized and params[cls] > 0:
            b += math.ceil(params[cls] / scheme["groupSize"]) * ((scheme["scaleBits"] + scheme["zeroBits"]) / 8)
        by[cls] = b
    derived = sum(by[c] for c in TENSOR_CLASSES)

    has_unmodelled = any(w["code"] == "vision_tower_excluded" for w in model.warnings)
    if model.checkpoint_quant is None:
        matches = quant in ("bf16", "fp16")
    else:
        matches = quant.startswith(model.checkpoint_quant.replace("compressed-tensors", ""))
    matches = (not has_unmodelled) and model.measured_weight_bytes is not None and matches

    if has_unmodelled and model.measured_weight_bytes is not None:
        warnings.append({"code": "measured_weights_unusable",
                         "message": "metadata.total_size covers the vision tower too, so the derived text-only count is used instead."})
    disagrees = matches and abs(model.measured_weight_bytes - derived) / derived > 0.05
    if disagrees:
        warnings.append({
            "code": "measured_weights_disagree",
            "message": "Checkpoint metadata says %.1f GiB but the parameter count gives %.1f GiB (%.0f%% apart). Using the derived figure."
                       % (model.measured_weight_bytes / 2 ** 30, derived / 2 ** 30,
                          (model.measured_weight_bytes / derived - 1) * 100),
        })
    if prefer_measured and matches and not disagrees:
        measured = model.measured_weight_bytes
        scale = measured / derived
        return {"totalBytes": measured, "bytesByClass": {c: by[c] * scale for c in TENSOR_CLASSES},
                "params": params, "method": "measured", "warnings": warnings}
    return {"totalBytes": derived, "bytesByClass": by, "params": params, "method": "derived", "warnings": warnings}


def active_parameters(model: ModelSpec) -> float:
    """Parameters multiplied per token. See docs/MATH.md#active-parameters."""
    p = parameter_counts(model)
    if not model.moe:
        return p["total"]
    e, k, dff = model.moe["num_experts"], model.moe["top_k"], model.moe["expert_intermediate"]
    moe_layers = len([l for l in model.layers if l.mlp == "moe"])
    inactive = moe_layers * max(0, e - k) * 3 * model.hidden_size * dff
    return p["total"] - inactive


def weight_bytes_per_device(total_bytes: float, tp: int, pp: int) -> float:
    """See docs/MATH.md#weight-sharding."""
    return total_bytes / (max(1, tp) * max(1, pp))


def kv_dtype_bytes(dtype: str) -> float:
    """See docs/MATH.md#kv-dtype."""
    bits = DATA["quant"]["kvDtypeBits"].get(dtype)
    if bits is None:
        raise UnknownEntityError("kv dtype", dtype, [k for k in DATA["quant"]["kvDtypeBits"] if not k.startswith("_")])
    return bits / 8


def kv_bytes_per_token_per_layer(layer, model: ModelSpec, dtype_bytes: float) -> float:
    """See docs/MATH.md#kv-per-token."""
    if layer.kind == "mla":
        return (model.mla["kv_lora_rank"] + model.mla["qk_rope_head_dim"]) * dtype_bytes
    if layer.kind in ("mamba", "linear"):
        return 0
    return 2 * model.num_key_value_heads * model.head_dim * dtype_bytes


def ssm_state_bytes_per_sequence_per_layer(model: ModelSpec, dtype_bytes: float) -> float:
    """See docs/MATH.md#ssm-state."""
    if not model.ssm:
        return 0
    return (model.ssm["d_inner"] * model.ssm["d_state"] + model.ssm["d_conv"] * model.ssm["d_inner"]) * dtype_bytes


def _round_up(n, block):
    return math.ceil(n / block) * block


def effective_tokens_for_layer(w, block_size: int, window_size=None) -> float:
    """See docs/MATH.md#kv-token-count."""
    cap = window_size if window_size else float("inf")
    per_seq = min(w["avgSeqLen"], w["maxModelLen"], cap)
    pc = w.get("prefixCache")
    if not pc or not pc.get("enabled") or pc.get("sharedPrefixTokens", 0) <= 0 or pc.get("hitRate", 0) <= 0:
        return w["concurrency"] * _round_up(per_seq, block_size)
    p = min(pc["sharedPrefixTokens"], cap, per_seq)
    unique = max(0, per_seq - pc["hitRate"] * p)
    return _round_up(p, block_size) + w["concurrency"] * _round_up(unique, block_size)


def kv_cache_bytes(model: ModelSpec, workload, tp: int, pp: int, kv_dtype: str, block_size: int) -> Dict[str, Any]:
    """See docs/MATH.md#kv-cache."""
    s = kv_dtype_bytes(kv_dtype)
    warnings: List[Dict[str, str]] = []
    tp = max(1, tp)
    pp = max(1, pp)
    shards = 1 if model.mla else min(tp, model.num_key_value_heads)

    if model.mla and tp > 1:
        warnings.append({"code": "mla_kv_replicated",
                         "message": "MLA caches a compressed latent that is not head-sharded: all %d ranks hold the full KV cache. TP adds weight capacity, not KV capacity." % tp})
    elif tp > model.num_key_value_heads:
        warnings.append({"code": "tp_exceeds_kv_heads",
                         "message": "TP=%d exceeds num_key_value_heads=%d: KV heads replicate, so per-GPU KV stays at 1/%d of the total no matter how far you scale TP."
                                    % (tp, model.num_key_value_heads, model.num_key_value_heads)})

    per_token = 0.0
    total = 0.0
    ssm_state = 0.0
    for layer in layers_on_stage(model, pp, 0):
        pt = kv_bytes_per_token_per_layer(layer, model, s)
        per_token += pt / shards
        total += pt * effective_tokens_for_layer(workload, block_size, layer.window_size) / shards
        if layer.kind in ("mamba", "linear"):
            ssm_state += ssm_state_bytes_per_sequence_per_layer(model, s) * workload["concurrency"] / shards

    if workload["avgSeqLen"] > workload["maxModelLen"]:
        warnings.append({"code": "seqlen_exceeds_max_model_len",
                         "message": "avgSeqLen=%s exceeds maxModelLen=%s; sequences will be truncated or rejected by the server."
                                    % (workload["avgSeqLen"], workload["maxModelLen"])})

    return {"perTokenBytesPerDevice": per_token,
            "effectiveTokens": effective_tokens_for_layer(workload, block_size),
            "ssmStateBytesPerDevice": ssm_state,
            "totalBytesPerDevice": total + ssm_state,
            "warnings": warnings}


def kv_bytes_for_sequence(model: ModelSpec, tokens: int, tp: int, pp: int, kv_dtype: str, block_size: int) -> float:
    """See docs/MATH.md#kv-cache."""
    s = kv_dtype_bytes(kv_dtype)
    shards = 1 if model.mla else min(max(1, tp), model.num_key_value_heads)
    b = 0.0
    for layer in layers_on_stage(model, max(1, pp), 0):
        t = _round_up(min(tokens, layer.window_size if layer.window_size else tokens), block_size)
        b += kv_bytes_per_token_per_layer(layer, model, s) * t / shards
    return b


def captured_graph_sizes(max_num_seqs: int) -> int:
    """See docs/MATH.md#cuda-graphs."""
    return 3 + math.floor(min(max_num_seqs, 512) / 8)


def prefill_activation_bytes(model: ModelSpec, workload, tp: int, act_dtype_bytes: int, a) -> float:
    """See docs/MATH.md#activation-peak."""
    k = assume(a, "prefill_activation_multiplier")
    c = max(1, workload["chunkTokens"])
    if model.moe and any(l.mlp == "moe" for l in model.layers):
        dff = model.moe["expert_intermediate"] * min(model.moe["top_k"], model.moe["num_experts"])
    else:
        dff = model.intermediate_size
    return (c * model.hidden_size * act_dtype_bytes * k + 2 * c * dff * act_dtype_bytes) / max(1, tp)


def logits_bytes(model: ModelSpec, workload, all_positions: bool = False) -> float:
    """See docs/MATH.md#logits."""
    n = max(workload["chunkTokens"], workload["concurrency"]) if all_positions else workload["concurrency"]
    return n * model.vocab_size * 4


def overhead_bytes(model: ModelSpec, workload, tp: int, pp: int, cuda_graphs: bool,
                   a, weight_bytes_per_device: float, all_positions: bool = False) -> Dict[str, Any]:
    """See docs/MATH.md#overhead."""
    tp = max(1, tp)
    warnings: List[Dict[str, str]] = []
    cuda_context = assume(a, "cuda_context_bytes")
    comm = assume(a, "nccl_buffer_bytes_per_rank") if (tp > 1 or pp > 1) else 0
    graph = (assume(a, "cudagraph_base_bytes")
             + captured_graph_sizes(workload["concurrency"]) * assume(a, "cudagraph_bytes_per_captured_size")) if cuda_graphs else 0
    activation = prefill_activation_bytes(model, workload, tp, 2, a)
    logits = logits_bytes(model, workload, all_positions)
    frag = assume(a, "allocator_fragmentation_fraction") * (weight_bytes_per_device + activation)
    if all_positions and logits > 2 ** 30:
        warnings.append({"code": "logits_all_positions",
                         "message": "This engine materialises fp32 logits for every prefill position: %.1f GiB at chunk=%s, vocab=%s. Lower --max-num-batched-tokens."
                                    % (logits / 2 ** 30, workload["chunkTokens"], model.vocab_size)})
    non_torch = cuda_context + comm + frag
    return {"cudaContextBytes": cuda_context, "commBytes": comm, "graphBytes": graph,
            "activationBytes": activation, "logitsBytes": logits, "fragmentationBytes": frag,
            "nonTorchBytes": non_torch, "totalBytes": non_torch + graph + activation + logits,
            "warnings": warnings}
