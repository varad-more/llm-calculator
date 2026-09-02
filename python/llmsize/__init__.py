"""llmsize — inference sizing and serving-config generation.

A Python port of @llmsize/core that reads the same data/ and passes the same
fixtures/golden/. Import it next to vLLM to size a deployment in-process.

    from llmsize import size
    r = size({"model": "meta-llama/Llama-3.1-70B-Instruct", "gpu": "h100-sxm-80",
              "engine": "vllm", "tp": 4, "context": 32768, "concurrency": 64})
    print(r["plan"]["fits"], r["command"])
"""
from .model import (
    IncompleteConfigError, LayerSpec, ModelSpec, UnknownEntityError,
    layers_on_stage, layers_per_stage, normalize_config,
)
from .memory import (
    active_parameters, assume, captured_graph_sizes, effective_tokens_for_layer, get_gpu,
    kv_bytes_for_sequence, kv_bytes_per_token_per_layer, kv_cache_bytes, kv_dtype_bytes,
    logits_bytes, overhead_bytes, parameter_counts, prefill_activation_bytes,
    resolve_assumptions, ssm_state_bytes_per_sequence_per_layer, weight_bytes, weight_bytes_per_device,
)
from .engines import (
    ENGINES, all_reduce_seconds, causal_attention_flops, emit_flags, estimate_throughput,
    feasible, plan_llamacpp, plan_sglang, plan_trtllm, plan_vllm, pool_tokens,
    resolve_model, size, solve_largest_fit, usable_vram, validation_for,
)
from ._generated import DATA

__version__ = "0.1.0"

GiB = 2 ** 30


def gib(byte_count: float, digits: int = 2) -> str:
    """Bytes -> '12.34 GiB'. See docs/MATH.md#units."""
    return "%.*f GiB" % (digits, byte_count / GiB)


def list_gpus():
    """Every GPU in data/gpus.json."""
    return DATA["gpus"]


def list_models():
    """Snapshotted HF model ids available offline."""
    return sorted(DATA["models"])
