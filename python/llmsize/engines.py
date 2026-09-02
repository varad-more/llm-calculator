"""Engine adapters, throughput roofline and the top-level sizing entry point.
Mirrors engines/*.ts, throughput.ts and plan.ts."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ._generated import DATA
from .model import ModelSpec, UnknownEntityError, normalize_config
from .memory import (
    active_parameters, assume, get_gpu, kv_bytes_for_sequence, kv_cache_bytes,
    overhead_bytes, parameter_counts, resolve_assumptions, weight_bytes, weight_bytes_per_device,
)


def usable_vram(gpu, a) -> float:
    """See docs/MATH.md#usable-vram."""
    return gpu["vramBytes"] * (1 - assume(a, "driver_reserved_vram_fraction"))


def _base_allocation(i, all_positions=False):
    """See docs/MATH.md#allocation."""
    w = weight_bytes(i["model"], i["quant"])
    per_device = weight_bytes_per_device(w["totalBytes"], i["tp"], i["pp"])
    kv = kv_cache_bytes(i["model"], i["workload"], i["tp"], i["pp"], i["kvDtype"], i["blockSize"])
    oh = overhead_bytes(i["model"], i["workload"], i["tp"], i["pp"], i["cudaGraphs"],
                        i["assumptions"], per_device, all_positions)
    return w, per_device, kv, oh


def solve_largest_fit(i, available_kv: float) -> Dict[str, int]:
    """See docs/MATH.md#autofix."""
    def cost(tokens):
        return kv_bytes_for_sequence(i["model"], tokens, i["tp"], i["pp"], i["kvDtype"], i["blockSize"])

    lo, hi = 0, i["model"].max_position_embeddings
    if cost(hi) <= available_kv:
        lo = hi
    else:
        while hi - lo > i["blockSize"]:
            mid = math.floor((lo + hi) / 2)
            if cost(mid) <= available_kv:
                lo = mid
            else:
                hi = mid
    per_seq = cost(min(i["workload"]["avgSeqLen"], i["model"].max_position_embeddings))
    return {"maxModelLen": math.floor(lo / i["blockSize"]) * i["blockSize"],
            "maxNumSeqs": math.floor(available_kv / per_seq) if per_seq > 0 else 0}


def feasible(i, available_kv: float, required_kv: float) -> bool:
    """See docs/MATH.md#feasibility."""
    if available_kv <= 0 or required_kv > available_kv:
        return False
    longest = kv_bytes_for_sequence(i["model"], i["workload"]["maxModelLen"], i["tp"], i["pp"],
                                    i["kvDtype"], i["blockSize"])
    return longest <= available_kv


def pool_tokens(available_kv: float, per_token: float, block_size: int):
    """See docs/MATH.md#feasibility."""
    if per_token <= 0 or available_kv <= 0:
        return 0, 0
    blocks = max(0, math.floor(available_kv / (block_size * per_token)))
    return blocks, blocks * block_size


def _plan_common(i, engine, available_kv, num_blocks, max_tokens, required_kv, fits, free, warnings, weights, per_device, kv, oh, usable, budget):
    plan = {
        "engine": engine, "input": i, "weights": weights, "weightBytesPerDevice": per_device,
        "kv": kv, "overhead": oh, "usableVramBytes": usable, "budgetBytes": budget,
        "availableKvBytes": available_kv, "numBlocks": num_blocks, "maxTokens": max_tokens,
        "requiredKvBytes": required_kv, "fits": fits, "freeBytes": free,
        "warnings": warnings, "validated": False, "autofix": None,
    }
    if not fits:
        plan["autofix"] = solve_largest_fit(i, max(0, available_kv))
    return plan


def plan_vllm(i):
    """See docs/MATH.md#vllm."""
    weights, per_device, kv, oh = _base_allocation(i)
    warnings = list(i["model"].warnings) + list(weights["warnings"]) + list(kv["warnings"]) + list(oh["warnings"])
    usable = usable_vram(i["gpu"], i["assumptions"])
    budget = usable * i["memoryUtilization"]
    reserved = per_device + oh["nonTorchBytes"] + oh["activationBytes"] + oh["logitsBytes"] + oh["graphBytes"]
    available_kv = budget - reserved
    if available_kv <= 0:
        warnings.append({"code": "no_kv_headroom",
                         "message": "Weights (%.1f GiB) plus overhead (%.1f GiB) exceed the %.1f GiB budget: vLLM will abort with \"No available memory for the cache blocks\"."
                                    % (per_device / 2 ** 30, (reserved - per_device) / 2 ** 30, budget / 2 ** 30)})
    blocks, max_tokens = pool_tokens(available_kv, kv["perTokenBytesPerDevice"], i["blockSize"])
    required = kv["totalBytesPerDevice"]
    fits = feasible(i, available_kv, required)
    plan = _plan_common(i, "vllm", available_kv, blocks, max_tokens, required, fits,
                        usable - reserved - required, warnings, weights, per_device, kv, oh, usable, budget)
    if not fits:
        plan["autofix"] = _refine_autofix(i, available_kv)
        need = i["workload"]["concurrency"] * i["workload"]["avgSeqLen"]
        if i["workload"]["maxModelLen"] > max_tokens:
            longest = kv_bytes_for_sequence(i["model"], i["workload"]["maxModelLen"], i["tp"], i["pp"], i["kvDtype"], i["blockSize"])
            warnings.append({"code": "infeasible",
                             "message": "--max-model-len %s needs %.1f GiB of KV but only %.1f GiB is free (KV pool holds %s tokens)."
                                        % (i["workload"]["maxModelLen"], longest / 2 ** 30, max(0, available_kv) / 2 ** 30, max_tokens)})
        else:
            warnings.append({"code": "infeasible",
                             "message": "%s sequences x %s tokens = %s tokens exceeds the %s-token KV pool."
                                        % (i["workload"]["concurrency"], i["workload"]["avgSeqLen"], need, max_tokens)})
    return plan


def _refine_autofix(i, available_kv):
    """See docs/MATH.md#autofix."""
    fix = solve_largest_fit(i, max(0, available_kv))
    for _ in range(3):
        seqs = max(1, fix["maxNumSeqs"])
        probe = dict(i, workload=dict(i["workload"], concurrency=seqs, maxModelLen=max(1, fix["maxModelLen"])))
        _, per_device, _, oh = _base_allocation(probe)
        avail = usable_vram(probe["gpu"], probe["assumptions"]) * probe["memoryUtilization"] - (
            per_device + oh["nonTorchBytes"] + oh["activationBytes"] + oh["logitsBytes"] + oh["graphBytes"])
        fix = solve_largest_fit(probe, max(0, avail))
    return fix


def plan_sglang(i):
    """See docs/MATH.md#sglang."""
    i = dict(i, workload=dict(i["workload"]))
    if i["workload"].get("prefixCache") is None:
        i["workload"]["prefixCache"] = {"enabled": True, "hitRate": 0.5, "sharedPrefixTokens": 0}
    weights, per_device, kv, oh = _base_allocation(i)
    warnings = list(i["model"].warnings) + list(weights["warnings"]) + list(kv["warnings"]) + list(oh["warnings"])
    usable = usable_vram(i["gpu"], i["assumptions"])
    budget = usable * i["memoryUtilization"]
    available_kv = budget - per_device
    dynamic_need = oh["activationBytes"] + oh["graphBytes"] + oh["cudaContextBytes"] + oh["commBytes"] + oh["logitsBytes"]
    dynamic_have = usable - budget
    if dynamic_have < dynamic_need:
        warnings.append({"code": "mem_fraction_static_too_high",
                         "message": "--mem-fraction-static %.2f leaves %.1f GiB for activations but %.1f GiB is needed. Lower it to %.2f or below."
                                    % (i["memoryUtilization"], dynamic_have / 2 ** 30, dynamic_need / 2 ** 30, 1 - dynamic_need / usable)})
    blocks, max_tokens = pool_tokens(available_kv, kv["perTokenBytesPerDevice"], i["blockSize"])
    fits = dynamic_have >= dynamic_need and feasible(i, available_kv, kv["totalBytesPerDevice"])
    return _plan_common(i, "sglang", available_kv, blocks, max_tokens, kv["totalBytesPerDevice"], fits,
                        usable - per_device - kv["totalBytesPerDevice"] - oh["totalBytes"],
                        warnings, weights, per_device, kv, oh, usable, budget)


def plan_trtllm(i):
    """See docs/MATH.md#tensorrt-llm."""
    weights, per_device, kv, oh = _base_allocation(i)
    warnings = list(i["model"].warnings) + list(weights["warnings"]) + list(kv["warnings"]) + list(oh["warnings"]) + [{
        "code": "trtllm_build_time_budget",
        "message": "TensorRT-LLM freezes activation workspace at build time; these numbers assume the engine was built with the same max_num_tokens/max_batch_size shown in the emitted flags.",
    }]
    usable = usable_vram(i["gpu"], i["assumptions"])
    build_activation = oh["activationBytes"] + oh["logitsBytes"]
    budget = usable * i["memoryUtilization"]
    available_kv = i["memoryUtilization"] * (usable - per_device - build_activation - oh["nonTorchBytes"])
    blocks, max_tokens = pool_tokens(available_kv, kv["perTokenBytesPerDevice"], i["blockSize"])
    fits = feasible(i, available_kv, kv["totalBytesPerDevice"])
    return _plan_common(i, "trtllm", available_kv, blocks, max_tokens, kv["totalBytesPerDevice"], fits,
                        usable - per_device - kv["totalBytesPerDevice"] - build_activation - oh["nonTorchBytes"],
                        warnings, weights, per_device, kv, oh, usable, budget)


def plan_llamacpp(i):
    """See docs/MATH.md#llamacpp."""
    weights, per_device, kv, oh = _base_allocation(i)
    warnings = list(i["model"].warnings) + list(weights["warnings"]) + list(kv["warnings"]) + list(oh["warnings"])
    if not i["quant"].startswith("gguf:"):
        warnings.append({"code": "llamacpp_needs_gguf",
                         "message": 'llama.cpp serves GGUF; quant "%s" is not a GGUF scheme. Use e.g. gguf:Q4_K_M.' % i["quant"]})
    usable = usable_vram(i["gpu"], i["assumptions"])
    budget = usable * i["memoryUtilization"]
    available_kv = budget - per_device - oh["cudaContextBytes"] - oh["activationBytes"]
    per_tok = kv["perTokenBytesPerDevice"]
    _, max_tokens = pool_tokens(available_kv, per_tok, i["blockSize"])
    need = i["workload"]["concurrency"] * i["workload"]["avgSeqLen"]
    fits = feasible(i, available_kv, need * per_tok)
    return _plan_common(i, "llamacpp", available_kv, 0, max_tokens, need * per_tok, fits,
                        usable - per_device - need * per_tok - oh["cudaContextBytes"] - oh["activationBytes"],
                        warnings, weights, per_device, kv, oh, usable, budget)


ENGINES = {"vllm": plan_vllm, "sglang": plan_sglang, "trtllm": plan_trtllm, "llamacpp": plan_llamacpp}


def emit_flags(plan) -> List[str]:
    """Runnable server flags for this plan. See docs/MATH.md#vllm."""
    i = plan["input"]
    fix = plan["autofix"] or {"maxModelLen": 0, "maxNumSeqs": 0}
    length = i["workload"]["maxModelLen"] if plan["fits"] else fix["maxModelLen"]
    seqs = i["workload"]["concurrency"] if plan["fits"] else fix["maxNumSeqs"]
    e = plan["engine"]

    if e == "vllm":
        flags = ["--tensor-parallel-size %d" % i["tp"]]
        if i["pp"] > 1:
            flags.append("--pipeline-parallel-size %d" % i["pp"])
        flags += ["--max-model-len %d" % length, "--max-num-seqs %d" % seqs,
                  "--gpu-memory-utilization %.2f" % i["memoryUtilization"],
                  "--max-num-batched-tokens %d" % i["workload"]["chunkTokens"]]
        if i["blockSize"] != 16:
            flags.append("--block-size %d" % i["blockSize"])
        if i["kvDtype"].startswith("fp8"):
            flags.append("--kv-cache-dtype %s" % ("fp8" if i["kvDtype"] == "fp8" else i["kvDtype"]))
        if i["quant"].startswith("awq"):
            flags.append("--quantization awq")
        elif i["quant"].startswith("gptq"):
            flags.append("--quantization gptq")
        elif i["quant"] == "fp8":
            flags.append("--quantization fp8")
        if not i["cudaGraphs"]:
            flags.append("--enforce-eager")
        flags.append("--enable-prefix-caching" if (i["workload"].get("prefixCache") or {}).get("enabled")
                     else "--no-enable-prefix-caching")
        return flags

    if e == "sglang":
        static = min(i["memoryUtilization"],
                     (plan["weightBytesPerDevice"] + plan["requiredKvBytes"]) / plan["usableVramBytes"])
        flags = ["--tp %d" % i["tp"]]
        if i["pp"] > 1:
            flags.append("--pp %d" % i["pp"])
        flags += ["--context-length %d" % length, "--max-running-requests %d" % seqs,
                  "--mem-fraction-static %.2f" % static,
                  "--chunked-prefill-size %d" % i["workload"]["chunkTokens"]]
        if i["kvDtype"].startswith("fp8"):
            flags.append("--kv-cache-dtype fp8_e5m2")
        if i["quant"].startswith("awq"):
            flags.append("--quantization awq")
        elif i["quant"].startswith("gptq"):
            flags.append("--quantization gptq")
        elif i["quant"] == "fp8":
            flags.append("--quantization fp8")
        if (i["workload"].get("prefixCache") or {}).get("enabled") is False:
            flags.append("--disable-radix-cache")
        if not i["cudaGraphs"]:
            flags.append("--disable-cuda-graph")
        return flags

    if e == "trtllm":
        flags = ["--max_batch_size %d" % seqs, "--max_seq_len %d" % length,
                 "--max_num_tokens %d" % i["workload"]["chunkTokens"], "--tp_size %d" % i["tp"]]
        if i["pp"] > 1:
            flags.append("--pp_size %d" % i["pp"])
        if i["kvDtype"].startswith("fp8"):
            flags.append("--kv_cache_dtype fp8")
        flags.append("--kv_cache_free_gpu_memory_fraction %.2f" % i["memoryUtilization"])
        return flags

    seqs = seqs if plan["fits"] else max(1, fix["maxNumSeqs"])
    flags = ["-c %d" % (length * max(1, seqs)), "-np %d" % seqs, "-ngl 99",
             "-b %d" % i["workload"]["chunkTokens"]]
    if i["kvDtype"].startswith("fp8") or i["kvDtype"] == "int8":
        flags += ["-ctk q8_0", "-ctv q8_0"]
    if i["tp"] > 1:
        flags.append("-ts %s" % ",".join(["1"] * i["tp"]))
    return flags


SERVE = {
    "vllm": lambda r: "vllm serve %s" % r,
    "sglang": lambda r: "python -m sglang.launch_server --model-path %s" % r,
    "trtllm": lambda r: "trtllm-build --checkpoint_dir %s" % r,
    "llamacpp": lambda r: "llama-server -m %s" % r,
}


def all_reduce_seconds(byte_count: float, n: int, link_bytes_per_sec: float, efficiency: float) -> float:
    """See docs/MATH.md#tp-communication."""
    if n <= 1:
        return 0
    return (2 * (n - 1) / n) * (byte_count / (link_bytes_per_sec * efficiency))


def causal_attention_flops(model: ModelSpec, s: int) -> float:
    """See docs/MATH.md#prefill-flops."""
    width = model.num_attention_heads * model.head_dim
    f = 0.0
    for layer in model.layers:
        if layer.kind in ("mamba", "linear"):
            continue
        f += 2 * s * min(s, layer.window_size if layer.window_size else s) * width
    return f


def estimate_throughput(model, gpu, tp, workload, weight_bytes_per_device_, kv_per_token, compute_dtype, a):
    """See docs/MATH.md#throughput."""
    tp = max(1, tp)
    mbu = assume(a, "mbu_decode")
    mfu = assume(a, "mfu_prefill")
    eta = assume(a, "interconnect_efficiency")
    link = gpu["interconnect"]["bidirectionalBytesPerSec"]

    p = parameter_counts(model)
    embed_fraction = p["embedding"] / p["total"] if p["total"] > 0 else 0
    streamed = weight_bytes_per_device_ * (1 - embed_fraction)

    batch_tokens = workload["concurrency"] * workload["avgSeqLen"]
    kv_read = kv_per_token * batch_tokens
    bytes_per_step = streamed + kv_read
    decode_comm = model.num_layers * all_reduce_seconds(workload["concurrency"] * model.hidden_size * 2, tp, link, eta)
    step = bytes_per_step / (gpu["memBandwidthBytesPerSec"] * mbu) + decode_comm

    tflops = (gpu["tflopsDense"].get(compute_dtype)
              or gpu["tflopsDense"].get("bf16") or gpu["tflopsDense"].get("fp16") or 0) * 1e12
    c = max(1, workload["chunkTokens"])
    flops = 2 * active_parameters(model) * c + causal_attention_flops(model, c)
    prefill_comm = model.num_layers * all_reduce_seconds(c * model.hidden_size * 2, tp, link, eta)
    compute_seconds = flops / (tflops * mfu * tp) if tflops > 0 else float("inf")
    ttft = compute_seconds + prefill_comm

    return {
        "method": "roofline",
        "decode": {"stepSeconds": step, "itlMs": step * 1000,
                   "tokensPerSecond": workload["concurrency"] / step,
                   "commSeconds": decode_comm, "bytesPerStep": bytes_per_step},
        "prefill": {"flops": flops, "ttftSeconds": ttft, "tokensPerSecond": c / ttft, "commSeconds": prefill_comm},
        "bound": "memory" if bytes_per_step / (gpu["memBandwidthBytesPerSec"] * mbu) > compute_seconds else "compute",
    }


def validation_for(model_id, gpu_id, engine):
    """See docs/MATH.md#validation."""
    for v in DATA.get("validation") or []:
        if v["model"] == model_id and v["gpu"] == gpu_id and v["engine"] == engine:
            return v
    return None


def resolve_model(model) -> ModelSpec:
    """Snapshot id or raw config -> ModelSpec. See docs/MATH.md#model-normalization."""
    if not isinstance(model, str):
        return normalize_config(model)
    snap = DATA["models"].get(model)
    if snap is None:
        raise UnknownEntityError("model", model, sorted(DATA["models"]))
    return normalize_config(snap["config"], model, snap["measuredWeightBytes"])


def size(request: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end sizing, mirroring plan.ts `size`. See docs/MATH.md#allocation."""
    engine = request["engine"]
    if engine not in ENGINES:
        raise UnknownEntityError("engine", engine, sorted(ENGINES))
    model = resolve_model(request["model"])
    context = request["context"]
    workload = {
        "concurrency": request["concurrency"],
        "avgSeqLen": request.get("avgSeqLen", context),
        "maxModelLen": context,
        "chunkTokens": request.get("chunkTokens", 8192),
        "prefixCache": request.get("prefixCache"),
    }
    i = {
        "model": model, "gpu": get_gpu(request["gpu"]),
        "tp": request.get("tp", 1), "pp": request.get("pp", 1),
        "workload": workload, "quant": request.get("quant", "bf16"),
        "kvDtype": request.get("kvDtype", "fp16"),
        "memoryUtilization": request.get("memoryUtilization", 0.9),
        "blockSize": request.get("blockSize", 16),
        "cudaGraphs": request.get("cudaGraphs", True),
        "assumptions": resolve_assumptions(request.get("assume")),
        "modelRef": request["model"] if isinstance(request["model"], str) else model.id,
    }
    plan = ENGINES[engine](i)
    throughput = estimate_throughput(model, i["gpu"], i["tp"], i["workload"], plan["weightBytesPerDevice"],
                                     plan["kv"]["perTokenBytesPerDevice"],
                                     "fp8" if i["quant"] == "fp8" else "bf16", i["assumptions"])
    validation = validation_for(model.id, i["gpu"]["id"], engine)
    if validation:
        plan["validated"] = {"engineVersion": validation["engineVersion"], "errors": validation["errors"]}
    flags = emit_flags(plan)
    return {"plan": plan, "throughput": throughput, "flags": flags,
            "command": " \\\n  ".join([SERVE[engine](i["modelRef"])] + flags),
            "validation": validation, "label": "validated" if validation else "predicted"}
