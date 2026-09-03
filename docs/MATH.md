# The math

Every formula the engine uses, with its derivation and a source. Each exported function in
`packages/core/src` links to the section that defines it, and a test
(`packages/core/test/docstrings.test.ts`) fails the build if a function has no link or a link
points at a section that does not exist.

Notation: \(d\) hidden size, \(n_h\) attention heads, \(n_{kv}\) key/value heads, \(d_h\) head
dim, \(d_{ff}\) MLP intermediate size, \(L\) layers, \(V\) vocab, \(s\) dtype width in bytes,
\(\ell\) sequence length, \(S\) concurrent sequences, \(B\) paged block size, \(C\) prefill chunk.

---

## Units

Everything inside `@llmsize/core` is **bytes**. No function signature mixes units, and no
intermediate value is in GiB. `format.ts` is the only module that converts, at the presentation
edge: `gib()`, `bytes()`, `seconds()`, `count()`.

GiB means \(2^{30}\) bytes. GPU capacities are quoted by vendors in GB but reported by drivers
in units that are effectively GiB, so an "80GB" H100 is \(80 \times 2^{30}\) nominal here — see
[Usable VRAM](#usable-vram) for the part the driver keeps.

---

## Model normalization

Input is a raw HuggingFace `config.json`; output is a `ModelSpec`. Multimodal wrappers nest the
language model under `text_config` (Gemma 3, Llama 4), so that is unwrapped first and the vision
tower is excluded with a warning — it is real memory we do not model.

\[ d_h = \texttt{config.head\_dim} \;?\; \texttt{config.head\_dim} : d / n_h \]

Attention kind follows head counts: \(n_{kv} = n_h\) is MHA, \(n_{kv} = 1\) is MQA, anything
between is GQA, and the presence of `kv_lora_rank` means MLA.

**Missing fields are never silently defaulted.** If a required field is absent we look in
`data/arch-defaults.json`, which transcribes the transformers config-class defaults with a
`source_url`; anything taken from there is recorded in `spec.assumed` and surfaced in every
output. With no sourced default we raise `IncompleteConfigError` naming the field. This matters:
Gemma 3's `text_config` carries no `vocab_size`, and guessing it silently mis-sizes 2.8 GiB of
embeddings.

Source: [transformers configuration classes](https://github.com/huggingface/transformers/tree/main/src/transformers/models).

## Per-layer dispatch

Layers are **not** homogeneous, and this is where most VRAM calculators are wrong. The layer
array is built per index, with this precedence:

1. An explicit `layer_types` array (gpt-oss, Qwen3-Next, hybrid Mamba models).
2. A `sliding_window_pattern` \(P\): layer \(i\) is global iff \((i+1) \bmod P = 0\). Gemma 3
   ships \(P = 6\) (52 of 62 layers windowed at 1024 tokens); Gemma 2 uses \(P = 2\).
3. A global `sliding_window` that is actually shorter than `max_position_embeddings`. Phi-3.5
   declares `sliding_window: 262144` against a 131072 context — that is not a windowed model.
   Qwen2.5 declares a window and then sets `use_sliding_window: false`.
4. Otherwise every layer takes the base kind.

MLP kind is per layer too: DeepSeek's `first_k_dense_replace` makes the first 3 blocks dense,
Qwen3-MoE's `mlp_only_layers` and `decoder_sparse_step` punch holes in the MoE pattern.

Sources: [Gemma 3 report](https://arxiv.org/abs/2503.19786),
[transformers `Gemma3` implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py).

---

## Weight parameters

Per layer, with a gated (SwiGLU) MLP:

\[ P_{attn} = \underbrace{d\,(n_h d_h)}_{Q} + \underbrace{2\,d\,(n_{kv} d_h)}_{K,V} + \underbrace{(n_h d_h)\,d}_{O} \]
\[ P_{mlp} = 3\,d\,d_{ff} \qquad P_{norm} = 2d \]

Embeddings are \(V d\), and the LM head another \(V d\) unless `tie_word_embeddings`.

**MLA** (DeepSeek V2/V3) replaces the attention block, optionally with a query LoRA of rank \(r_q\):

\[ P_{attn}^{MLA} = \underbrace{d r_q + r_q n_h (d_{nope}+d_{rope})}_{Q} + \underbrace{d\,(r_{kv}+d_{rope})}_{KV_a} + \underbrace{r_{kv} n_h (d_{nope}+d_v)}_{KV_b} + \underbrace{n_h d_v d}_{O} \]

**MoE**: every expert is resident, so memory uses \(E\), not top-\(k\):

\[ P_{moe} = (E + S)\,3\,d\,d_{ff}^{exp} + \underbrace{d E}_{router} \]

**MTP**: DeepSeek V3/R1 ship `num_nextn_predict_layers` multi-token-prediction modules that are
in the checkpoint but not in `num_hidden_layers`. Each is a full transformer layer plus its own
copy of the embedding and head and a \(2d \times d\) projection. Omitting them under-counts
DeepSeek-V3 by 13.5B parameters (2%).

Verification: the analytic count reproduces `metadata.total_size` from
`model.safetensors.index.json` to within 0.006% for 19 of the 20 shipped snapshots (the
exception is Gemma 3, where the index includes the vision tower). See
`packages/core/test/weights.test.ts`.

## Weight bytes

Quantization is per tensor, not global. For tensor class \(c\) at \(b_c\) bits, group size \(g\):

\[ B_c = P_c \frac{b_c}{8} + \underbrace{\left\lceil \frac{P_c}{g} \right\rceil \frac{b_{scale}+b_{zero}}{8}}_{\text{grouped schemes only}} \]

AWQ/GPTQ int4 at \(g = 128\) with an fp16 scale and 4-bit zero point costs
\((16+4)/128 = 0.156\) extra bits per weight — 4.16 bpw, not 4. Embeddings, the LM head and
norms stay at fp16 in every scheme we ship (`tensorPolicy` in `data/quant-bpw.json`), which for
a 128k-vocab 8B model is 2 GiB that a flat "params × 0.5 bytes" estimate loses.

A checkpoint may also declare its own exclusions: gpt-oss is MXFP4 with
`modules_to_not_convert` keeping attention, router, embeddings and LM head in bf16. Those are
mapped onto tensor classes and honoured.

**GGUF** uses *measured* bits-per-weight for the whole file, because k-quants mix block formats
across tensors: Q4_K_M is 4.85 bpw, not 4. Table and source in `data/quant-bpw.json`
([llama.cpp k-quants PR](https://github.com/ggml-org/llama.cpp/pull/1684)).

**Preferred path:** when the requested scheme matches the checkpoint's own, we use
`metadata.total_size` directly and skip parameter counting entirely. The result is labelled
`method: 'measured'`.

## Weight sharding

\[ B_{dev} = B_{total} / (TP \cdot PP) \]

Linear and (vocab-parallel) embedding tensors shard across TP ranks; layers split across PP
stages. Norms are replicated but are ~0.01% of the total.

## Active parameters

Only top-\(k\) experts run per token, so FLOPs use:

\[ P_{active} = P_{total} - \sum_{\ell \in MoE} (E - k)\,3\,d\,d_{ff}^{exp} \]

Mixtral-8x7B: 46.7B resident, 12.9B active. Memory must use the first number and compute the
second — mixing them up is the single most common MoE sizing error.

---

## KV dtype

Element width in bytes, from `data/quant-bpw.json`. fp8 (E4M3 or E5M2) is 1 byte per element
plus a per-tensor scale that is negligible at cache scale.

## KV per token

Per layer, per token:

\[ b_{tok}^{GQA} = 2\,n_{kv}\,d_h\,s \qquad b_{tok}^{MLA} = (r_{kv} + d_{rope})\,s \qquad b_{tok}^{SSM} = 0 \]

The MLA form has **no factor of 2 and no \(n_{kv}\)**: the cache holds one compressed latent of
rank \(r_{kv}\) plus the decoupled RoPE key. For DeepSeek-V3 that is 1152 B/layer/token against
the 65536 B/layer/token a naive \(2 n_{kv} d_h s\) with \(n_{kv} = 128\) would predict — a
factor of 57.

A sliding-window layer has the same per-token cost; it just stores fewer tokens.

Sources: [GQA](https://arxiv.org/abs/2305.13245), [DeepSeek-V2](https://arxiv.org/abs/2405.04434).

## SSM state

Mamba and linear-attention layers hold a constant state per **sequence**, not per token:

\[ b_{seq} = (d_{inner} d_{state} + d_{conv} d_{inner})\,s \]

This is why a hybrid model's KV curve flattens with context. Source:
[Mamba](https://arxiv.org/abs/2312.00752).

## KV token count

Paged attention charges whole blocks. With a prefix cache of hit rate \(h\) over \(p\) shared
tokens, the prefix is stored once:

\[ T = \left\lceil \frac{p}{B} \right\rceil B + \sum_{i=1}^{S} \left\lceil \frac{\ell_i - h p}{B} \right\rceil B \]

Without a prefix cache this collapses to \(S \lceil \ell / B \rceil B\). A windowed layer caps
\(\ell\) and \(p\) at \(W\). No sequence exceeds `max_model_len` — a server rejects longer
requests rather than caching them.

Source: [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180),
[SGLang RadixAttention](https://arxiv.org/abs/2312.07104).

## KV cache

\[ B_{KV}^{dev} = \frac{1}{\min(TP, n_{kv})} \sum_{\ell \in \text{stage}} b_{tok}(\ell)\, T(\ell) \]

Two footguns are surfaced as warnings rather than absorbed:

- **\(TP > n_{kv}\).** KV heads replicate; per-GPU KV stops shrinking. Going TP=8 → TP=16 on an
  8-KV-head model buys zero KV headroom and costs an all-reduce.
- **MLA under TP.** The compressed latent is not head-sharded, so *every* rank holds the full KV
  cache. TP buys weight capacity, not KV capacity, on DeepSeek models.

## Pipeline parallel

\[ L_{stage} = \lceil L / PP \rceil \]

The widest stage sets the memory budget; with \(L = 61\) and \(PP = 8\), that is 8 layers, not 7.625.

---

## Usable VRAM

\[ V_{usable} = V_{nominal}\,(1 - r_{driver}) \]

The driver reserves a slice for ECC, page tables and its own context, which is why an "80GB"
H100 reports ~79.6 GiB. \(r_{driver}\) is an assumption (`driver_reserved_vram_fraction`).

## Overhead

\[ B_{overhead} = B_{ctx} + [TP>1]\,B_{nccl} + B_{graph} + B_{act} + B_{logits} + f\,(B_{w}+B_{act}) \]

Every constant is an entry in `data/assumptions.json` with a rationale, a source and an honest
confidence label, and every one is user-overridable (`--assume key=value`, or the UI panel).
There are no magic numbers inline. See [ASSUMPTIONS.md](./ASSUMPTIONS.md).

## Activation peak

For one prefill chunk of \(C\) tokens:

\[ B_{act} = \frac{C\,d\,s\,k + 2\,C\,d_{ff}\,s}{TP} \]

\(k\) live hidden-sized buffers (residual stream, QKV output, attention output, one transient)
plus the two gated-MLP intermediates. There is **no** \(S^2\) term: FlashAttention keeps the
score matrix in SRAM. An MoE layer materialises top-\(k\) expert intermediates per token, so
\(d_{ff} = k\,d_{ff}^{exp}\).

Source: [FlashAttention](https://arxiv.org/abs/2205.14135).

## Logits

\[ B_{logits} = n\,V\,4 \]

fp32 upcast for sampling. Normally \(n = S\), but an engine that keeps logits for every prefill
position pays \(n = C\): at \(C = 8192\) and \(V = 128256\) that is 3.9 GiB on its own, and it is
flagged.

## CUDA graphs

\[ B_{graph} = B_{base} + N_{sizes}\,B_{per} , \qquad N_{sizes} = 3 + \left\lfloor \frac{\min(S, 512)}{8} \right\rfloor \]

vLLM captures graphs for \(\{1,2,4\} \cup \{8,16,\dots\}\) up to `max_num_seqs`. `--enforce-eager`
sets this to zero and is the first thing to try when startup OOMs.

---

## Allocation

Shared by every adapter: weights → per-device weights → KV → overhead. The engines differ in
what they subtract from what, and in what is fixed versus elastic.

## Feasibility

Decided in **bytes**, not tokens:

\[ \text{fits} \iff B_{KV}^{avail} > 0 \;\wedge\; B_{KV}^{req} \le B_{KV}^{avail} \;\wedge\; b_{seq}(L) \le B_{KV}^{avail} \]

The token form \(L \le T_{max} \wedge S\bar{\ell} \le T_{max}\) is only valid when every layer
caches every token. A sliding-window model caches \(\min(\ell, W)\) in most layers, so its pool
holds far more sequences than \(T_{max}/\bar{\ell}\) suggests. Byte comparison is exactly
equivalent for homogeneous models and correct for hybrid ones. \(T_{max}\) is still reported,
because that is the number engines print.

## Autofix

When a request does not fit we solve for the largest one that does:

\[ L^{*} = \max\{L : b_{seq}(L) \le B_{KV}^{avail}\}, \qquad S^{*} = \left\lfloor \frac{B_{KV}^{avail}}{b_{seq}(\bar{\ell})} \right\rfloor \]

\(L^{*}\) is found by bisection rather than division, because \(b_{seq}\) is sublinear in \(L\)
for windowed models. Shrinking \(S\) also frees CUDA-graph and logits memory, which becomes more
KV, so the solve is iterated three times to a fixed point.

---

## vLLM

vLLM profiles a dummy forward pass at startup, subtracts everything it observed from its
utilisation budget, and freezes the remainder into blocks:

\[ B_{KV} = V_{usable}\,u - (B_{w} + B_{nontorch} + B_{act} + B_{graph}) \]
\[ N_{blocks} = \left\lfloor \frac{B_{KV}}{B\, b_{tok}} \right\rfloor, \qquad T_{max} = N_{blocks}\,B \]

Emitted flags: `--tensor-parallel-size`, `--pipeline-parallel-size`, `--max-model-len`,
`--max-num-seqs`, `--gpu-memory-utilization`, `--max-num-batched-tokens`, `--block-size`,
`--kv-cache-dtype`, `--quantization`, `--enforce-eager`, `--enable-prefix-caching`.

Source: [vLLM memory profiling](https://docs.vllm.ai/en/latest/configuration/conserving_memory.html).

## SGLang

SGLang reserves a **static** pool up front. `--mem-fraction-static` covers weights *and* the KV
pool; whatever is left must absorb activations, graphs and the CUDA context:

\[ B_{KV} = V_{usable}\,f_{static} - B_{w}, \qquad f_{static}^{*} = \frac{B_{w} + B_{KV}^{req}}{V_{usable}} \]
\[ \text{feasible} \iff (1-f_{static})\,V_{usable} \ge B_{act} + B_{graph} + B_{ctx} \]

RadixAttention makes prefix reuse the default, so the prefix-cache term is on unless disabled.

Source: [SGLang server arguments](https://docs.sglang.ai/backend/server_arguments.html).

## TensorRT-LLM

Structurally different: the activation budget is **frozen at engine build time**.
`trtllm-build --max_batch_size/--max_num_tokens` bakes fixed workspaces into the plan file, and
only the KV pool is elastic at serve time:

\[ B_{act} = \text{const}(\text{build max\_num\_tokens}), \qquad B_{KV} = f_{free}\,(V_{usable} - B_{w} - B_{act}^{build}) \]

Running a smaller runtime batch does **not** give the KV pool more room. That is the usual
surprise when moving a vLLM config across.

Source: [TensorRT-LLM memory docs](https://nvidia.github.io/TensorRT-LLM/reference/memory.html).

## llamacpp

llama.cpp allocates one contiguous, unpaged KV buffer for `n_ctx` tokens shared by `n_parallel`
slots:

\[ B_{KV} = n_{ctx}\,b_{tok}, \qquad \ell_{slot} = n_{ctx} / n_{parallel} \]

A slot holds its full share whether or not it is used, so there is no paging win. Weights come
from the measured GGUF bits-per-weight table.

---

## Throughput

Explicitly a **roofline**: an upper bound scaled by MBU/MFU. No scheduler, no queueing, no
batching jitter. Labelled `method: 'roofline'` in every result.

Decode is memory-bound — each step streams the weights plus the whole KV cache of the batch:

\[ t_{step} = \frac{B_{w}^{dev,stream} + b_{tok} \sum_i \ell_i}{BW \cdot \text{MBU}} + t_{comm} , \qquad \text{tok/s} = \frac{S}{t_{step}} , \qquad \text{ITL} = t_{step} \]

The embedding table is excluded from \(B_w^{stream}\): it is gathered per token, not streamed.
For a 262k-vocab model that is 2.8 GiB of false traffic.

MoE decode streams **all** expert bytes (they are resident and gathered) while computing only
top-\(k\) — the two use different parameter counts.

Prefill is compute-bound:

\[ \text{TTFT} = \frac{2 P_{active} C + F_{attn}(C)}{\text{TFLOPS}_{dense} \cdot \text{MFU} \cdot TP} + t_{comm} \]

TFLOPS are always **dense**. Vendor datasheets headline the 2:1 structured-sparsity figure
(H100 SXM: 1979 sparse, 989 dense) and a test asserts no entry in `data/gpus.json` exceeds its
dense ceiling.

Sources: [MBU / inference performance](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices),
[MFU](https://arxiv.org/abs/2204.02311).

## Prefill FLOPs

\[ F = 2 P_{active} C + F_{attn}(C), \qquad F_{attn}(s) = \sum_{\ell} 2\,s\,\min(s, W_\ell)\,n_h d_h \]

The \(2\) in \(2 P C\) is multiply-accumulate. The attention term is \(QK^\top\) and \(PV\) at
\(2 s^2 n_h d_h\) each, halved by causality. A windowed layer replaces \(s^2\) with \(s W\),
which is why Gemma 3's prefill cost grows near-linearly where Llama's grows quadratically.

## TP communication

Two all-reduces per layer (after attention output, after the MLP), ring algorithm:

\[ t = \frac{2(n-1)}{n} \cdot \frac{\text{bytes}}{BW_{link}\,\eta} , \qquad \text{bytes} = \text{tokens} \cdot d \cdot s \]

\(\eta\) is `interconnect_efficiency`. This is why TP scaling is sublinear, and why TP across
PCIe behaves nothing like TP across NVLink.

---

## Validation

Predictions are diffed against real engine startup logs. `validation/cases/*.yaml` holds the
engine version, the flags used and the raw log; a parser extracts the engine's own reported
breakdown and `pnpm validate` reports per-component error against these formulas.

CI tolerances: **weights ±1%**, **total KV bytes ±5%**, **overhead ±15%**. Exceeding one fails
the build.

Any (model, GPU, engine) triple with no case is labelled `predicted`, and says so in every
output. A triple with a case is labelled `validated` and carries its measured error. See
[VALIDATION.md](./VALIDATION.md).

---

# Phase 4: serving features

## Multi-LoRA

A rank-\(r\) adapter on a projection \(W \in \mathbb{R}^{m \times n}\) stores \(A \in
\mathbb{R}^{m \times r}\) and \(B \in \mathbb{R}^{r \times n}\) — \(r(m+n)\) parameters, not
\(r m n\). Per layer over vLLM's default target set:

\[ P_{lora} = r\big[\underbrace{(d + n_h d_h) + 2(d + n_{kv} d_h)}_{qkv} + \underbrace{(n_h d_h + d)}_{o} + \underbrace{2(d + d_{ff})}_{gate,up} + \underbrace{(d_{ff} + d)}_{down}\big] \]

Every slot is pre-allocated at `--max-lora-rank` whether or not the loaded adapter uses it, so
resident cost is \(S \cdot P_{lora} \cdot s\) for \(S\) = `--max-loras`. That memory comes out of
the KV pool, which is why raising `--max-loras` shortens the maximum context.

Source: [LoRA](https://arxiv.org/abs/2106.09685),
[vLLM multi-LoRA](https://docs.vllm.ai/en/latest/features/lora.html).

## Speculative decoding

With \(k\) proposed tokens and per-token acceptance \(\alpha\), the draft chain is accepted
until the first rejection and the target adds one bonus token:

\[ E[\text{tokens}] = \sum_{j=0}^{k} \alpha^{j} = \frac{1 - \alpha^{k+1}}{1 - \alpha} \]

\[ t_{cycle} = k\, t_{draft} + t_{target} , \qquad \text{ITL} = \frac{t_{cycle}}{E[\text{tokens}]} \]

The target verifies all \(k+1\) positions in one pass; decode is memory-bound, so that pass
costs about what a one-token step costs. Draft weights and draft KV are resident on the same
device and come out of the same budget.

Speculation is a **latency** optimisation. At high batch the target step is already
compute-saturated and the draft work is pure overhead, so the computed speedup can come out
below 1 — and it is reported that way rather than hidden.

Source: [speculative decoding](https://arxiv.org/abs/2211.17192),
[Medusa](https://arxiv.org/abs/2401.10774).

## Disaggregation

Prefill is compute-bound and decode is bandwidth-bound, so splitting them lets each pool be
sized and parallelised independently. The cost is that every request's KV must cross the wire:

\[ B_{xfer} = b_{seq}(\ell_{prompt}) \cdot TP_{prefill} , \qquad t_{xfer} = \frac{B_{xfer}}{BW_{link}} \]

Pool balance follows from each side's own rate:

\[ R_{prefill} = \frac{\text{tok/s}_{prefill}}{\ell_{prompt}} , \qquad R_{decode} = \frac{\text{tok/s}_{decode}}{\ell_{out}} , \qquad \frac{N_{decode}}{N_{prefill}} = \frac{R_{prefill}}{R_{decode}} \]

The bottleneck is whichever of prefill, decode or transfer sustains the fewest requests per
second. On commodity fabric it is usually transfer: a 32k-token Llama-70B prompt is ~2.5 GiB of
KV per request, which is 200 ms on 100G Ethernet before any compute happens. fp8 KV halves it;
MLA models cut it by ~50x.

Source: [DistServe](https://arxiv.org/abs/2401.09670),
[Splitwise](https://arxiv.org/abs/2311.18677).

## Reverse lookup

Given hardware, enumerate what runs on it: a cartesian sweep of (model, quant, KV dtype,
context, TP) through the same allocator, ranked by fit and then by \(\text{context} \times
\text{tok/s}\). No heuristics and no scoring model — every row is a real plan with a real
command attached.

# Phase 5: what to rent

## Cost per token

A rented machine bills by the hour; an inference workload is priced per token. The bridge is
the machine's own throughput:

\[ \$_{1M} = \frac{P_{hour}}{T \cdot 3600} \times 10^{6} \]

\(T\) is tokens/second for the side being priced — decode throughput gives the output-token
price, prefill throughput the input-token price. The two differ by one to two orders of
magnitude, which is exactly why every commercial API quotes them separately.

The machine is billed whether or not it is busy, so this is the price at **full saturation**.
It is a floor, not a forecast. A deployment running at 40% utilization pays 2.5x this, and the
arithmetic is identical to raising \(P_{hour}\) by 2.5x — so a duty cycle belongs on the rate,
not in this formula.

Prices are a dated snapshot in [`data/instances.json`](../data/instances.json), on-demand and
region-specific. They move. Override the rate with whatever you actually pay.

## Machine search

An \(N\)-GPU machine can be run as one \(TP=N\) server, or as

\[ R = \frac{N}{TP} \]

independent replicas behind a load balancer, each serving \(\lceil C / R \rceil\) of the
\(C\) concurrent sequences, for a whole-machine throughput of \(T_{machine} = R \cdot
T_{replica}\).

Which layout wins is not a matter of taste; it falls out of the decode roofline. A step streams
the weights resident on the device **once** and serves the entire batch from that one read.
Tensor parallel divides that read by \(TP\). Replicas do not — each replica re-reads the whole
model for its own slice of the batch. So at fixed total concurrency TP wins on throughput and
on latency together, until the all-reduce term costs more than the read it saves:

| 8x H100, Llama-3.1-8B fp8 | TP=1, R=8 | TP=2, R=4 | TP=4, R=2 | TP=8, R=1 |
| --- | --- | --- | --- | --- |
| C = 64 | 14,585 tok/s | 26,153 | 43,051 | **62,358** |
| C = 1024 | 86,925 | 103,026 | 111,566 | **112,310** |
| C = 4096 | does not fit | 120,777 | **121,212** | 116,997 |

The crossover is real but it is a long way out — past a thousand concurrent sequences on this
model — and it moves with model size, quantization and link speed. So the search tries every
\(TP\) that divides \(N\) and reads the answer off the allocator, rather than encoding a rule
of thumb that is right in one regime and wrong in the next.

What this comparison does **not** price is why people still run replicas at low concurrency:
blast radius, rolling restarts, and heterogeneous models sharing a box. Those are real and they
are not throughput.

TTFT and ITL are one replica's, since a request is served by exactly one replica. Tokens per
second, and therefore price per token, are the whole machine's.

The layout kept per machine is the cheapest per output token among those that fit and meet the
SLO. When nothing meets it, the cheapest fitting layout is returned with the failing clauses
attached, because "closest, and here is what it misses" beats an empty table.

SLO clauses are latency only — TTFT and ITL. Throughput is not an SLO here; it is the thing
being bought, and it shows up as the price.
