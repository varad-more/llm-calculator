# llmsize

Inference sizing and serving-config generation for LLMs. Give it a model, a GPU and a workload;
it tells you what the serving engine will allocate, whether it fits, roughly how fast it goes —
and prints the exact flags to run it.

```bash
llmsize plan --model meta-llama/Llama-3.1-70B-Instruct --gpu h100-sxm-80 --tp 4 \
             --engine vllm --context 32768 --concurrency 64 --kv-dtype fp8
```

```
meta-llama/Llama-3.1-70B-Instruct on 4x NVIDIA H100 SXM5 80GB with vllm
70.55B params (measured), 80 layers, gqa, quant bf16, kv fp8

PER-GPU MEMORY of 79.68 GiB usable
  weights       32.85 GiB  ████████████················ 41.2%
  kv cache      10.00 GiB  ████························ 12.6%
  activations    0.37 GiB  ···························· 0.5%
  overhead       2.00 GiB  █··························· 2.5%
  free          34.45 GiB  ████████████················ 43.2%

FIT
  ✓ fits — KV pool holds 956.4k tokens (59774 blocks), 36.48 GiB reserved

RUN IT
  vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching
```

## Predicted vs. measured

**Every number this tool currently prints is predicted, not measured.**

There are no contributed engine startup logs yet, so `docs/VALIDATION.md` is empty and every
result is labelled `predicted`. The harness that would validate them is built and tested — it
parses vLLM and SGLang startup output, diffs each component against our formulas, and fails CI
outside ±1% (weights) / ±5% (KV) / ±15% (overhead). It just has nothing to chew on.

If you have a GPU, [pasting one startup log](.github/ISSUE_TEMPLATE/validation-log.yml) is worth
more than any argument about the math.

What *is* measured today is the weights. The analytic parameter count is checked against each
checkpoint's own file sizes on Hugging Face:

| Model | Derived | Checkpoint | Error |
| --- | --- | --- | --- |
| meta-llama/Llama-3.1-8B-Instruct | 14.96 GiB | 14.96 GiB | 0.000% |
| meta-llama/Llama-3.1-405B-Instruct | 755.96 GiB | 755.96 GiB | 0.000% |
| Qwen/Qwen3-235B-A22B | 437.90 GiB | 437.90 GiB | 0.000% |
| deepseek-ai/DeepSeek-V3 (fp8) | 640.93 GiB | 641.30 GiB | −0.06% |
| openai/gpt-oss-120b (mxfp4) | 60.69 GiB | 60.77 GiB | −0.12% |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 86.99 GiB | 86.99 GiB | 0.000% |

All 20 shipped snapshots are within 0.12%. (Gemma 3 is the deliberate exception: its file listing
includes a vision tower we exclude from text-only serving, so the derived count is used and the
discrepancy is reported rather than absorbed.)

## Why another one of these

There are plenty of VRAM calculators. Four things here are different.

**1. It models what the engine allocates, not an abstract sum.** vLLM profiles a forward pass,
subtracts weights + non-torch memory + activation peak + CUDA graphs from its utilisation budget,
and freezes the rest into fixed-size blocks. SGLang reserves weights and KV together up front via
`--mem-fraction-static`. TensorRT-LLM bakes its activation workspace in at *build* time, so a
smaller runtime batch buys the KV pool nothing. Those are different equations, and each adapter
implements its own.

**2. Layers are not assumed homogeneous.** This is where most tools are wrong:

- **Gemma 3 27B** is 52 sliding-window layers (1024 tokens) to 10 global. At 32k context that is
  2.9 GiB of KV, not the 15.5 GiB a uniform model predicts — a 5.3× error.
- **DeepSeek-V3** uses MLA: one compressed latent per token, `(512 + 64) × 2` = 1152 B/layer,
  no factor of 2 and no KV heads. A naive `2 · n_kv · d_h · s` with its 128 heads is 57× too big.
- **gpt-oss** ships an explicit `layer_types` array alternating 128-token windows with full
  attention, and MXFP4 experts alongside bf16 attention.
- **DeepSeek-V3** also ships an MTP module that is in the checkpoint but not in
  `num_hidden_layers` — 13.5B parameters most tools drop.

**3. The output runs.** Not a number to eyeball, a command to paste. When a config does not fit,
it solves for the largest context and concurrency that do and emits *those*.

**4. It models serving features nobody else does.** Prefix-cache hit rate, disaggregated
prefill/decode pools (including the KV bytes that have to cross the wire, which is usually the
real bottleneck), speculative decoding with acceptance-rate-driven yield, multi-LoRA slot memory,
and reverse lookup from a GPU budget to the set of configurations that fit.

## Install

```bash
pnpm install && pnpm build
node packages/cli/bin/llmsize.mjs plan --model Qwen/Qwen3-30B-A3B --gpu l40s-48 --context 8192 --concurrency 16 --quant fp8
```

Python, importable next to vLLM:

```python
from llmsize import size
r = size({"model": "meta-llama/Llama-3.1-70B-Instruct", "gpu": "h100-sxm-80",
          "engine": "vllm", "tp": 4, "context": 32768, "concurrency": 64})
print(r["plan"]["fits"], r["command"])
```

TypeScript:

```ts
import { size, gib } from '@llmsize/core'
const r = size({ model: 'deepseek-ai/DeepSeek-V3', gpu: 'h200-sxm-141', engine: 'sglang',
                 tp: 8, context: 32768, concurrency: 16, quant: 'fp8' })
console.log(r.plan.fits, gib(r.plan.availableKvBytes), r.command)
```

## Web UI

```bash
pnpm install
pnpm dev       # http://localhost:3000
pnpm preview   # static export, served on :8080 — no server, no backend
```

Two pages, both driven by the same pure functions as the CLI:

- **Size a config** — pick a model, GPU, engine and workload and watch the per-GPU allocation
  redraw. Tabs on the result surface the four serving features: the runnable command, the memory
  breakdown line by line, **speculative decoding** (draft model, acceptance rate, and an honest
  sub-1.0 speedup when the draft is too slow), **multi-LoRA** slot memory, and **disaggregated**
  prefill/decode with the KV bytes that have to cross the wire. Every assumption is editable in
  place. The config lives in the query string, so a sizing is a link you can paste.
- **What fits my GPU** — the question backwards: given the hardware, sweep every model,
  quantization, KV dtype, context and TP degree through the same allocator and rank what survives.

- **How it works** (`/explained/`) — the method, written out: what quantization actually stores and
  why a 4-bit checkpoint is never 4 bits per weight, how the KV cache is sized under GQA/MLA/sliding
  window, the residual each engine is really solving for, and the constants it all rests on. Every
  figure on the page is computed at build time by the same functions the sizer calls, so the
  documentation cannot drift from the math.

Local hardware is in the same catalogue as the datacentre parts: Apple M2/M3/M4/M5 Ultra, Max and
Pro, and GeForce RTX 3090/4090/5090, alongside H100s and MI300X. Unified memory is modelled the way Metal actually hands it out —
a process gets `recommendedMaxWorkingSetSize`, about 75% of RAM, which is why a 128GB M4 Max shows
96 GiB usable and a 512GB M5 Ultra cannot quite hold DeepSeek-V3 at Q4_K_M without raising
`iogpu.wired_limit_mb`.

Plus one pre-rendered static page per model (`/llama-3-1-70b-instruct-vram/`) with real numbers in
the HTML before any JS runs.

## How it is kept honest

- **No invented hardware specs.** Every entry in `data/gpus.json` carries a `source_url`, and a
  test asserts no entry exceeds its known *dense* ceiling — vendor datasheets headline the 2:1
  sparsity figure, which is twice the real number.
- **No invented constants.** MBU, MFU, CUDA-context size, GGUF bits-per-weight: each is an entry
  in `data/assumptions.json` with a rationale, a source and an honest confidence label, and each
  is overridable (`--assume mbu_decode=0.72`). Nothing is hardcoded in the math.
- **No silent defaults.** A config missing `num_key_value_heads` raises `IncompleteConfigError`
  naming the field. Fields taken from transformers' own class defaults (Gemma 3 omits
  `vocab_size`) are sourced *and* surfaced in every output.
- **Bytes everywhere.** GiB exists only at the presentation edge.
- **Every formula is documented.** [docs/MATH.md](docs/MATH.md) has each one in LaTeX with a
  derivation and a citation; a test fails the build if an exported math function lacks a
  docstring linking to its section, or links to a section that does not exist.
- **The two language ports cannot drift.** 16 golden fixtures in `fixtures/golden/` are executed
  by both the TypeScript and Python suites and compared to 1e-9.
- **Core is pure.** Zero runtime dependencies, no I/O, no clock, no randomness — a test greps for
  them.

## Layout

```
packages/core/     TypeScript. Zero runtime deps. All the math.
packages/cli/      llmsize CLI.
python/llmsize/    Python port. Same data, same fixtures.
apps/web/          Next.js static export.
data/              gpus.json, assumptions.json, quant-bpw.json, models/, arch-defaults.json
fixtures/golden/   Language-agnostic test vectors.
validation/        Real engine logs, their parser, and the CI gate.
docs/MATH.md       Every formula, derived and cited.
```

## Scope

Inference only. Training and optimizer memory are out of scope. So are GPU marketplaces, model
recommendations, auth and any backend service — this is a pure function with a few front ends.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers submitting a validation log, adding a GPU, and adding an
architecture adapter. Validation logs are the most useful thing you can send.

## Licence

Apache-2.0
