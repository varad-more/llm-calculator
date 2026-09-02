# Contributing

```bash
pnpm install
pnpm test          # TypeScript suite + Python golden parity
pnpm validate      # diff predictions against contributed engine logs
pnpm build         # core -> dist, web -> static export
```

Node ≥ 22.6 (TypeScript runs directly, there is no build step for tests) and Python ≥ 3.9.

## Submit a validation log

The most valuable contribution. Every number this project prints is a prediction until a real
engine log says otherwise.

1. Serve a model however you normally would.
2. Copy the startup output **verbatim** — from model loading through the KV cache line. Do not
   trim or reformat it; the parser reads the engine's own numbers out of it.
3. Open a [validation log issue](.github/ISSUE_TEMPLATE/validation-log.yml), or send a PR adding
   `validation/cases/<model-slug>__<gpu-id>__<engine>.yaml` directly:

```yaml
model: meta-llama/Llama-3.1-8B-Instruct   # must match a data/models/ snapshot id
gpu: h100-sxm-80                          # must match a data/gpus.json id
engine: vllm
engine_version: "0.8.5"
request:                                  # exactly what the server was started with
  tp: 1
  context: 32768
  concurrency: 256
  avgSeqLen: 4096
  kvDtype: fp16
  memoryUtilization: 0.90
  chunkTokens: 8192
command: |
  vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 ...
log: |
  <paste, unedited>
```

`pnpm validate` diffs it and regenerates `docs/VALIDATION.md`. CI fails outside **weights ±1%**,
**KV ±5%**, **overhead ±15%**.

**Nothing synthetic is ever accepted in `validation/cases/`.** If a case fails tolerance, that is
the point — the formula or an assumption is wrong, and the fix goes in the math, not the log.
Parser unit tests use inline fixtures in `validation/parse.test.ts` so a passing parser can never
be mistaken for a validated prediction.

## Add a GPU

1. Append to `data/gpus.json` with a `source_url` pointing at a primary datasheet.
2. **Dense TFLOPS only.** Vendors headline the 2:1 structured-sparsity number (H100 SXM: 1979
   sparse, 989 dense). Add the sparse bf16 and fp8 figures to the tables in
   `packages/core/test/data.test.ts` so the guard test can reject the wrong one.
3. `pnpm gen && pnpm test`.

`vramBytes` is nominal capacity; the driver's own reservation is the
`driver_reserved_vram_fraction` assumption, not a per-GPU fudge.

## Add a model snapshot

Add the repo to `data/model-list.json` (with ungated `mirrors` if it is gated) and run
`pnpm fetch-models && pnpm gen`. The fetcher records the config plus the **summed size of the
shards the safetensors index names** — not `metadata.total_size`, which is wrong for some repos
(DeepSeek-V3 reports 1369 GB for files that sum to 688 GB) and double-counts repos that ship a
second copy of the weights.

## Add an architecture

`packages/core/src/model.ts` builds the per-layer array. New architectures usually need one of:

- a new `AttentionKind` and its per-token KV cost in `kv.ts`
- a branch in `parameterCounts` for a non-standard block
- a `layer_types` / pattern rule in `buildLayers`

Then, in order:

1. A test in `packages/core/test/model.test.ts` asserting the per-layer breakdown.
2. A test asserting the derived weight bytes land within 1% of the real checkpoint.
3. A docstring on any new exported function with its formula in LaTeX and a
   `docs/MATH.md#anchor` link — a test enforces both, and that the anchor exists.
4. A section in `docs/MATH.md` with a citation.

## Add an empirical constant

Never inline a number. Add it to `data/assumptions.json` with `{value, unit, rationale,
source_url, confidence}`, read it through `assume(assumptions, key)`, and let `pnpm gen`
regenerate `docs/ASSUMPTIONS.md`. Be honest in `confidence`: `low` means you could not find a
primary source, and that is fine — it tells users where to aim their scepticism.

## Keep the ports in sync

`packages/core` and `python/llmsize` must agree. After changing math:

```bash
pnpm goldens   # regenerate fixtures/golden from the TS implementation
pnpm test      # the Python suite then re-executes them
```

A `fixtures/golden` diff in a PR is a behaviour change and should be explained in the
description. If the Python suite goes red after regenerating, the ports have drifted — mirror
the change rather than loosening the tolerance (it is 1e-9 on purpose).

## Style

- Bytes everywhere; format to GiB only in `format.ts`.
- `core` stays pure: no `fs`, no `fetch`, no `Date.now()`, no `Math.random()`, no `process.env`.
  A test greps for them.
- Fail loudly. A missing field raises a typed error naming it; it never defaults quietly.
- Comments explain *why*, especially when a formula looks wrong but is right (MLA has no factor
  of 2; MoE memory uses all experts while FLOPs use top-k).
