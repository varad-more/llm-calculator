# llmsize — implementation plan

Execution order from the spec. Each step: implement, run the full suite, report.

- [x] 1. Scaffold monorepo + `data/gpus.json` (12 sourced entries) + schema tests
- [x] 2. `ModelSpec` normalization + weight memory + tests
- [x] 3. KV cache with per-layer dispatch (GQA, MLA, SWA, Mamba) + tests
- [x] 4. Overhead + vLLM adapter + flag emission + tests (also SGLang, TRT-LLM, llama.cpp)
- [x] 5. Validation harness + CI gate — harness and parser done; **0 real cases** (no hardware)
- [x] 6. CLI
- [x] 7. Throughput roofline + tests
- [x] 8. Python port + parity tests (16 golden fixtures, 1e-9)
- [x] 9. Web UI (Next.js static export, per-model routes)
- [x] 10. Phase 4: disaggregation, speculative decoding, multi-LoRA, reverse lookup

## Review

**What shipped.** 82 TypeScript tests and a 16-fixture Python parity suite, all green. Core is
zero-dependency and pure. Four engine adapters, each modelling its own allocation shape rather
than sharing one. 20 real HF model snapshots, 12 sourced GPUs, every empirical constant in
`data/assumptions.json` with a source and a confidence label.

**Bugs the tests caught, not review.**

- Property tests found `maxTokens`/`numBlocks` going negative in three adapters (only vLLM
  clamped), and found that the spec's token-based feasibility check is wrong for
  sliding-window models — a hybrid KV pool holds far more sequences than `T_max/avg_len`
  suggests. Feasibility is now decided in bytes, which is exactly equivalent for homogeneous
  models and correct for hybrid ones.
- Cross-checking derived weights against real checkpoints found DeepSeek-V3's MTP module
  (13.5B params outside `num_hidden_layers`) and gpt-oss's partial MXFP4 quantization.
- `metadata.total_size` in the safetensors index turned out to be unreliable — DeepSeek-V3
  reports 1369 GB for files that sum to 688 GB. Fixed at the source (sum the shards the index
  names) and defended in the math (measured is rejected when it disagrees with the count by
  >5%, with a warning).
- The disaggregation transfer size multiplied MLA KV by TP, an 8× overcount, because the MLA
  latent is replicated rather than sharded. Fixed by extracting `kvShards()`.

**Known gaps, stated rather than hidden.**

- **Zero validation cases.** No GPU access. The harness, parser, tolerances and CI gate are
  built and tested against inline fixtures; `docs/VALIDATION.md` says "no cases yet" and every
  output is labelled `predicted`. This is the project's headline claim and it is currently
  unbacked.
- Overhead constants (CUDA context, NCCL buffers, CUDA-graph memory) are `confidence: low`.
  They are the components a validation log would correct first.
- SSM/Mamba: KV state and per-layer dispatch are modelled, mixer *weights* are not (no
  snapshot uses them yet); `weightBytes` warns that the total is a lower bound.
- The web UI was verified by build and typecheck only — the browser extension was not
  connected, so there was no visual pass.
- Throughput is a roofline. It is labelled `roofline` and `predicted` everywhere it appears.

---

## Machine selection on AWS (#4 compare, #2 SLO -> machine, #1 cost)

One sweep serves all three: #2 is the sweep filtered by an SLO, #1 is a column of it.

- [ ] `data/instances.json` — AWS GPU instance catalogue. Price from the AWS pricing feed
      (`b0.p.awsstatic.com/pricing/2.0/...`), GPU count / host RAM / NVMe / network from the
      AWS accelerated-computing docs table. Every row carries `source_url` and `priceRetrieved`.
- [ ] Fix first: GDDR6 datacenter cards (A10G, L4, L40S) ship with in-band ECC on, which eats
      1/16 of the framebuffer. AWS documents 22 GiB of 24 and 44 GiB of 48. Without this the
      tool says "fits" on a g6e.xlarge when it does not. Regenerate goldens.
- [ ] `packages/core/src/cost.ts` — `costPerMillionTokens`, `compareMachines`. Replicas, not
      just TP: if the model fits on one GPU you run 8 replicas on a p5.48xlarge, you do not run
      TP=8. Aggregate throughput scales with replicas, latency does not.
- [ ] `data.ts` — `listInstances` / `getInstance`.
- [ ] docs/MATH.md `## Cost per token`, `## Machine search`; `cost.ts` into MATH_MODULES.
- [ ] `data.test.ts` — every instance resolves its GPU, has a price, a source and a date.
- [ ] Web: a "Machines" tab in the sizer. Columns: fits, layout, tok/s, TTFT, $/1M in, $/1M out.
      SLO inputs (max TTFT, min tok/s per user) filter it. Hourly rate editable per row.
- [ ] `gen-docs.mjs` -> docs/INSTANCES.md.
