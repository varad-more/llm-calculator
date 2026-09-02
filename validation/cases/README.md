# Validation cases

One YAML file per real, observed engine startup. Filename convention:
`<model-slug>__<gpu-id>__<engine>.yaml`.

```yaml
model: meta-llama/Llama-3.1-8B-Instruct   # must match a data/models/ snapshot id
gpu: h100-sxm-80                          # must match a data/gpus.json id
engine: vllm
engine_version: "0.8.5"
request:                                  # exactly the flags the server was started with
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
  <paste the startup output verbatim, from "Starting to load model" through
   "GPU KV cache size", with nothing edited or elided>
```

**Nothing in this directory is synthetic.** Every file must be a copy-paste from a machine
that actually ran the command. Parser unit tests use inline fixtures in
`validation/parse.test.ts` instead, so a passing parser can never be mistaken for a
validated prediction.

`pnpm validate` diffs every case and regenerates `docs/VALIDATION.md`; `pnpm validate --check`
is the CI gate.
