import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseVllmLog, parseSglangLog, parseLog } from './parse.ts'

const GiB = 2 ** 30

// Synthetic logs in the shape the engines print. These exercise the PARSER only; they are
// not validation cases and never appear in docs/VALIDATION.md. Real logs live in
// validation/cases/ and come from people with the hardware.
const VLLM_V1 = `
INFO 03-14 09:12:41 [gpu_model_runner.py:1273] Starting to load model meta-llama/Llama-3.1-8B-Instruct...
INFO 03-14 09:12:55 [gpu_model_runner.py:1289] Model loading took 14.9888 GiB and 13.204526 seconds
INFO 03-14 09:13:07 [gpu_worker.py:227] Memory profiling takes 11.85 seconds
the current vLLM instance can use total_gpu_memory (79.11GiB) x gpu_memory_utilization (0.90) = 71.20GiB
model weights take 14.99GiB; non_torch_memory takes 0.35GiB; PyTorch activation peak memory takes 1.24GiB;
the rest of the memory reserved for KV Cache is 54.62GiB.
INFO 03-14 09:13:08 [kv_cache_utils.py:566] GPU KV cache size: 447,088 tokens
INFO 03-14 09:13:08 [kv_cache_utils.py:570] Maximum concurrency for 32,768 tokens per request: 13.64x
`
const SGLANG = `
[2025-03-14 09:20:11] load_weight_end. type=Llama, dtype=torch.bfloat16, avail mem=63.12 GB, mem usage=15.06 GB.
[2025-03-14 09:20:12] KV Cache is allocated. #tokens: 371661, K size: 12.74 GB, V size: 12.74 GB
[2025-03-14 09:20:12] max_total_num_tokens=371661, chunked_prefill_size=8192, mem_fraction_static=0.88
`

test('vLLM V1 memory summary is parsed into every component', () => {
  const p = parseVllmLog(VLLM_V1)
  assert.equal(p.engine, 'vllm')
  assert.ok(Math.abs(p.weightBytes! - 14.99 * GiB) < 1)
  assert.ok(Math.abs(p.nonTorchBytes! - 0.35 * GiB) < 1)
  assert.ok(Math.abs(p.activationBytes! - 1.24 * GiB) < 1)
  assert.ok(Math.abs(p.kvCacheBytes! - 54.62 * GiB) < 1)
  assert.equal(p.memoryUtilization, 0.9)
  assert.equal(p.maxTokens, 447088)
})

test('SGLang KV allocation line is parsed', () => {
  const p = parseSglangLog(SGLANG)
  assert.equal(p.maxTokens, 371661)
  assert.ok(Math.abs(p.kvCacheBytes! - 25.48e9) < 1e6)
  assert.equal(p.memoryUtilization, 0.88)
})

test('a missing field parses as undefined, never as zero', () => {
  const p = parseVllmLog('INFO nothing useful here')
  assert.equal(p.weightBytes, undefined)
  assert.equal(p.maxTokens, undefined)
})

test('an unknown engine is refused rather than guessed at', () => {
  assert.throws(() => parseLog('trtllm', VLLM_V1), /No log parser/)
})
