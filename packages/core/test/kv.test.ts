import { test } from 'node:test'
import assert from 'node:assert/strict'
import { normalizeConfig } from '../src/model.ts'
import { getModelSnapshot } from '../src/data.ts'
import { kvBytesPerTokenPerLayer, kvCacheBytes, kvDtypeBytes, effectiveTokensForLayer, kvBytesForSequence } from '../src/kv.ts'
import type { Workload } from '../src/types.ts'

const spec = (id: string) => normalizeConfig(getModelSnapshot(id).config, { id })
const wl = (o: Partial<Workload> = {}): Workload =>
  ({ concurrency: 1, avgSeqLen: 32768, maxModelLen: 32768, chunkTokens: 2048, ...o })
const OPT = { kvDtype: 'fp16' as const, blockSize: 16 }

test('GQA per-token KV, computed by hand', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  // 2 (K and V) x 8 kv heads x 128 head_dim x 2 bytes = 4096 B per layer
  assert.equal(kvBytesPerTokenPerLayer(m.layers[0]!, m, 2), 4096)
  const r = kvCacheBytes(m, wl(), { tp: 1, pp: 1 }, OPT)
  assert.equal(r.perTokenBytesPerDevice, 32 * 4096) // 128 KiB/token
  assert.equal(r.perTokenBytesPerDevice, 131072)
  assert.equal(r.totalBytesPerDevice, 131072 * 32768) // exactly 4 GiB at 32k
})

test('MLA caches one latent: no factor of 2, no kv heads', () => {
  const m = spec('deepseek-ai/DeepSeek-V3')
  // (kv_lora_rank 512 + qk_rope_head_dim 64) x 2 bytes = 1152 B per layer
  assert.equal(kvBytesPerTokenPerLayer(m.layers[0]!, m, 2), 1152)
  const r = kvCacheBytes(m, wl(), { tp: 1, pp: 1 }, OPT)
  assert.equal(r.perTokenBytesPerDevice, 61 * 1152) // 70,272 B/token
  // 128 kv heads x 128 head_dim would have been 64 KiB/token/layer: MLA is 57x smaller
  assert.ok(2 * 128 * 128 * 2 / 1152 > 56)
})

test('sliding-window layers store only their window', () => {
  const m = spec('google/gemma-3-27b-it')
  const r = kvCacheBytes(m, wl({ avgSeqLen: 32768 }), { tp: 1, pp: 1 }, OPT)
  const perLayerPerToken = 2 * 16 * 128 * 2 // 8192
  const expected = 10 * perLayerPerToken * 32768 + 52 * perLayerPerToken * 1024
  assert.equal(r.totalBytesPerDevice, expected)
  // a naive homogeneous model would predict 5.3x more
  assert.ok((62 * perLayerPerToken * 32768) / expected > 5)
})

test('paged allocation charges whole blocks', () => {
  assert.equal(effectiveTokensForLayer(wl({ avgSeqLen: 100, concurrency: 1 }), 16), 112) // ceil(100/16)*16
  assert.equal(effectiveTokensForLayer(wl({ avgSeqLen: 100, concurrency: 4 }), 16), 448)
  assert.equal(effectiveTokensForLayer(wl({ avgSeqLen: 100 }), 16, 32), 32) // window caps it
})

test('a shared prefix is stored once', () => {
  const w = wl({ concurrency: 10, avgSeqLen: 2000, prefixCache: { enabled: true, hitRate: 1, sharedPrefixTokens: 1000 } })
  // 1000 shared + 10 x 1000 unique, vs 10 x 2000 without the cache
  assert.equal(effectiveTokensForLayer(w, 16), 1008 + 10 * 1008)
  assert.equal(effectiveTokensForLayer({ ...w, prefixCache: undefined }, 16), 10 * 2000)
  // a 50% hit rate recovers half the sharing
  const half = effectiveTokensForLayer({ ...w, prefixCache: { enabled: true, hitRate: 0.5, sharedPrefixTokens: 1000 } }, 16)
  assert.equal(half, 1008 + 10 * 1504)
})

test('TP shards KV heads, and stops helping past num_key_value_heads', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct') // 8 kv heads
  const at = (tp: number) => kvCacheBytes(m, wl(), { tp, pp: 1 }, OPT)
  assert.equal(at(2).totalBytesPerDevice, at(1).totalBytesPerDevice / 2)
  assert.equal(at(8).totalBytesPerDevice, at(1).totalBytesPerDevice / 8)
  assert.equal(at(16).totalBytesPerDevice, at(8).totalBytesPerDevice, 'TP=16 buys nothing past 8 KV heads')
  assert.ok(at(16).warnings.some((w) => w.code === 'tp_exceeds_kv_heads'))
  assert.ok(!at(8).warnings.some((w) => w.code === 'tp_exceeds_kv_heads'))
})

test('MLA KV is replicated across TP ranks, and says so', () => {
  const m = spec('deepseek-ai/DeepSeek-V3')
  const at = (tp: number) => kvCacheBytes(m, wl(), { tp, pp: 1 }, OPT)
  assert.equal(at(8).totalBytesPerDevice, at(1).totalBytesPerDevice)
  assert.ok(at(8).warnings.some((w) => w.code === 'mla_kv_replicated'))
})

test('PP splits KV by layer', () => {
  const m = spec('meta-llama/Llama-3.1-70B-Instruct')
  const one = kvCacheBytes(m, wl(), { tp: 1, pp: 1 }, OPT)
  const four = kvCacheBytes(m, wl(), { tp: 1, pp: 4 }, OPT)
  assert.equal(four.totalBytesPerDevice, one.totalBytesPerDevice / 4)
})

test('fp8 KV halves the cache', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  assert.equal(kvDtypeBytes('fp8'), 1)
  const fp16 = kvCacheBytes(m, wl(), { tp: 1, pp: 1 }, OPT)
  const fp8 = kvCacheBytes(m, wl(), { tp: 1, pp: 1 }, { kvDtype: 'fp8', blockSize: 16 })
  assert.equal(fp8.totalBytesPerDevice, fp16.totalBytesPerDevice / 2)
})

test('single-sequence KV cost respects windows', () => {
  const m = spec('google/gemma-3-27b-it')
  const full = kvBytesForSequence(m, 32768, { tp: 1, pp: 1 }, OPT)
  const short = kvBytesForSequence(m, 1024, { tp: 1, pp: 1 }, OPT)
  assert.ok(full / short < 32, 'sliding layers stop growing, so cost is sublinear in context')
})
