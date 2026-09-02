import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { listGpus, getGpu, defaultAssumptions, resolveAssumptions, listModels } from '../src/data.ts'
import { UnknownEntityError } from '../src/errors.ts'

// Vendor datasheets headline the 2:1 structured-sparsity number. Anyone who pastes it in
// doubles every throughput estimate. These are the SPARSE bf16 figures; dense must be half.
const SPARSE_BF16_TFLOPS: Record<string, number> = {
  'h100-sxm-80': 1978.9, 'h100-pcie-80': 1513, 'h200-sxm-141': 1978.9, 'b200-sxm-192': 4500,
  'a100-sxm-80': 624, 'a100-pcie-80': 624, 'a100-sxm-40': 624, 'l40s-48': 733,
  'l4-24': 242, 'a10g-24': 250, 'rtx4090-24': 330.4, 'mi300x-192': 2614.9,
}
const SPARSE_FP8_TFLOPS: Record<string, number> = {
  'h100-sxm-80': 3957.8, 'h100-pcie-80': 3026, 'h200-sxm-141': 3957.8, 'b200-sxm-192': 9000,
  'l40s-48': 1466, 'l4-24': 485, 'rtx4090-24': 1321.2, 'mi300x-192': 5229.8,
}

test('every GPU entry is schema-complete and sourced', () => {
  for (const g of listGpus()) {
    assert.ok(g.id && g.name && g.vendor, `${g.id}: identity`)
    assert.ok(g.vramBytes > 0, `${g.id}: vramBytes`)
    assert.ok(g.memBandwidthBytesPerSec > 1e9, `${g.id}: bandwidth looks like it is not in bytes/s`)
    assert.ok(g.tflopsDense.fp16 && g.tflopsDense.bf16, `${g.id}: needs fp16 and bf16 dense TFLOPS`)
    assert.ok(g.tflopsDense.fp32, `${g.id}: needs fp32`)
    assert.ok(g.interconnect?.bidirectionalBytesPerSec > 0, `${g.id}: interconnect`)
    assert.match(g.source_url, /^https:\/\//, `${g.id}: source_url must be a URL`)
    assert.equal(g.sparsity_excluded, true, `${g.id}: must assert sparsity exclusion`)
  }
})

test('no GPU quotes a sparsity-inflated TFLOPS number', () => {
  for (const g of listGpus()) {
    const sparse = SPARSE_BF16_TFLOPS[g.id]
    assert.ok(sparse !== undefined, `${g.id} has no known dense ceiling in the test table; add one`)
    assert.ok(
      g.tflopsDense.bf16! <= (sparse / 2) * 1.001,
      `${g.id}: bf16 ${g.tflopsDense.bf16} exceeds the dense ceiling ${sparse / 2} (looks like the 2:1 sparsity figure)`,
    )
    // fp8 is checked against its own sparse figure rather than as a multiple of bf16:
    // consumer Ada halves FP16-with-FP32-accumulate, so the RTX 4090's fp8/bf16 ratio is 4,
    // not 2, and a bf16-derived bound would either fail it or be uselessly loose.
    if (g.tflopsDense.fp8) {
      const sparseFp8 = SPARSE_FP8_TFLOPS[g.id]
      assert.ok(sparseFp8 !== undefined, `${g.id} lists fp8 but has no known dense fp8 ceiling in the test table`)
      assert.ok(
        g.tflopsDense.fp8 <= (sparseFp8 / 2) * 1.001,
        `${g.id}: fp8 ${g.tflopsDense.fp8} exceeds the dense ceiling ${sparseFp8 / 2}`,
      )
    }
  }
})

test('unknown ids fail loudly with the known list', () => {
  assert.throws(() => getGpu('h100-sxm-8000'), UnknownEntityError)
  assert.throws(() => resolveAssumptions({ mbu: 0.5 } as any), UnknownEntityError)
})

test('every assumption carries value, rationale, source and confidence', () => {
  for (const [k, a] of Object.entries(defaultAssumptions())) {
    assert.equal(typeof a.value, 'number', `${k}: value`)
    assert.ok(a.rationale.length > 30, `${k}: rationale must explain the number`)
    assert.match(a.source_url, /^https:\/\//, `${k}: source_url`)
    assert.ok(['low', 'medium', 'high'].includes(a.confidence), `${k}: confidence`)
  }
})

test('user overrides replace the value and record what they replaced', () => {
  const a = resolveAssumptions({ mbu_decode: 0.9 })
  assert.equal(a.mbu_decode.value, 0.9)
  assert.match(a.mbu_decode.rationale, /user override \(was 0\.65/)
  assert.equal(a.mfu_prefill.value, defaultAssumptions().mfu_prefill.value)
})

test('generated.ts is in sync with data/ (run `pnpm gen`)', () => {
  const before = readFileSync(new URL('../src/generated.ts', import.meta.url), 'utf8')
  execFileSync('node', ['scripts/gen-data.mjs'], { cwd: new URL('../../../', import.meta.url).pathname })
  const after = readFileSync(new URL('../src/generated.ts', import.meta.url), 'utf8')
  assert.equal(before, after, 'generated.ts is stale; run `pnpm gen` and commit')
})

test('model snapshots are present', () => {
  assert.ok(listModels().length >= 20)
  assert.ok(listModels().includes('meta-llama/Llama-3.1-8B-Instruct'))
})
