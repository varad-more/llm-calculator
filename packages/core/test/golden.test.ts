import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { size } from '../src/plan.ts'

const DIR = new URL('../../../fixtures/golden/', import.meta.url).pathname
const files = readdirSync(DIR).filter((f) => f.endsWith('.json') && !f.startsWith('_'))

// Executed identically by python/tests/test_golden.py. Drift between the ports fails there.
test('golden fixtures reproduce exactly', () => {
  assert.ok(files.length >= 16, `only ${files.length} fixtures found`)
  for (const file of files) {
    const g = JSON.parse(readFileSync(join(DIR, file), 'utf8'))
    const r = size(g.request)
    const p = r.plan
    const actual: Record<string, unknown> = {
      weightBytes: p.weights.totalBytes,
      weightMethod: p.weights.method,
      totalParams: p.weights.params.total,
      weightBytesPerDevice: p.weightBytesPerDevice,
      kvPerTokenBytesPerDevice: p.kv.perTokenBytesPerDevice,
      kvBytesPerDevice: p.kv.totalBytesPerDevice,
      activationBytes: p.overhead.activationBytes,
      overheadTotalBytes: p.overhead.totalBytes,
      usableVramBytes: p.usableVramBytes,
      availableKvBytes: p.availableKvBytes,
      numBlocks: p.numBlocks,
      maxTokens: p.maxTokens,
      fits: p.fits,
      autofix: p.autofix ?? null,
      warningCodes: p.warnings.map((w) => w.code).sort(),
      flags: r.flags,
      label: r.label,
      decodeStepSeconds: r.throughput.decode.stepSeconds,
      decodeTokensPerSecond: r.throughput.decode.tokensPerSecond,
      prefillFlops: r.throughput.prefill.flops,
      ttftSeconds: r.throughput.prefill.ttftSeconds,
      bound: r.throughput.bound,
    }
    for (const [k, want] of Object.entries(g.expected)) {
      const got = actual[k]
      if (typeof want === 'number' && typeof got === 'number') {
        const tol = Math.max(Math.abs(want) * 1e-9, 1e-12)
        assert.ok(Math.abs(got - want) <= tol, `${file}: ${k} = ${got}, expected ${want}`)
      } else {
        assert.deepEqual(got, want, `${file}: ${k}`)
      }
    }
  }
})
