// fixtures/golden/_cases.json -> one expected-output file per case.
// Both language ports execute these; a diff means the two have drifted or behaviour changed.
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { size } from '../packages/core/src/plan.ts'

const ROOT = new URL('..', import.meta.url).pathname
const { cases } = JSON.parse(readFileSync(join(ROOT, 'fixtures/golden/_cases.json'), 'utf8'))

for (const c of cases) {
  const r = size(c.request)
  const p = r.plan
  const golden = {
    name: c.name,
    request: c.request,
    expected: {
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
    },
  }
  writeFileSync(join(ROOT, 'fixtures/golden', `${c.name}.json`), JSON.stringify(golden, null, 2) + '\n')
}
console.log(`wrote ${cases.length} golden fixtures`)
