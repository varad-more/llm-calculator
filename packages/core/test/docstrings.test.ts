import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

const SRC = new URL('../src/', import.meta.url).pathname
// Every module that carries math. types.ts, errors.ts, generated.ts and data.ts hold no
// formulas (data.ts is lookups over data/, and is covered by data.test.ts instead).
const MATH_MODULES = [
  'model.ts', 'weights.ts', 'kv.ts', 'overhead.ts', 'throughput.ts', 'plan.ts',
  'validation.ts', 'format.ts', 'lora.ts', 'speculative.ts', 'disagg.ts', 'reverse.ts',
  'cost.ts',
  ...readdirSync(join(SRC, 'engines')).filter((f) => f !== 'types.ts').map((f) => `engines/${f}`),
]

function exportedFunctions(source: string): { name: string; doc: string }[] {
  const out: { name: string; doc: string }[] = []
  const re = /(\/\*\*[\s\S]*?\*\/\s*)?export function (\w+)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(source))) out.push({ name: m[2]!, doc: m[1] ?? '' })
  return out
}

test('every exported math function has a docstring linking to docs/MATH.md', () => {
  const missing: string[] = []
  for (const file of MATH_MODULES) {
    const src = readFileSync(join(SRC, file), 'utf8')
    for (const fn of exportedFunctions(src)) {
      if (!fn.doc) missing.push(`${file}:${fn.name} has no docstring`)
      else if (!fn.doc.includes('docs/MATH.md#')) missing.push(`${file}:${fn.name} does not link to docs/MATH.md#<anchor>`)
    }
  }
  assert.deepEqual(missing, [])
})

test('every docs/MATH.md anchor referenced from code exists in the document', () => {
  const math = readFileSync(new URL('../../../docs/MATH.md', import.meta.url).pathname, 'utf8')
  const headings = new Set(
    math.split('\n').filter((l) => l.startsWith('#'))
      .map((l) => l.replace(/^#+\s*/, '').toLowerCase().replace(/[^a-z0-9 -]/g, '').trim().replace(/\s+/g, '-')),
  )
  const missing = new Set<string>()
  for (const file of MATH_MODULES) {
    const src = readFileSync(join(SRC, file), 'utf8')
    for (const m of src.matchAll(/docs\/MATH\.md#([a-z0-9-]+)/g)) {
      if (!headings.has(m[1]!)) missing.add(`${m[1]} (referenced from ${file})`)
    }
  }
  assert.deepEqual([...missing], [], 'docs/MATH.md is missing sections referenced from the code')
})

test('core has no runtime dependencies and does no I/O', () => {
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url).pathname, 'utf8'))
  assert.deepEqual(pkg.dependencies, {})
  for (const file of [...MATH_MODULES, 'types.ts', 'errors.ts', 'index.ts', 'data.ts']) {
    const src = readFileSync(join(SRC, file), 'utf8')
    for (const banned of ['node:fs', 'node:http', 'fetch(', 'Date.now(', 'Math.random(', 'process.env']) {
      assert.ok(!src.includes(banned), `${file} must stay pure but contains "${banned}"`)
    }
  }
})
