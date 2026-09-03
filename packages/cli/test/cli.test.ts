import { test } from 'node:test'
import assert from 'node:assert/strict'
import { main } from '../src/cli.ts'

test('CLI reports invalid arguments as usage errors', async () => {
  const original = console.error
  console.error = () => {}
  try {
    assert.equal(await main(['plan', '--unknown']), 2)
    assert.equal(await main([
      'plan', '--model', 'meta-llama/Llama-3.1-8B-Instruct', '--gpu', 'h100-sxm-80', '--context=0',
    ]), 2)
  } finally {
    console.error = original
  }
})
