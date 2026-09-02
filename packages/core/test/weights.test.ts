import { test } from 'node:test'
import assert from 'node:assert/strict'
import { normalizeConfig } from '../src/model.ts'
import { parameterCounts, weightBytes, activeParameters, weightBytesPerDevice } from '../src/weights.ts'
import { getModelSnapshot, listModels } from '../src/data.ts'

const spec = (id: string) => {
  const s = getModelSnapshot(id)
  return normalizeConfig(s.config, { id, measuredWeightBytes: s.measuredWeightBytes })
}

test('Llama-3.1-8B parameter count, computed by hand', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  // vocab 128256, hidden 4096, layers 32, heads 32, kv heads 8, head_dim 128, ffn 14336, untied.
  const embed = 128256 * 4096                        //   525,336,576
  const attn = (4096 * 4096) + 2 * (4096 * 1024) + (4096 * 4096) //  41,943,040 per layer
  const mlp = 3 * 4096 * 14336                       // 176,160,768 per layer
  const norm = 2 * 4096                              //       8,192 per layer
  const total = 2 * embed + 32 * (attn + mlp + norm) + 4096
  assert.equal(total, 8_030_261_248)

  const p = parameterCounts(m)
  assert.equal(p.embedding, embed)
  assert.equal(p.lm_head, embed)
  assert.equal(p.attn, 32 * attn)
  assert.equal(p.mlp, 32 * mlp)
  assert.equal(p.total, 8_030_261_248)
  assert.equal(weightBytes(m, { quant: 'bf16', preferMeasured: false }).totalBytes, 16_060_522_496)
})

test('the derived count matches the checkpoint byte count within 1%', () => {
  for (const id of listModels()) {
    const m = spec(id)
    if (!m.measuredWeightBytes) continue
    if (m.warnings.some((w) => w.code === 'vision_tower_excluded')) continue // index covers the vision tower too
    const quant = (m.checkpointQuant ?? 'bf16') as 'bf16' | 'fp8' | 'mxfp4'
    const derived = weightBytes(m, { quant, preferMeasured: false }).totalBytes
    const err = Math.abs(derived - m.measuredWeightBytes) / m.measuredWeightBytes
    assert.ok(err < 0.01, `${id}: derived ${derived} vs measured ${m.measuredWeightBytes} = ${(err * 100).toFixed(2)}% error`)
  }
})

test('the measured path is preferred and reported as such', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const r = weightBytes(m, { quant: 'bf16' })
  assert.equal(r.method, 'measured')
  assert.equal(r.totalBytes, m.measuredWeightBytes)
  assert.equal(weightBytes(m, { quant: 'awq-int4' }).method, 'derived')
})

test('a multimodal checkpoint falls back to the derived text-only count', () => {
  const r = weightBytes(spec('google/gemma-3-27b-it'), { quant: 'bf16' })
  assert.equal(r.method, 'derived')
  assert.ok(r.warnings.some((w) => w.code === 'measured_weights_unusable'))
})

test('int4 group scales are counted, so awq is above the nominal 4 bits', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const p = parameterCounts(m)
  const r = weightBytes(m, { quant: 'awq-int4', preferMeasured: false })
  // embeddings + lm_head + norms stay fp16; linears go to 4 bits + (16+4)/128 bits of scale/zero.
  const fp16Params = p.embedding + p.lm_head + p.norm
  const quantParams = p.attn + p.mlp + p.router
  const expected = fp16Params * 2 + quantParams * 0.5 + Math.ceil(quantParams / 128) * 2.5
  assert.ok(Math.abs(r.totalBytes - expected) < 8, `${r.totalBytes} vs ${expected}`)
  assert.ok(r.totalBytes > (p.total * 4) / 8, 'must exceed the nominal 4-bit size')
})

test('GGUF uses measured bits-per-weight, not the nominal bit width', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const q4 = weightBytes(m, { quant: 'gguf:Q4_K_M', preferMeasured: false }).totalBytes
  assert.equal(q4, (parameterCounts(m).total * 4.85) / 8)
  assert.ok(q4 > (parameterCounts(m).total * 4) / 8, 'Q4_K_M is 4.85 bpw, not 4')
})

test('all MoE experts are resident; only FLOPs use top-k', () => {
  const m = spec('mistralai/Mixtral-8x7B-Instruct-v0.1')
  const p = parameterCounts(m)
  // 8 experts x 3 x 4096 x 14336 per layer, all resident
  assert.equal(p.mlp, 32 * 8 * 3 * 4096 * 14336)
  assert.ok(p.total > 46e9 && p.total < 47e9)
  // top-2 of 8 active
  const active = activeParameters(m)
  assert.equal(p.total - active, 32 * 6 * 3 * 4096 * 14336)
  assert.ok(active > 12e9 && active < 13e9)
})

test('weights shard across TP and PP', () => {
  assert.equal(weightBytesPerDevice(80, { tp: 4, pp: 2 }), 10)
  assert.equal(weightBytesPerDevice(80, { tp: 1, pp: 1 }), 80)
})

test('a checkpoint whose metadata disagrees with the count is not trusted silently', () => {
  const m = spec('meta-llama/Llama-3.1-8B-Instruct')
  const lying = { ...m, measuredWeightBytes: m.measuredWeightBytes! * 2 }
  const r = weightBytes(lying, { quant: 'bf16' })
  assert.equal(r.method, 'derived')
  assert.ok(r.warnings.some((w) => w.code === 'measured_weights_disagree'))
  assert.equal(r.totalBytes, 16_060_522_496)
})

test('DeepSeek-V3 fp8 weights land within 1% of the real checkpoint', () => {
  const m = spec('deepseek-ai/DeepSeek-V3')
  const r = weightBytes(m, { quant: 'fp8', preferMeasured: false })
  const err = Math.abs(r.totalBytes - m.measuredWeightBytes!) / m.measuredWeightBytes!
  assert.ok(err < 0.01, `${(err * 100).toFixed(2)}% off the 688.6 GB checkpoint`)
})
