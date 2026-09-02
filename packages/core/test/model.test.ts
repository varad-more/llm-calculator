import { test } from 'node:test'
import assert from 'node:assert/strict'
import { normalizeConfig, layersOnStage, layersPerStage } from '../src/model.ts'
import { getModelSnapshot } from '../src/data.ts'
import { IncompleteConfigError } from '../src/errors.ts'

const spec = (id: string) => {
  const s = getModelSnapshot(id)
  return normalizeConfig(s.config, { id, measuredWeightBytes: s.measuredWeightBytes })
}

test('GQA / MHA / MQA are told apart by head counts', () => {
  assert.equal(spec('meta-llama/Llama-3.1-8B-Instruct').layers[0]!.kind, 'gqa') // 32 q, 8 kv
  assert.equal(spec('microsoft/Phi-3.5-mini-instruct').layers[0]!.kind, 'mha') // 32 q, 32 kv
})

test('Gemma 3 is 5 sliding-window layers to 1 global, not homogeneous', () => {
  const m = spec('google/gemma-3-27b-it')
  assert.equal(m.numLayers, 62)
  const sliding = m.layers.filter((l) => l.kind === 'sliding_window')
  assert.equal(sliding.length, 52)
  assert.equal(m.layers.filter((l) => l.kind === 'gqa').length, 10)
  assert.equal(sliding[0]!.windowSize, 1024)
  // pattern 6: layer index 5, 11, ... are the global ones
  assert.equal(m.layers[5]!.kind, 'gqa')
  assert.equal(m.layers[4]!.kind, 'sliding_window')
})

test('Gemma 2 alternates every other layer', () => {
  const m = spec('google/gemma-2-9b-it')
  assert.equal(m.layers.filter((l) => l.kind === 'sliding_window').length, 21)
  assert.equal(m.layers[0]!.kind, 'sliding_window')
  assert.equal(m.layers[1]!.kind, 'gqa')
})

test('gpt-oss takes its per-layer pattern from the explicit layer_types array', () => {
  const m = spec('openai/gpt-oss-120b')
  assert.equal(m.layers[0]!.kind, 'sliding_window')
  assert.equal(m.layers[0]!.windowSize, 128)
  assert.equal(m.layers[1]!.kind, 'gqa')
  assert.ok(m.layers.every((l) => l.mlp === 'moe'))
  assert.deepEqual(m.unquantizedClasses?.sort(), ['attn', 'embedding', 'lm_head', 'norm', 'router'])
})

test('DeepSeek V3 is MLA with the first 3 MLP blocks dense, plus an MTP module', () => {
  const m = spec('deepseek-ai/DeepSeek-V3')
  assert.ok(m.mla)
  assert.equal(m.mla!.kvLoraRank, 512)
  assert.equal(m.mla!.qkRopeHeadDim, 64)
  assert.equal(m.layers.filter((l) => l.mlp === 'dense').length, 3)
  assert.equal(m.layers.filter((l) => l.mlp === 'moe').length, 58)
  assert.equal(m.mtpLayers, 1)
  assert.equal(m.moe!.numExperts, 256)
  assert.equal(m.moe!.topK, 8)
  assert.equal(m.moe!.sharedExperts, 1)
})

test('a sliding_window longer than the context is not a sliding-window model', () => {
  // Phi-3.5-mini ships sliding_window=262144 with max_position_embeddings=131072.
  assert.ok(spec('microsoft/Phi-3.5-mini-instruct').layers.every((l) => l.kind === 'mha'))
  // Qwen2.5 ships sliding_window but sets use_sliding_window=false.
  assert.ok(spec('Qwen/Qwen2.5-7B-Instruct').layers.every((l) => l.kind === 'gqa'))
})

test('a missing field throws IncompleteConfigError naming the field', () => {
  const bad = { model_type: 'llama', num_hidden_layers: 32, hidden_size: 4096, num_attention_heads: 32 }
  assert.throws(() => normalizeConfig(bad), IncompleteConfigError)
  let err: IncompleteConfigError | undefined
  try { normalizeConfig(bad) } catch (e) { err = e as IncompleteConfigError }
  assert.equal(err!.field, 'num_key_value_heads')
  assert.equal(err!.modelType, 'llama')
  assert.match(err!.message, /num_key_value_heads/)
})

test('fields taken from transformers defaults are recorded, never silent', () => {
  const m = spec('google/gemma-3-27b-it')
  assert.ok(m.assumed.some((a) => a.startsWith('vocab_size=262208')))
  assert.ok(m.assumed.some((a) => a.startsWith('sliding_window_pattern=6')))
  assert.ok(m.warnings.some((w) => w.code === 'vision_tower_excluded'))
})

test('pipeline stages split the layer list', () => {
  const m = spec('meta-llama/Llama-3.1-70B-Instruct')
  assert.equal(layersPerStage(m.numLayers, 4), 20)
  assert.equal(layersOnStage(m, 4, 0).length, 20)
  assert.equal(layersOnStage(m, 1, 0).length, 80)
  assert.equal(layersPerStage(61, 8), 8) // ceil: the widest stage sets the budget
})
