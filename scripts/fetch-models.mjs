// Build-time snapshot of HF configs into data/models/. Gated repos (meta-llama/*)
// return 401 without a token, so ungated mirrors are listed as fallbacks.
import { writeFileSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = new URL('..', import.meta.url).pathname
const LIST = JSON.parse(readFileSync(join(ROOT, 'data/model-list.json'), 'utf8'))
const token = process.env.HF_TOKEN
const hdr = token ? { authorization: `Bearer ${token}` } : {}

async function hf(repo, file) {
  const r = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`, { headers: hdr })
  if (!r.ok) throw new Error(`${r.status} ${repo}/${file}`)
  return r.json()
}

// metadata.total_size in model.safetensors.index.json is NOT always the on-disk size:
// DeepSeek-V3 reports 1369 GB for a checkpoint whose files sum to 688 GB (the index was
// generated against a bf16 copy). The summed file listing is authoritative, so that is
// what we store; the index figure is kept alongside it for reference only.
async function repoWeightBytes(repo, index) {
  const r = await fetch(`https://huggingface.co/api/models/${repo}/tree/main?recursive=1`, { headers: hdr })
  if (!r.ok) throw new Error(`${r.status} ${repo}/tree`)
  const tree = await r.json()
  const sizeOf = new Map(tree.map((f) => [f.path, f.size ?? 0]))
  // The index's weight_map names exactly the shards the model loads. Summing those file
  // sizes avoids both traps: a wrong metadata.total_size, and repos that ship a second
  // copy of the weights under original/ or metal/ (gpt-oss doubles otherwise).
  const shards = index?.weight_map
    ? [...new Set(Object.values(index.weight_map))]
    : sizeOf.has('model.safetensors') ? ['model.safetensors'] : []
  if (!shards.length) return null
  const missing = shards.filter((f) => !sizeOf.has(f))
  if (missing.length) throw new Error(`${repo}: index names ${missing.length} shards not in the file tree`)
  const total = shards.reduce((a, f) => a + sizeOf.get(f), 0)
  return total > 0 ? { bytes: total, files: shards.length } : null
}

async function snapshot(entry) {
  const repos = [entry.repo, ...(entry.mirrors ?? [])]
  let lastErr
  for (const repo of repos) {
    try {
      const config = await hf(repo, 'config.json')
      let index = null
      try {
        index = await hf(repo, 'model.safetensors.index.json')
      } catch { /* single-shard models have no index */ }
      const measured = await repoWeightBytes(repo, index)
      return {
        id: entry.repo,
        fetchedFrom: repo,
        gated: repo !== entry.repo,
        source_url: `https://huggingface.co/${repo}/resolve/main/config.json`,
        measuredWeightBytes: measured?.bytes ?? null,
        measuredWeightSource: measured ? `https://huggingface.co/api/models/${repo}/tree/main (${measured.files} weight files)` : null,
        indexTotalSize: index?.metadata?.total_size ?? null,
        config,
      }
    } catch (e) { lastErr = e }
  }
  throw lastErr
}

const ok = [], failed = []
for (const entry of LIST.models) {
  try {
    const snap = await snapshot(entry)
    const name = entry.repo.replace(/\//g, '__') + '.json'
    writeFileSync(join(ROOT, 'data/models', name), JSON.stringify(snap, null, 2) + '\n')
    ok.push(`${entry.repo}${snap.gated ? ` (via ${snap.fetchedFrom})` : ''}`)
  } catch (e) { failed.push(`${entry.repo}: ${e.message}`) }
}
console.log(`snapshotted ${ok.length}:\n  ` + ok.join('\n  '))
if (failed.length) console.log(`failed ${failed.length}:\n  ` + failed.join('\n  '))
