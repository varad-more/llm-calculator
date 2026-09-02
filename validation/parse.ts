// Parses what a serving engine prints at startup into the components we predict.
// Formats change between releases, so every pattern is tried and misses are reported
// rather than defaulted.

export interface ParsedLog {
  engine: 'vllm' | 'sglang'
  weightBytes?: number
  nonTorchBytes?: number
  activationBytes?: number
  kvCacheBytes?: number
  gpuBlocks?: number
  maxTokens?: number
  memoryUtilization?: number
}

const GiB = 2 ** 30
const num = (s: string) => Number(s.replace(/,/g, ''))

/**
 * vLLM V1 prints one summary line, e.g.
 *   "Memory profiling takes 12.34 seconds. the current vLLM instance can use
 *    total_gpu_memory (79.11GiB) x gpu_memory_utilization (0.90) = 71.20GiB;
 *    model weights take 14.99GiB; non_torch_memory takes 0.35GiB;
 *    PyTorch activation peak memory takes 1.24GiB; the rest of the memory reserved
 *    for KV Cache is 54.62GiB."
 * and separately "GPU KV cache size: 447,088 tokens".
 */
export function parseVllmLog(log: string): ParsedLog {
  const out: ParsedLog = { engine: 'vllm' }
  const grab = (re: RegExp): number | undefined => {
    const m = log.match(re)
    return m ? num(m[1]!) : undefined
  }
  const gib = (re: RegExp): number | undefined => {
    const v = grab(re)
    return v === undefined ? undefined : v * GiB
  }
  out.weightBytes = gib(/model weights take ([\d.,]+)\s*GiB/i)
  out.nonTorchBytes = gib(/non[_ ]torch[_ ]memory takes ([\d.,]+)\s*GiB/i)
  out.activationBytes = gib(/activation peak memory takes ([\d.,]+)\s*GiB/i)
  out.kvCacheBytes = gib(/(?:reserved for KV Cache is|KV cache size:?) ([\d.,]+)\s*GiB/i)
  out.memoryUtilization = grab(/gpu_memory_utilization \(([\d.]+)\)/i)
  out.gpuBlocks = grab(/(?:GPU blocks|# GPU blocks):?\s*([\d,]+)/i)
  out.maxTokens = grab(/GPU KV cache size:?\s*([\d,]+) tokens/i)
  if (out.maxTokens === undefined && out.gpuBlocks !== undefined) {
    const bs = grab(/block[_ ]size[=: ]+(\d+)/i) ?? 16
    out.maxTokens = out.gpuBlocks * bs
  }
  return out
}

/** SGLang prints "KV Cache is allocated. #tokens: 371661, K size: 12.74 GB, V size: 12.74 GB". */
export function parseSglangLog(log: string): ParsedLog {
  const out: ParsedLog = { engine: 'sglang' }
  const m = log.match(/#tokens:\s*([\d,]+)/i)
  if (m) out.maxTokens = num(m[1]!)
  const k = log.match(/K size:\s*([\d.]+)\s*GB/i)
  const v = log.match(/V size:\s*([\d.]+)\s*GB/i)
  if (k && v) out.kvCacheBytes = (num(k[1]!) + num(v[1]!)) * 1e9
  const w = log.match(/(?:load_weight_end|Load weight end).*?avail mem=([\d.]+) GB/i)
  if (w) out.weightBytes = num(w[1]!) * 1e9
  const f = log.match(/mem[_-]fraction[_-]static[=: ]+([\d.]+)/i)
  if (f) out.memoryUtilization = num(f[1]!)
  return out
}

/** Dispatch on the declared engine. */
export function parseLog(engine: string, log: string): ParsedLog {
  if (engine === 'sglang') return parseSglangLog(log)
  if (engine === 'vllm') return parseVllmLog(log)
  throw new Error(`No log parser for engine "${engine}". Add one in validation/parse.ts.`)
}
