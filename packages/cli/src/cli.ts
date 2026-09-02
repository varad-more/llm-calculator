import { parseArgs } from 'node:util'
import {
  size, listGpus, listModels, getGpu, defaultAssumptions, resolveModel,
  gib, bytes as fmtBytes, seconds, count, allValidations,
  IncompleteConfigError, UnknownEntityError,
} from '@llmsize/core'

const color = !process.env.NO_COLOR && process.stdout.isTTY
const c = (code: string, s: string) => (color ? `\x1b[${code}m${s}\x1b[0m` : s)
const bold = (s: string) => c('1', s)
const dim = (s: string) => c('2', s)
const green = (s: string) => c('32', s)
const red = (s: string) => c('31', s)
const yellow = (s: string) => c('33', s)

const USAGE = `${bold('llmsize')} — inference sizing and serving-config generator

${bold('USAGE')}
  llmsize plan --model <id> --gpu <id> [options]
  llmsize gpus | models | assumptions | validation

${bold('PLAN OPTIONS')}
  --model <id>          HF repo id, or a path to a config.json
  --gpu <id>            GPU id (llmsize gpus)
  --engine <name>       vllm | sglang | trtllm | llamacpp   (default vllm)
  --tp <n> --pp <n>     tensor / pipeline parallel size     (default 1)
  --context <n>         max sequence length (--max-model-len)
  --concurrency <n>     concurrent sequences (--max-num-seqs)
  --avg-seq-len <n>     mean tokens per sequence            (default: --context)
  --quant <scheme>      bf16 fp16 fp8 int8 awq-int4 gptq-int4 mxfp4 gguf:Q4_K_M
  --kv-dtype <dtype>    fp16 bf16 fp8 int8                  (default fp16)
  --chunk <n>           --max-num-batched-tokens            (default 8192)
  --util <f>            gpu memory utilization              (default 0.90)
  --block-size <n>      paged block size                    (default 16)
  --prefix-hit <f>      prefix cache hit rate 0..1
  --prefix-tokens <n>   shared prefix length
  --eager               disable CUDA graphs
  --assume <k=v>        override an assumption (repeatable)
  --json                machine-readable output

${bold('EXAMPLE')}
  llmsize plan --model meta-llama/Llama-3.1-70B-Instruct --gpu h100-sxm-80 --tp 4 \\
               --engine vllm --context 32768 --concurrency 64 --kv-dtype fp8
`

/** Render a labelled byte breakdown as an aligned table with a proportional bar. */
function breakdownTable(rows: [string, number][], capacity: number): string {
  const width = 28
  const label = Math.max(...rows.map((r) => r[0].length))
  return rows.map(([name, value]) => {
    const frac = capacity > 0 ? Math.max(0, value / capacity) : 0
    const filled = Math.min(width, Math.round(frac * width))
    const bar = '█'.repeat(filled) + dim('·'.repeat(Math.max(0, width - filled)))
    return `  ${name.padEnd(label)}  ${gib(value).padStart(10)}  ${bar} ${(frac * 100).toFixed(1)}%`
  }).join('\n')
}

/** Entry point. Returns a process exit code: 0 fits, 1 does not fit, 2 usage/lookup error. */
export async function main(argv: string[]): Promise<number> {
  const cmd = argv[0]
  if (!cmd || cmd === '--help' || cmd === '-h' || cmd === 'help') { console.log(USAGE); return 0 }

  try {
    if (cmd === 'gpus') {
      for (const g of listGpus()) {
        console.log(`${g.id.padEnd(16)} ${gib(g.vramBytes).padStart(10)}  ${(g.memBandwidthBytesPerSec / 1e12).toFixed(2)} TB/s  ${String(g.tflopsDense.bf16).padStart(7)} bf16 TFLOPS (dense)  ${dim(g.name)}`)
      }
      return 0
    }
    if (cmd === 'models') { for (const m of listModels()) console.log(m); return 0 }
    if (cmd === 'assumptions') {
      for (const [k, a] of Object.entries(defaultAssumptions())) {
        console.log(`${bold(k)} = ${a.value} ${dim(a.unit)}  [${a.confidence} confidence]\n  ${a.rationale}\n  ${dim(a.source_url)}\n`)
      }
      return 0
    }
    if (cmd === 'validation') {
      const all = allValidations()
      if (!all.length) {
        console.log(yellow('No validation cases yet — every number this tool prints is PREDICTED, not measured.'))
        console.log(dim('Submit a real engine startup log: see validation/cases/README.md'))
        return 0
      }
      for (const v of all) console.log(`${v.model} / ${v.gpu} / ${v.engine} @ ${v.engineVersion}: ${Object.entries(v.errors).map(([k, e]) => `${k} ${(e * 100).toFixed(1)}%`).join(', ')}`)
      return 0
    }
    if (cmd !== 'plan') { console.error(`Unknown command "${cmd}".\n`); console.log(USAGE); return 2 }

    const { values } = parseArgs({
      args: argv.slice(1),
      options: {
        model: { type: 'string' }, gpu: { type: 'string' }, engine: { type: 'string', default: 'vllm' },
        tp: { type: 'string', default: '1' }, pp: { type: 'string', default: '1' },
        context: { type: 'string' }, concurrency: { type: 'string', default: '1' },
        'avg-seq-len': { type: 'string' }, quant: { type: 'string', default: 'bf16' },
        'kv-dtype': { type: 'string', default: 'fp16' }, chunk: { type: 'string', default: '8192' },
        util: { type: 'string', default: '0.90' }, 'block-size': { type: 'string', default: '16' },
        'prefix-hit': { type: 'string' }, 'prefix-tokens': { type: 'string' },
        eager: { type: 'boolean', default: false }, json: { type: 'boolean', default: false },
        assume: { type: 'string', multiple: true, default: [] },
      },
      allowPositionals: false,
    })
    if (!values.model || !values.gpu || !values.context) {
      console.error(red('--model, --gpu and --context are required.\n')); console.log(USAGE); return 2
    }

    const assume: Record<string, number> = {}
    for (const kv of values.assume as string[]) {
      const [k, v] = kv.split('=')
      if (!k || v === undefined || Number.isNaN(Number(v))) { console.error(red(`--assume expects key=number, got "${kv}"`)); return 2 }
      assume[k] = Number(v)
    }

    const model = values.model.endsWith('.json')
      ? JSON.parse(await (await import('node:fs/promises')).readFile(values.model, 'utf8'))
      : values.model
    const context = Number(values.context)
    const r = size({
      model, gpu: values.gpu, engine: values.engine as any,
      tp: Number(values.tp), pp: Number(values.pp),
      quant: values.quant as any, kvDtype: values['kv-dtype'] as any,
      context, concurrency: Number(values.concurrency),
      avgSeqLen: values['avg-seq-len'] ? Number(values['avg-seq-len']) : context,
      chunkTokens: Number(values.chunk), memoryUtilization: Number(values.util),
      blockSize: Number(values['block-size']), cudaGraphs: !values.eager,
      prefixCache: values['prefix-hit']
        ? { enabled: true, hitRate: Number(values['prefix-hit']), sharedPrefixTokens: Number(values['prefix-tokens'] ?? 0) }
        : undefined,
      assume,
    })

    if (values.json) { console.log(JSON.stringify(r, null, 2)); return r.plan.fits ? 0 : 1 }

    const p = r.plan
    const spec = p.input.model
    const gpu = p.input.gpu
    const devices = p.input.parallel.tp * p.input.parallel.pp

    console.log(`\n${bold(spec.id)} on ${bold(`${devices}x ${gpu.name}`)} with ${bold(p.engine)}`)
    console.log(dim(`${count(p.weights.params.total)} params (${p.weights.method}), ${spec.numLayers} layers, ${new Set(spec.layers.map((l) => l.kind)).size > 1 ? [...new Set(spec.layers.map((l) => l.kind))].join('+') : spec.layers[0]!.kind}, quant ${p.input.quant}, kv ${p.input.kvDtype}`))

    console.log(`\n${bold('PER-GPU MEMORY')} ${dim(`of ${gib(p.usableVramBytes)} usable`)}`)
    console.log(breakdownTable([
      ['weights', p.weightBytesPerDevice],
      ['kv cache', p.requiredKvBytes],
      ['activations', p.overhead.activationBytes + p.overhead.logitsBytes],
      ['overhead', p.overhead.nonTorchBytes + p.overhead.graphBytes],
      ['free', Math.max(0, p.freeBytes)],
    ], p.usableVramBytes))

    console.log(`\n${bold('FIT')}`)
    if (p.fits) {
      console.log(`  ${green('✓ fits')} — KV pool holds ${count(p.maxTokens)} tokens (${p.numBlocks || '—'} blocks), ${gib(p.availableKvBytes)} reserved`)
    } else {
      console.log(`  ${red('✗ does not fit')}`)
      if (p.autofix) console.log(`  ${yellow('largest that fits:')} --context ${p.autofix.maxModelLen} --concurrency ${p.autofix.maxNumSeqs}`)
    }

    console.log(`\n${bold('THROUGHPUT')} ${dim('(roofline, ' + r.label + ')')}`)
    console.log(`  decode      ${r.throughput.decode.tokensPerSecond.toFixed(0).padStart(8)} tok/s aggregate   ITL ${seconds(r.throughput.decode.stepSeconds)}   ${dim(fmtBytes(r.throughput.decode.bytesPerStep) + '/step')}`)
    console.log(`  prefill     ${r.throughput.prefill.tokensPerSecond.toFixed(0).padStart(8)} tok/s             TTFT ${seconds(r.throughput.prefill.ttftSeconds)} for a ${count(p.input.workload.chunkTokens)}-token chunk`)
    console.log(`  bound by    ${r.throughput.bound}`)

    if (spec.assumed.length) {
      console.log(`\n${bold('ASSUMED CONFIG FIELDS')} ${dim('(absent from config.json)')}`)
      for (const a of spec.assumed) console.log(`  ${yellow('·')} ${a}`)
    }
    if (p.warnings.length) {
      console.log(`\n${bold('WARNINGS')}`)
      for (const w of p.warnings) console.log(`  ${yellow('!')} ${dim(`[${w.code}]`)} ${w.message}`)
    }

    console.log(`\n${bold('RUN IT')}`)
    console.log('  ' + r.command.split('\n').join('\n  '))
    console.log(`\n${dim(r.validation
      ? `validated against ${r.validation.engine} ${r.validation.engineVersion} (case ${r.validation.caseFile})`
      : 'PREDICTED — no validation case covers this (model, gpu, engine) triple. See docs/VALIDATION.md.')}\n`)
    return p.fits ? 0 : 1
  } catch (e) {
    if (e instanceof IncompleteConfigError || e instanceof UnknownEntityError) { console.error(red(e.message)); return 2 }
    throw e
  }
}
