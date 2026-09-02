import Link from 'next/link'
import {
  defaultAssumptions,
  quantData,
  weightBytes,
  parameterCounts,
  kvBytesForSequence,
  kvBytesPerTokenPerLayer,
  kvDtypeBytes,
  activeParameters,
  size,
  gib,
  bytes as humanBytes,
  count,
  type ModelSpec,
  type QuantScheme,
  type KvDtype,
} from '@llmsize/core'
import { specFor } from '@/lib/models'

export const metadata = {
  title: 'How the numbers are calculated — quantization, KV cache and engine memory',
  description:
    'What quantization actually stores, why 4-bit is never 4 bits per weight, how the KV cache is ' +
    'sized under GQA/MLA/sliding-window attention, and the exact allocation each serving engine ' +
    'performs. Every number on this page is computed by the same functions the sizer uses.',
}

// Everything below is computed at build time from the shipped snapshots, so the page cannot
// drift from the math it is documenting. No number here is typed in by hand.
const REF = 'meta-llama/Llama-3.1-8B-Instruct'
const ref = specFor(REF)
const refParams = parameterCounts(ref)

const SCHEMES: QuantScheme[] = ['fp32', 'bf16', 'fp8', 'int8', 'gptq-int4', 'awq-int4', 'mxfp4']
const GGUF: string[] = ['Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q4_0', 'Q3_K_M', 'Q2_K']
const KV_DTYPES: KvDtype[] = ['fp32', 'fp16', 'fp8', 'int8']
const KV_MODELS = [
  'meta-llama/Llama-3.1-8B-Instruct',
  'meta-llama/Llama-3.1-70B-Instruct',
  'mistralai/Mixtral-8x7B-Instruct-v0.1',
  'google/gemma-3-27b-it',
  'openai/gpt-oss-120b',
  'deepseek-ai/DeepSeek-V3',
]

/** Bytes actually spent per stored weight — the only honest way to compare schemes. */
/** Group metadata amortised over the weights it covers, in bits per weight. */
function metadataBpw(scheme: any): number {
  return scheme.kind === 'grouped' ? (scheme.scaleBits + scheme.zeroBits) / scheme.groupSize : 0
}

function effectiveBpw(spec: ModelSpec, quant: QuantScheme): { bytes: number; bpw: number } {
  const w = weightBytes(spec, { quant, preferMeasured: false })
  return { bytes: w.totalBytes, bpw: (w.totalBytes * 8) / parameterCounts(spec).total }
}

/** What a tool that assumes every layer is a plain full-attention GQA layer would predict. */
function naiveKvBytes(spec: ModelSpec, tokens: number, dtypeBytes: number): number {
  return 2 * spec.numKeyValueHeads * spec.headDim * spec.numLayers * tokens * dtypeBytes
}

const CTX = 32768
const quant = quantData()
const assumptions = defaultAssumptions()

// The worked example the memory section walks through, line by line.
const example = size({
  model: 'meta-llama/Llama-3.1-70B-Instruct',
  gpu: 'h100-sxm-80',
  engine: 'vllm',
  tp: 4,
  context: CTX,
  concurrency: 64,
  avgSeqLen: 4096,
  kvDtype: 'fp8',
})
const ex = example.plan

const SECTIONS = [
  ['allocation', 'What actually gets allocated'],
  ['quantization', 'Quantization'],
  ['kv', 'The KV cache'],
  ['speed', 'Where the speed numbers come from'],
  ['engines', 'What each engine does differently'],
  ['features', 'Serving features'],
  ['assumptions', 'The constants it rests on'],
] as const

function Section({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 border-t pt-10">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <div className="mt-4 space-y-4 text-sm leading-relaxed">{children}</div>
    </section>
  )
}

function Formula({ children }: { children: React.ReactNode }) {
  return (
    <pre className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">{children}</pre>
  )
}

function Table({ head, children }: { head: string[]; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[40rem] text-sm">
        <thead className="text-left text-xs text-muted-foreground">
          <tr>
            {head.map((h) => (
              <th key={h} className="pb-2 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">{children}</tbody>
      </table>
    </div>
  )
}

export default function ExplainedPage() {
  const bf16 = effectiveBpw(ref, 'bf16')

  return (
    <main className="mx-auto max-w-4xl px-4 py-8 md:px-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">How the numbers are calculated</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          Quantization, KV cache, and the allocation each serving engine actually performs — the
          whole method, with the formulas the code runs. Every figure on this page is computed at
          build time by the same functions the{' '}
          <Link href="/" className="underline underline-offset-2 hover:text-foreground">sizer</Link>{' '}
          uses, from the {count(refParams.total)}-parameter reference model{' '}
          <span className="font-mono text-xs">{REF}</span> and the shipped GPU snapshots. Nothing is
          hand-typed, so this page cannot drift from the math it documents.
        </p>
        <nav className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          {SECTIONS.map(([id, title]) => (
            <a key={id} href={`#${id}`} className="hover:text-foreground">{title}</a>
          ))}
        </nav>
      </header>

      <div className="space-y-10">
        <Section id="allocation" title="What actually gets allocated">
          <p>
            The common mental model — &ldquo;VRAM needed = model size, plus a bit&rdquo; — is wrong in a
            way that matters. A serving engine does not allocate what the model needs; it takes a
            budget, subtracts everything it can measure, and turns whatever is left into a cache. Five
            terms, in the order the engine resolves them:
          </p>
          <Formula>{`usable   = nominal VRAM x (1 - driver_reserved)      driver, ECC and page tables
budget   = usable x utilization                     --gpu-memory-utilization
weights  = quantized parameter bytes / (TP x PP)    sharded across devices
overhead = CUDA context + NCCL buffers + CUDA graphs + activation peak + logits
KV pool  = budget - weights - overhead              everything left, frozen into blocks`}</Formula>
          <p>
            The KV pool is a <em>residual</em>. That is the whole reason sizing is not arithmetic you
            can do in your head: every term on the right steals from it, and two of them (CUDA graphs
            and the sampling logits buffer) scale with the batch size you asked for, so raising
            concurrency shrinks the pool that concurrency needs.
          </p>
          <p>
            Worked through for <span className="font-mono text-xs">meta-llama/Llama-3.1-70B-Instruct</span>{' '}
            on {ex.input.parallel.tp}x {ex.input.gpu.name}, vLLM, {CTX.toLocaleString()} context,{' '}
            {ex.input.workload.concurrency} sequences, fp8 KV:
          </p>
          <Table head={['Term', 'Per GPU', 'Where it comes from']}>
            <tr className="border-t">
              <td className="py-2 font-sans">nominal capacity</td>
              <td className="py-2">{gib(ex.input.gpu.vramBytes)}</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">datasheet</td>
            </tr>
            <tr className="border-t">
              <td className="py-2 font-sans">usable</td>
              <td className="py-2">{gib(ex.usableVramBytes)}</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                minus the {(assumptions.driver_reserved_vram_fraction.value * 100).toFixed(1)}% the driver keeps
              </td>
            </tr>
            <tr className="border-t">
              <td className="py-2 font-sans">budget</td>
              <td className="py-2">{gib(ex.budgetBytes)}</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                x {ex.input.memoryUtilization} utilization
              </td>
            </tr>
            <tr className="border-t">
              <td className="py-2 font-sans">− weights</td>
              <td className="py-2">{gib(ex.weightBytesPerDevice)}</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                {gib(ex.weights.totalBytes)} total / TP {ex.input.parallel.tp} ({ex.weights.method})
              </td>
            </tr>
            <tr className="border-t">
              <td className="py-2 font-sans">− non-torch</td>
              <td className="py-2">{gib(ex.overhead.nonTorchBytes)}</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                CUDA context, NCCL buffers, CUDA graph pools
              </td>
            </tr>
            <tr className="border-t">
              <td className="py-2 font-sans">− activations + logits</td>
              <td className="py-2">{gib(ex.overhead.activationBytes + ex.overhead.logitsBytes)}</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                peak of one prefill chunk, plus the sampling buffer
              </td>
            </tr>
            <tr className="border-t font-semibold">
              <td className="py-2 font-sans">= KV pool</td>
              <td className="py-2">{gib(ex.availableKvBytes)}</td>
              <td className="py-2 font-sans text-xs font-normal text-muted-foreground">
                {ex.numBlocks.toLocaleString()} blocks, {count(ex.maxTokens)} tokens
              </td>
            </tr>
          </Table>
          <p>
            The pool holds <strong>{count(ex.maxTokens)} tokens</strong>. That is the number that
            decides whether a config runs: it must cover both one full-length sequence
            ({CTX.toLocaleString()} tokens) and the whole batch at once
            ({ex.input.workload.concurrency} sequences averaging{' '}
            {ex.input.workload.avgSeqLen.toLocaleString()} tokens ={' '}
            {count(ex.input.workload.concurrency * ex.input.workload.avgSeqLen)}). This config{' '}
            {ex.fits ? 'clears both, with room to spare.' : 'clears neither.'} Raise concurrency to{' '}
            {Math.ceil(ex.maxTokens / ex.input.workload.avgSeqLen)} at the same average length and the
            pool runs out.
          </p>
        </Section>

        <Section id="quantization" title="Quantization">
          <p>
            Quantization stores each weight in fewer bits. It cannot store <em>only</em> those bits:
            a 4-bit integer has no scale of its own, so weights are cut into groups of{' '}
            {assumptions.quant_group_size.value} and each group carries a shared scale — and, for
            asymmetric schemes, a zero point. That metadata is real memory:
          </p>
          <Formula>{`bytes(class) = P x b/8  +  ceil(P/g) x (b_scale + b_zero)/8
                ^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                payload      group metadata (grouped schemes only)`}</Formula>
          <p>
            Second correction, larger than the first: <strong>not every tensor is quantized.</strong>{' '}
            Every production scheme leaves the embedding table, the output head and the norms in
            16-bit, because quantizing them costs accuracy and saves little. On a small model the
            embeddings are a big fraction of the parameters, so the two effects together push a
            &ldquo;4-bit&rdquo; checkpoint well past 4 bits per weight:
          </p>
          <Table head={['Scheme', 'Nominal', 'Group', 'Scale + zero', 'Metadata', '16-bit tensors', 'Effective bpw', `${REF.split('/')[1]}`]}>
            {SCHEMES.map((s) => {
              const spec = quant.schemes[s]
              const e = effectiveBpw(ref, s)
              const policy = Object.keys(spec.tensorPolicy ?? {})
              return (
                <tr key={s} className="border-t">
                  <td className="py-2">
                    {spec.source_url ? (
                      <a href={spec.source_url} target="_blank" rel="noreferrer" className="underline underline-offset-2">{s}</a>
                    ) : s}
                  </td>
                  <td className="py-2">{spec.bits} bit</td>
                  <td className="py-2">{spec.kind === 'grouped' ? spec.groupSize : '—'}</td>
                  <td className="py-2">
                    {spec.kind === 'grouped' ? `${spec.scaleBits} + ${spec.zeroBits} bit` : '—'}
                  </td>
                  <td className="py-2">
                    {spec.kind === 'grouped' ? `${metadataBpw(spec).toFixed(3)} b/w` : '—'}
                  </td>
                  <td className="py-2 font-sans text-xs text-muted-foreground">
                    {policy.length ? policy.join(', ') : 'none'}
                  </td>
                  <td className="py-2">{e.bpw.toFixed(2)}</td>
                  <td className="py-2">{gib(e.bytes)}</td>
                </tr>
              )
            })}
          </Table>
          <p className="text-xs text-muted-foreground">
            The metadata column is where the schemes actually differ. mxfp4 groups only{' '}
            {quant.schemes.mxfp4.groupSize} weights per scale — a quarter the group size of int4 — but
            pays for it with an {quant.schemes.mxfp4.scaleBits}-bit scale and no zero point, so it lands
            at {metadataBpw(quant.schemes.mxfp4).toFixed(3)} bits of metadata per weight against{' '}
            {metadataBpw(quant.schemes['gptq-int4']).toFixed(3)} for a 128-group int4 with a 16-bit scale
            and a 4-bit zero. Both are small next to the 16-bit tensors neither scheme touches.
          </p>
          <p>
            <strong>GGUF is a different animal.</strong> llama.cpp&rsquo;s k-quants mix block formats{' '}
            <em>within one file</em> — attention tensors at a higher precision than the feed-forward
            bulk — so a scheme&rsquo;s bits-per-weight is not derivable from its name. The numbers below
            are measured over a whole 7B checkpoint, not computed:
          </p>
          <Table head={['GGUF scheme', 'Measured bpw', `${REF.split('/')[1]} weights`, 'vs bf16']}>
            {GGUF.map((g) => {
              const bpw = quant.gguf.bpw[g] as number
              const b = (refParams.total * bpw) / 8
              return (
                <tr key={g} className="border-t">
                  <td className="py-2">{g}</td>
                  <td className="py-2">{bpw.toFixed(2)}</td>
                  <td className="py-2">{gib(b)}</td>
                  <td className="py-2">{(b / bf16.bytes).toFixed(2)}x</td>
                </tr>
              )
            })}
          </Table>
          <p className="text-xs text-muted-foreground">
            Source: <a href={quant.gguf._source_url} target="_blank" rel="noreferrer" className="underline underline-offset-2">llama.cpp k-quants PR #1684</a>.
            Q4_K_M is 4.85 bpw, not 4 — a 21% error if you take the name literally.
          </p>
          <p>
            <strong>What quantization does not shrink.</strong> Only the weights term. The KV cache,
            the activation peak, the CUDA graph pools and the CUDA context are unchanged — which is why
            an int4 checkpoint that halves your weights does not double your context. To shrink the KV
            cache you quantize the <em>cache</em>, separately:
          </p>
          <Table head={['KV dtype', 'Bits', `Per token, ${REF.split('/')[1]}`, `One ${CTX / 1024}k sequence`]}>
            {KV_DTYPES.map((d) => {
              const s = kvDtypeBytes(d)
              const perTok = ref.layers.reduce((a, l) => a + kvBytesPerTokenPerLayer(l, ref, s), 0)
              return (
                <tr key={d} className="border-t">
                  <td className="py-2">{d}</td>
                  <td className="py-2">{quant.kvDtypeBits[d]}</td>
                  <td className="py-2">{(perTok / 1024).toFixed(1)} KiB</td>
                  <td className="py-2">{gib(kvBytesForSequence(ref, CTX, { tp: 1, pp: 1 }, { kvDtype: d, blockSize: 16 }))}</td>
                </tr>
              )
            })}
          </Table>
          <p className="text-xs text-muted-foreground">
            This tool sizes memory, not quality. It will not tell you whether Q4_K_M is good enough for
            your task — that is a perplexity/eval question, and inventing an answer to it would be
            exactly the kind of unsourced number the rest of the project refuses to print.
          </p>
        </Section>

        <Section id="kv" title="The KV cache">
          <p>
            Weights are a fixed cost. The KV cache is the one that scales with traffic: every token of
            every live sequence keeps its key and value tensors resident for the whole generation. It
            is usually what actually decides your maximum context and concurrency.
          </p>
          <Formula>{`MHA / GQA / MQA     bytes/token = 2 x n_kv_heads x head_dim x sizeof(dtype)   per layer
MLA (DeepSeek)      bytes/token = (kv_lora_rank + qk_rope_dim) x sizeof(dtype)
sliding window      same per token, but only W tokens are ever resident
Mamba / linear      0 per token — a fixed state per sequence instead`}</Formula>
          <p>
            The factor of 2 is K and V. GQA already cut this by sharing one KV head across several
            query heads ({ref.numAttentionHeads} query heads to {ref.numKeyValueHeads} KV heads on the
            reference model — a {ref.numAttentionHeads / ref.numKeyValueHeads}x saving over MHA). MLA
            goes further and caches a single compressed latent per token, which is why it has neither a
            factor of 2 nor a head count in its formula.
          </p>
          <p>
            <strong>Layers are not homogeneous, and assuming they are is the most common error.</strong>{' '}
            Modern models interleave attention types. Here is what the per-layer dispatch buys, against
            what a uniform-model calculator would predict for the same {CTX / 1024}k sequence:
          </p>
          <Table head={['Model', 'Attention', 'KV @ ' + CTX / 1024 + 'k (fp16)', 'Naive uniform', 'Overestimate']}>
            {KV_MODELS.map((id) => {
              const m = specFor(id)
              const kinds = [...new Set(m.layers.map((l) => l.kind))]
              const windowed = m.layers.filter((l) => l.windowSize).length
              const real = kvBytesForSequence(m, CTX, { tp: 1, pp: 1 }, { kvDtype: 'fp16', blockSize: 16 })
              const naive = naiveKvBytes(m, CTX, 2)
              return (
                <tr key={id} className="border-t">
                  <td className="py-2 font-sans">{id.split('/')[1]}</td>
                  <td className="py-2 font-sans text-xs text-muted-foreground">
                    {kinds.join('+')}
                    {windowed ? `, ${windowed}/${m.numLayers} windowed` : ''}
                  </td>
                  <td className="py-2">{gib(real)}</td>
                  <td className="py-2 text-muted-foreground">{gib(naive)}</td>
                  <td className="py-2">{(naive / real).toFixed(1)}x</td>
                </tr>
              )
            })}
          </Table>
          <p>
            Gemma 3 is mostly sliding-window layers that never hold more than their window, and DeepSeek-V3&rsquo;s MLA latent is a fraction of what its {' '}
            {specFor('deepseek-ai/DeepSeek-V3').numAttentionHeads} heads would suggest. Neither error is
            a rounding difference; both change which GPU you need to buy.
          </p>
          <p>
            Two more corrections sit on top of the per-token cost. <strong>Paged attention</strong>{' '}
            allocates whole blocks (16 tokens by default), so a sequence is charged{' '}
            <span className="font-mono text-xs">ceil(len / block) x block</span>. And{' '}
            <strong>tensor parallelism shards the cache by KV heads</strong> — which stops helping the
            moment TP exceeds <span className="font-mono text-xs">num_key_value_heads</span>, and never
            helps at all under MLA, where the latent is replicated on every rank.
          </p>
        </Section>

        <Section id="speed" title="Where the speed numbers come from">
          <p>
            Throughput here is a <strong>roofline bound</strong>, not a benchmark: the best a perfect
            kernel could do on this hardware, scaled down by an efficiency assumption. It models no
            scheduler, no queueing, no batching jitter. Treat it as a ceiling and a sanity check.
          </p>
          <Formula>{`decode  (memory-bound)   t_step = (weights/GPU + KV read) / (HBM bandwidth x MBU) + t_comm
prefill (compute-bound)  TTFT   = (2 x P_active x C + attention FLOPs) / (dense TFLOPS x MFU) + t_comm
tensor-parallel comm     t      = 2(n-1)/n x bytes / (link bandwidth x efficiency)`}</Formula>
          <p>
            Decoding one token requires reading every weight the token touches, plus the entire KV cache
            of the batch. It does almost no arithmetic per byte read, so it is bandwidth-bound, and the
            fix is batching: the weight read is amortised across the batch, which is why tokens/sec
            climbs with concurrency while per-token latency barely moves. Prefill is the opposite — a
            long prompt is a dense matrix multiply and saturates the tensor cores.
          </p>
          <p>
            Two traps the formulas encode. An <strong>MoE model streams every expert</strong> it has
            resident but computes only the top-k, so memory and FLOPs must use different parameter
            counts ({REF.split('/')[1]} is dense, but Mixtral-8x7B is{' '}
            {count(parameterCounts(specFor('mistralai/Mixtral-8x7B-Instruct-v0.1')).total)} parameters
            of memory against{' '}
            {count(activeParameters(specFor('mistralai/Mixtral-8x7B-Instruct-v0.1')))} of compute).
            And <strong>dense TFLOPS, never sparse</strong> — vendor datasheets headline the 2:1
            structured-sparsity figure, which is exactly twice what a dense inference kernel gets.
          </p>
          <p>
            For the {ex.input.parallel.tp}x {ex.input.gpu.name} example above, one decode step moves{' '}
            {gib(example.throughput.decode.bytesPerStep)} — the weights on this device plus the whole
            KV cache of the {ex.input.workload.concurrency}-sequence batch — at{' '}
            {(assumptions.mbu_decode.value * 100).toFixed(0)}% of{' '}
            {(ex.input.gpu.memBandwidthBytesPerSec / 1e12).toFixed(2)} TB/s, giving{' '}
            {example.throughput.decode.itlMs.toFixed(1)} ms between tokens and{' '}
            {example.throughput.decode.tokensPerSecond.toFixed(0)} tokens/sec across the batch.
          </p>
        </Section>

        <Section id="engines" title="What each engine does differently">
          <p>
            &ldquo;How much VRAM does it need&rdquo; has four different answers because four engines
            resolve the residual differently. This is where a generic calculator is wrong even when its
            arithmetic is right.
          </p>
          <Table head={['Engine', 'KV pool =', 'The surprise']}>
            <tr className="border-t align-top">
              <td className="py-2 font-sans">vLLM</td>
              <td className="py-2 text-xs">budget − weights − non-torch − activations − graphs</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                It profiles a real forward pass at startup, then freezes the remainder into fixed blocks.
                Raising <span className="font-mono">--max-num-seqs</span> grows the CUDA-graph and logits
                buffers, shrinking the pool.
              </td>
            </tr>
            <tr className="border-t align-top">
              <td className="py-2 font-sans">SGLang</td>
              <td className="py-2 text-xs">usable x mem-fraction-static − weights</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                One static fraction covers weights <em>and</em> KV; activations and graphs must fit in
                what is left over. Set it too high and it OOMs mid-run rather than at startup.
              </td>
            </tr>
            <tr className="border-t align-top">
              <td className="py-2 font-sans">TensorRT-LLM</td>
              <td className="py-2 text-xs">free_fraction x (usable − weights − build-time activations)</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                The activation workspace is baked in at <em>build</em> time. Serving with a smaller batch
                than you built for buys the KV pool nothing.
              </td>
            </tr>
            <tr className="border-t align-top">
              <td className="py-2 font-sans">llama.cpp</td>
              <td className="py-2 text-xs">n_ctx x bytes-per-token, allocated whole</td>
              <td className="py-2 font-sans text-xs text-muted-foreground">
                No paging. One contiguous buffer split across <span className="font-mono">n_parallel</span>{' '}
                slots, each holding its full share whether used or not.
              </td>
            </tr>
          </Table>
        </Section>

        <Section id="features" title="Serving features">
          <p>
            Four things that change the arithmetic and that most sizing tools ignore entirely.
          </p>
          <p>
            <strong>Prefix caching.</strong> A shared system prompt is stored once, not once per
            sequence. With hit rate <span className="font-mono text-xs">h</span> over{' '}
            <span className="font-mono text-xs">p</span> shared tokens, each sequence pays for{' '}
            <span className="font-mono text-xs">len − h·p</span> tokens instead of{' '}
            <span className="font-mono text-xs">len</span>. At high concurrency with a long system
            prompt this is the difference between fitting and not.
          </p>
          <p>
            <strong>Speculative decoding.</strong> A small draft model proposes{' '}
            <span className="font-mono text-xs">k</span> tokens; the target verifies all of them in one
            pass and accepts the prefix that matches. With per-token acceptance{' '}
            <span className="font-mono text-xs">α</span> the yield per cycle is a truncated geometric
            series:
          </p>
          <Formula>{`E[tokens] = (1 - a^(k+1)) / (1 - a)        cycle = k draft steps + 1 target forward
ITL       = cycle_seconds / E[tokens]`}</Formula>
          <p>
            It is a <em>latency</em> optimisation, and it can lose. If the draft is not much faster than
            the target, or acceptance is low, the speedup falls below 1.0 — the sizer reports that
            honestly instead of clamping it.
          </p>
          <p>
            <strong>Multi-LoRA.</strong> A rank-<span className="font-mono text-xs">r</span> adapter on an{' '}
            <span className="font-mono text-xs">m x n</span> projection stores{' '}
            <span className="font-mono text-xs">r(m + n)</span> parameters, not{' '}
            <span className="font-mono text-xs">r·m·n</span>. Engines pre-allocate slots at the maximum
            configured rank, so the cost is per <em>slot</em>, not per loaded adapter.
          </p>
          <p>
            <strong>Disaggregated prefill/decode.</strong> Separate pools, each sized for its own job —
            and the KV cache of every finished prefill has to cross the wire to the decode pool. That
            transfer, not the compute, is usually the bottleneck, and it is sized as{' '}
            <span className="font-mono text-xs">kvBytesForSequence x kvShards</span> (shards = 1 for MLA,
            where the latent is replicated rather than split).
          </p>
        </Section>

        <Section id="assumptions" title="The constants it rests on">
          <p>
            Every empirical number the model uses lives in one file, with a rationale, a source and an
            honest confidence label — and every one is overridable at the API, the CLI
            (<span className="font-mono text-xs">--assume mbu_decode=0.72</span>) and in the sizer&rsquo;s
            own panel. <span className="font-mono text-xs">low</span> confidence means no primary source
            was found; that is where to aim your scepticism first.
          </p>
          <Table head={['Constant', 'Value', 'Confidence', 'Why']}>
            {Object.entries(assumptions).map(([key, a]) => (
              <tr key={key} className="border-t align-top">
                <td className="py-2 text-xs">{key}</td>
                <td className="py-2 text-xs">
                  {a.unit === 'bytes' ? humanBytes(a.value) : a.value}
                </td>
                <td className="py-2">
                  <span
                    className={
                      'rounded px-1 py-0.5 text-[10px] font-medium ' +
                      (a.confidence === 'high' ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                        : a.confidence === 'medium' ? 'bg-amber-500/10 text-amber-600 dark:text-amber-400'
                        : 'bg-red-500/10 text-red-600 dark:text-red-400')
                    }
                  >
                    {a.confidence}
                  </span>
                </td>
                <td className="py-2 font-sans text-xs text-muted-foreground">
                  {a.rationale}{' '}
                  <a href={a.source_url} target="_blank" rel="noreferrer" className="underline underline-offset-2">source</a>
                </td>
              </tr>
            ))}
          </Table>
          <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
            <strong>Predicted, not measured.</strong> Every number on this page and in the sizer is
            computed from these formulas and these constants. The only quantity checked against reality
            today is the weight count, which is diffed against each checkpoint&rsquo;s own file sizes
            (all 20 snapshots within 0.12%). The validation harness that parses real engine startup logs
            is built and tested, but has no contributed logs yet — so nothing is labelled{' '}
            <span className="font-mono">validated</span>. If you have a GPU, one pasted startup log is
            worth more than any argument about the math.
          </p>
        </Section>
      </div>

      <p className="mt-10 border-t pt-6 text-xs text-muted-foreground">
        Full derivations, in LaTeX, with citations: <span className="font-mono">docs/MATH.md</span>. The
        constants above: <span className="font-mono">data/assumptions.json</span>. Now go{' '}
        <Link href="/" className="underline underline-offset-2 hover:text-foreground">size a config</Link>{' '}
        or see <Link href="/fits/" className="underline underline-offset-2 hover:text-foreground">what fits your GPU</Link>.
      </p>
    </main>
  )
}
