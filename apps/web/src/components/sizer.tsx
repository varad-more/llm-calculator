'use client'

import { useMemo, useState } from 'react'
import {
  size, listGpus, listModels, normalizeConfig, gib, seconds, count,
  IncompleteConfigError, type KvDtype, type QuantScheme,
} from '@llmsize/core'
import type { EngineName } from '@llmsize/core'
import { resolveConfig, parsePasted, type ConfigSource } from '@/lib/hf'
import { MemoryBar, SEGMENT_COLORS } from '@/components/memory-bar'
import { AssumptionsPanel } from '@/components/assumptions-panel'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'

const ENGINES: EngineName[] = ['vllm', 'sglang', 'trtllm', 'llamacpp']
const QUANTS: QuantScheme[] = ['bf16', 'fp16', 'fp8', 'int8', 'awq-int4', 'gptq-int4', 'mxfp4', 'gguf:Q4_K_M', 'gguf:Q5_K_M', 'gguf:Q8_0']
const KV_DTYPES: KvDtype[] = ['fp16', 'bf16', 'fp8', 'int8']
const CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label className="text-xs">{label}{hint ? <span className="ml-1 font-normal text-muted-foreground">{hint}</span> : null}</Label>
      {children}
    </div>
  )
}

export function Sizer({ initialModel }: { initialModel?: string }) {
  const gpus = listGpus()
  const models = listModels()

  const [modelId, setModelId] = useState(initialModel ?? 'meta-llama/Llama-3.1-8B-Instruct')
  const [customId, setCustomId] = useState('')
  const [pasted, setPasted] = useState('')
  const [token, setToken] = useState('')
  const [source, setSource] = useState<ConfigSource | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const [gpu, setGpu] = useState('h100-sxm-80')
  const [engine, setEngine] = useState<EngineName>('vllm')
  const [tp, setTp] = useState(1)
  const [pp, setPp] = useState(1)
  const [quant, setQuant] = useState<QuantScheme>('bf16')
  const [kvDtype, setKvDtype] = useState<KvDtype>('fp16')
  const [context, setContext] = useState(32768)
  const [concurrency, setConcurrency] = useState(32)
  const [avgSeqLen, setAvgSeqLen] = useState(4096)
  const [util, setUtil] = useState(0.9)
  const [prefixHit, setPrefixHit] = useState(0)
  const [prefixTokens, setPrefixTokens] = useState(2048)
  const [assume, setAssume] = useState<Record<string, number>>({})

  async function loadCustom() {
    setBusy(true); setFetchError(null)
    try {
      setSource(await resolveConfig(customId.trim(), token.trim() || undefined))
      setModelId(customId.trim())
    } catch (e) {
      setFetchError((e as Error).message)
    } finally { setBusy(false) }
  }

  function loadPasted() {
    setFetchError(null)
    try {
      const s = parsePasted(customId.trim(), pasted)
      normalizeConfig(s.config)
      setSource(s); setModelId(s.id)
    } catch (e) { setFetchError((e as Error).message) }
  }

  const result = useMemo(() => {
    try {
      const model = source && source.id === modelId ? source.config : modelId
      return {
        ok: true as const,
        value: size({
          model: model as string, gpu, engine, tp, pp, quant, kvDtype,
          context, concurrency, avgSeqLen: Math.min(avgSeqLen, context),
          memoryUtilization: util,
          prefixCache: prefixHit > 0 ? { enabled: true, hitRate: prefixHit, sharedPrefixTokens: prefixTokens } : undefined,
          assume,
        }),
      }
    } catch (e) {
      return { ok: false as const, error: e as Error }
    }
  }, [source, modelId, gpu, engine, tp, pp, quant, kvDtype, context, concurrency, avgSeqLen, util, prefixHit, prefixTokens, assume])

  if (!result.ok) {
    const e = result.error
    return (
      <Card className="border-destructive">
        <CardHeader><CardTitle className="text-destructive">Cannot size this model</CardTitle></CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p className="font-mono text-xs">{e.message}</p>
          {e instanceof IncompleteConfigError ? (
            <p className="text-muted-foreground">
              We do not guess missing config fields. Add <code>{e.field}</code> to the pasted config, or open an
              issue so a sourced default can be added for <code>{e.modelType}</code>.
            </p>
          ) : null}
        </CardContent>
      </Card>
    )
  }

  const r = result.value
  const p = r.plan
  const spec = p.input.model
  const devices = tp * pp
  const segments = [
    { key: 'weights', label: 'weights', bytes: p.weightBytesPerDevice, color: SEGMENT_COLORS.weights },
    { key: 'kv', label: 'kv cache', bytes: p.requiredKvBytes, color: SEGMENT_COLORS.kv },
    { key: 'activations', label: 'activations', bytes: p.overhead.activationBytes + p.overhead.logitsBytes, color: SEGMENT_COLORS.activations },
    { key: 'overhead', label: 'overhead', bytes: p.overhead.nonTorchBytes + p.overhead.graphBytes, color: SEGMENT_COLORS.overhead },
    { key: 'free', label: 'free', bytes: Math.max(0, p.freeBytes), color: SEGMENT_COLORS.free },
  ]

  return (
    <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-3"><CardTitle className="text-sm">Model</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Select value={models.includes(modelId) ? modelId : ''} onValueChange={(v) => { setModelId(v); setSource(null) }}>
              <SelectTrigger className="w-full"><SelectValue placeholder={modelId} /></SelectTrigger>
              <SelectContent>{models.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}</SelectContent>
            </Select>
            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground">any other Hugging Face model</summary>
              <div className="mt-2 space-y-2">
                <Input placeholder="org/model-name" value={customId} onChange={(e) => setCustomId(e.target.value)} />
                <Input placeholder="hf_… read token (only for gated repos)" value={token} onChange={(e) => setToken(e.target.value)} type="password" />
                <Button size="sm" onClick={loadCustom} disabled={busy || !customId.trim()}>{busy ? 'fetching…' : 'fetch config.json'}</Button>
                <textarea
                  className="h-24 w-full rounded-md border bg-transparent p-2 font-mono text-[11px]"
                  placeholder="…or paste a raw config.json here"
                  value={pasted} onChange={(e) => setPasted(e.target.value)}
                />
                <Button size="sm" variant="secondary" onClick={loadPasted} disabled={!pasted.trim()}>use pasted config</Button>
                {fetchError ? <p className="text-destructive">{fetchError}</p> : null}
                {source ? <p className="text-muted-foreground">loaded from {source.kind}</p> : null}
              </div>
            </details>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3"><CardTitle className="text-sm">Hardware &amp; engine</CardTitle></CardHeader>
          <CardContent className="grid grid-cols-2 gap-3">
            <div className="col-span-2">
              <Field label="GPU">
                <Select value={gpu} onValueChange={setGpu}>
                  <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>{gpus.map((g) => <SelectItem key={g.id} value={g.id}>{g.name}</SelectItem>)}</SelectContent>
                </Select>
              </Field>
            </div>
            <Field label="Engine">
              <Select value={engine} onValueChange={(v) => setEngine(v as EngineName)}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{ENGINES.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}</SelectContent>
              </Select>
            </Field>
            <Field label="Utilization" hint={util.toFixed(2)}>
              <Slider min={0.5} max={0.99} step={0.01} value={[util]} onValueChange={([v]) => setUtil(v!)} />
            </Field>
            <Field label="Tensor parallel"><Input type="number" min={1} max={64} value={tp} onChange={(e) => setTp(Math.max(1, Number(e.target.value)))} /></Field>
            <Field label="Pipeline parallel"><Input type="number" min={1} max={16} value={pp} onChange={(e) => setPp(Math.max(1, Number(e.target.value)))} /></Field>
            <Field label="Weights">
              <Select value={quant} onValueChange={(v) => setQuant(v as QuantScheme)}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{QUANTS.map((q) => <SelectItem key={q} value={q}>{q}</SelectItem>)}</SelectContent>
              </Select>
            </Field>
            <Field label="KV dtype">
              <Select value={kvDtype} onValueChange={(v) => setKvDtype(v as KvDtype)}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{KV_DTYPES.map((d) => <SelectItem key={d} value={d}>{d}</SelectItem>)}</SelectContent>
              </Select>
            </Field>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3"><CardTitle className="text-sm">Workload</CardTitle></CardHeader>
          <CardContent className="grid grid-cols-2 gap-3">
            <Field label="Context">
              <Select value={String(context)} onValueChange={(v) => setContext(Number(v))}>
                <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{CONTEXTS.map((c) => <SelectItem key={c} value={String(c)}>{c.toLocaleString()}</SelectItem>)}</SelectContent>
              </Select>
            </Field>
            <Field label="Concurrency"><Input type="number" min={1} value={concurrency} onChange={(e) => setConcurrency(Math.max(1, Number(e.target.value)))} /></Field>
            <div className="col-span-2">
              <Field label="Mean sequence length" hint={`${Math.min(avgSeqLen, context).toLocaleString()} tokens`}>
                <Slider min={128} max={context} step={128} value={[Math.min(avgSeqLen, context)]} onValueChange={([v]) => setAvgSeqLen(v!)} />
              </Field>
            </div>
            <div className="col-span-2">
              <Field label="Prefix cache hit rate" hint={prefixHit === 0 ? 'off' : `${(prefixHit * 100).toFixed(0)}%`}>
                <Slider min={0} max={1} step={0.05} value={[prefixHit]} onValueChange={([v]) => setPrefixHit(v!)} />
              </Field>
            </div>
            {prefixHit > 0 ? (
              <div className="col-span-2">
                <Field label="Shared prefix length"><Input type="number" min={0} value={prefixTokens} onChange={(e) => setPrefixTokens(Math.max(0, Number(e.target.value)))} /></Field>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Assumptions</CardTitle>
            <p className="text-xs text-muted-foreground">
              Every empirical constant behind these numbers. Editable, sourced, and labelled by how much we trust it.
            </p>
          </CardHeader>
          <CardContent><AssumptionsPanel values={assume} onChange={(k, v) => setAssume((a) => ({ ...a, [k]: v }))} /></CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-base">{spec.id}</CardTitle>
              <Badge variant={p.fits ? 'default' : 'destructive'}>{p.fits ? 'fits' : 'does not fit'}</Badge>
              <Badge variant="outline">{r.label}</Badge>
              <span className="ml-auto text-xs text-muted-foreground">
                {devices}x {p.input.gpu.name} · {count(p.weights.params.total)} params ({p.weights.method})
              </span>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <MemoryBar segments={segments} capacityBytes={p.usableVramBytes} budgetBytes={p.budgetBytes} />

            {!p.fits && p.autofix ? (
              <div className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
                <p className="font-medium">Largest configuration that does fit</p>
                <p className="mt-1 text-muted-foreground">
                  context <span className="font-mono">{p.autofix.maxModelLen.toLocaleString()}</span>,
                  concurrency <span className="font-mono">{p.autofix.maxNumSeqs.toLocaleString()}</span> at a mean of{' '}
                  {Math.min(avgSeqLen, context).toLocaleString()} tokens
                </p>
                <Button
                  size="sm" variant="secondary" className="mt-2"
                  onClick={() => { setContext(p.autofix!.maxModelLen); setConcurrency(Math.max(1, p.autofix!.maxNumSeqs)) }}
                >
                  apply
                </Button>
              </div>
            ) : null}

            <div className="grid gap-4 sm:grid-cols-3">
              <Stat label="KV pool" value={`${count(p.maxTokens)} tokens`} sub={gib(p.availableKvBytes)} />
              <Stat label="Decode" value={`${r.throughput.decode.tokensPerSecond.toFixed(0)} tok/s`} sub={`ITL ${seconds(r.throughput.decode.stepSeconds)}`} />
              <Stat label="Prefill" value={`${r.throughput.prefill.tokensPerSecond.toFixed(0)} tok/s`} sub={`TTFT ${seconds(r.throughput.prefill.ttftSeconds)} · ${r.throughput.bound}-bound`} />
            </div>
          </CardContent>
        </Card>

        {spec.assumed.length || p.warnings.length ? (
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">What to know</CardTitle></CardHeader>
            <CardContent className="space-y-2 text-sm">
              {spec.assumed.map((a) => (
                <p key={a} className="text-muted-foreground">
                  <span className="mr-2 rounded bg-amber-500/10 px-1 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">assumed</span>
                  {a}
                </p>
              ))}
              {p.warnings.map((w) => (
                <p key={w.code + w.message} className="text-muted-foreground">
                  <span className="mr-2 font-mono text-[10px] text-muted-foreground">[{w.code}]</span>
                  {w.message}
                </p>
              ))}
            </CardContent>
          </Card>
        ) : null}

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Run it</CardTitle>
            <p className="text-xs text-muted-foreground">
              {p.fits ? 'These flags match the numbers above.' : 'Flags are emitted for the largest configuration that fits.'}
            </p>
          </CardHeader>
          <CardContent>
            <pre className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed">{r.command}</pre>
            <Button size="sm" variant="secondary" className="mt-2" onClick={() => navigator.clipboard?.writeText(r.command)}>copy</Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2"><CardTitle className="text-sm">Where the memory goes</CardTitle></CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <tbody className="font-mono tabular-nums">
                <Row label="model weights (per GPU)" bytes={p.weightBytesPerDevice} />
                <Row label="kv cache required" bytes={p.requiredKvBytes} />
                <Row label="kv pool reserved by engine" bytes={p.availableKvBytes} />
                <Separator className="my-1" />
                <Row label="cuda context" bytes={p.overhead.cudaContextBytes} />
                <Row label="nccl buffers" bytes={p.overhead.commBytes} />
                <Row label="cuda graphs" bytes={p.overhead.graphBytes} />
                <Row label="activation peak" bytes={p.overhead.activationBytes} />
                <Row label="sampling logits" bytes={p.overhead.logitsBytes} />
                <Row label="allocator slack" bytes={p.overhead.fragmentationBytes} />
                <Separator className="my-1" />
                <Row label="usable vram per GPU" bytes={p.usableVramBytes} />
                <Row label="free" bytes={p.freeBytes} />
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-md border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-0.5 font-mono text-lg tabular-nums">{value}</div>
      <div className="text-xs text-muted-foreground">{sub}</div>
    </div>
  )
}

function Row({ label, bytes }: { label: string; bytes: number }) {
  return (
    <tr>
      <td className="py-0.5 font-sans text-muted-foreground">{label}</td>
      <td className="py-0.5 text-right">{gib(bytes)}</td>
    </tr>
  )
}
