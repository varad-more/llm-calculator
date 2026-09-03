'use client'

import { useMemo, useState } from 'react'
import {
  size, listGpus, listModels, normalizeConfig, gib, bytes as humanBytes, seconds, count,
  qualityFor, pplPenalty,
  IncompleteConfigError, type KvDtype, type QuantScheme, type EngineName, type SizingRequest,
} from '@llmsize/core'
import { resolveConfig, parsePasted, type ConfigSource } from '@/lib/hf'
import { useUrlState } from '@/lib/url-state'
import { MemoryBar, SEGMENT_COLORS } from '@/components/memory-bar'
import { AssumptionsPanel } from '@/components/assumptions-panel'
import { Field, Stat, CommandBlock } from '@/components/field'
import { Speculative, MultiLora, Disaggregated } from '@/components/advanced'
import { ScalingCurve } from '@/components/scaling-curve'
import { HostPanel } from '@/components/host-panel'
import { Machines } from '@/components/machines'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

const ENGINES: EngineName[] = ['vllm', 'sglang', 'trtllm', 'llamacpp']
const QUANTS: QuantScheme[] = ['bf16', 'fp16', 'fp8', 'int8', 'awq-int4', 'gptq-int4', 'mxfp4', 'gguf:Q4_K_M', 'gguf:Q5_K_M', 'gguf:Q8_0']
const KV_DTYPES: KvDtype[] = ['fp16', 'bf16', 'fp8', 'int8']
const CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

export function Sizer({ initialModel }: { initialModel?: string }) {
  const gpus = listGpus()
  const models = listModels()

  const [cfg, set, setCfg] = useUrlState({
    model: initialModel ?? 'meta-llama/Llama-3.1-8B-Instruct',
    gpu: 'h100-sxm-80',
    engine: 'vllm' as string,
    tp: 1,
    pp: 1,
    quant: 'bf16' as string,
    kvDtype: 'fp16' as string,
    context: 32768,
    concurrency: 32,
    avgSeqLen: 4096,
    util: 0.9,
    prefixHit: 0,
    prefixTokens: 2048,
  })

  const [customId, setCustomId] = useState('')
  const [pasted, setPasted] = useState('')
  const [token, setToken] = useState('')
  const [source, setSource] = useState<ConfigSource | null>(null)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [assume, setAssume] = useState<Record<string, number>>({})

  async function loadCustom() {
    setBusy(true); setFetchError(null)
    try {
      const s = await resolveConfig(customId.trim(), token.trim() || undefined)
      setSource(s); set('model', customId.trim())
    } catch (e) {
      setFetchError((e as Error).message)
    } finally { setBusy(false) }
  }

  function loadPasted() {
    setFetchError(null)
    try {
      const s = parsePasted(customId.trim(), pasted)
      normalizeConfig(s.config)
      setSource(s); set('model', s.id)
    } catch (e) { setFetchError((e as Error).message) }
  }

  // The exact request the CLI would take, so every panel below derives from one object.
  const req: SizingRequest = useMemo(() => ({
    model: (source && source.id === cfg.model ? source.config : cfg.model) as string,
    gpu: cfg.gpu,
    engine: cfg.engine as EngineName,
    tp: cfg.tp,
    pp: cfg.pp,
    quant: cfg.quant as QuantScheme,
    kvDtype: cfg.kvDtype as KvDtype,
    context: cfg.context,
    concurrency: cfg.concurrency,
    avgSeqLen: Math.min(cfg.avgSeqLen, cfg.context),
    memoryUtilization: cfg.util,
    prefixCache: cfg.prefixHit > 0
      ? { enabled: true, hitRate: cfg.prefixHit, sharedPrefixTokens: cfg.prefixTokens }
      : undefined,
    assume,
  }), [source, cfg, assume])

  const result = useMemo(() => {
    try { return { ok: true as const, value: size(req) } }
    catch (e) { return { ok: false as const, error: e as Error } }
  }, [req])

  const controls = (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3"><CardTitle className="text-sm">Model</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          <Select value={models.includes(cfg.model) ? cfg.model : ''}
                  onValueChange={(v) => { setSource(null); set('model', v) }}>
            <SelectTrigger className="w-full"><SelectValue placeholder={cfg.model} /></SelectTrigger>
            <SelectContent>{models.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}</SelectContent>
          </Select>
          <details className="text-xs">
            <summary className="cursor-pointer text-muted-foreground">any other Hugging Face model</summary>
            <div className="mt-2 space-y-2">
              <Input placeholder="org/model-name" value={customId} onChange={(e) => setCustomId(e.target.value)} />
              <Input placeholder="hf_… read token (only for gated repos)" value={token}
                     onChange={(e) => setToken(e.target.value)} type="password" />
              <Button size="sm" onClick={loadCustom} disabled={busy || !customId.trim()}>
                {busy ? 'fetching…' : 'fetch config.json'}
              </Button>
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
          <Field label="GPU" className="col-span-2">
            <Select value={cfg.gpu} onValueChange={(v) => set('gpu', v)}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{gpus.map((g) => <SelectItem key={g.id} value={g.id}>{g.name}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="Engine">
            <Select value={cfg.engine} onValueChange={(v) => set('engine', v)}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{ENGINES.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="Utilization" hint={cfg.util.toFixed(2)}>
            <Slider min={0.5} max={0.99} step={0.01} value={[cfg.util]} onValueChange={([v]) => set('util', v!)} />
          </Field>
          <Field label="Tensor parallel">
            <Input type="number" min={1} max={64} value={cfg.tp} onChange={(e) => set('tp', Math.max(1, Number(e.target.value) || 1))} />
          </Field>
          <Field label="Pipeline parallel">
            <Input type="number" min={1} max={16} value={cfg.pp} onChange={(e) => set('pp', Math.max(1, Number(e.target.value) || 1))} />
          </Field>
          <Field label="Weights">
            <Select value={cfg.quant} onValueChange={(v) => set('quant', v)}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{QUANTS.map((q) => <SelectItem key={q} value={q}>{q}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="KV dtype">
            <Select value={cfg.kvDtype} onValueChange={(v) => set('kvDtype', v)}>
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
            <Select value={String(cfg.context)} onValueChange={(v) => set('context', Number(v))}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{CONTEXTS.map((c) => <SelectItem key={c} value={String(c)}>{c.toLocaleString()}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="Concurrency">
            <Input type="number" min={1} value={cfg.concurrency}
                   onChange={(e) => set('concurrency', Math.max(1, Number(e.target.value) || 1))} />
          </Field>
          <Field label="Mean sequence length" className="col-span-2"
                 hint={`${Math.min(cfg.avgSeqLen, cfg.context).toLocaleString()} tokens`}>
            <Slider min={128} max={cfg.context} step={128}
                    value={[Math.min(cfg.avgSeqLen, cfg.context)]} onValueChange={([v]) => set('avgSeqLen', v!)} />
          </Field>
          <Field label="Prefix cache hit rate" className="col-span-2"
                 hint={cfg.prefixHit === 0 ? 'off' : `${(cfg.prefixHit * 100).toFixed(0)}%`}>
            <Slider min={0} max={1} step={0.05} value={[cfg.prefixHit]} onValueChange={([v]) => set('prefixHit', v!)} />
          </Field>
          {cfg.prefixHit > 0 ? (
            <Field label="Shared prefix length" className="col-span-2">
              <Input type="number" min={0} value={cfg.prefixTokens}
                     onChange={(e) => set('prefixTokens', Math.max(0, Number(e.target.value) || 0))} />
            </Field>
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
        <CardContent>
          <AssumptionsPanel values={assume} onChange={(k, v) => setAssume((a) => ({ ...a, [k]: v }))} />
        </CardContent>
      </Card>
    </div>
  )

  if (!result.ok) {
    const e = result.error
    return (
      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        {controls}
        <Card className="h-fit border-destructive">
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
      </div>
    )
  }

  const r = result.value
  const p = r.plan
  const spec = p.input.model
  const devices = cfg.tp * cfg.pp
  const segments = [
    { key: 'weights', label: 'weights', bytes: p.weightBytesPerDevice, color: SEGMENT_COLORS.weights },
    { key: 'kv', label: 'kv cache', bytes: p.requiredKvBytes, color: SEGMENT_COLORS.kv },
    { key: 'activations', label: 'activations', bytes: p.overhead.activationBytes + p.overhead.logitsBytes, color: SEGMENT_COLORS.activations },
    { key: 'overhead', label: 'overhead', bytes: p.overhead.nonTorchBytes + p.overhead.graphBytes, color: SEGMENT_COLORS.overhead },
    { key: 'free', label: 'free', bytes: Math.max(0, p.freeBytes), color: SEGMENT_COLORS.free },
  ]

  return (
    <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
      {controls}

      <div className="space-y-4">
        <Card>
          <CardHeader className="pb-2">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className="text-base">{spec.id}</CardTitle>
              <Badge variant={p.fits ? 'default' : 'destructive'}>{p.fits ? 'fits' : 'does not fit'}</Badge>
              <Badge variant="outline">{r.label}</Badge>
              <span className="ml-auto text-xs text-muted-foreground">
                {devices}x {p.input.gpu.name} · {count(spec.numLayers)} layers · {count(p.weights.params.total)} params ({p.weights.method})
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
                  {Math.min(cfg.avgSeqLen, cfg.context).toLocaleString()} tokens
                </p>
                <Button
                  size="sm" variant="secondary" className="mt-2"
                  onClick={() => setCfg((c) => ({
                    ...c,
                    context: p.autofix!.maxModelLen,
                    concurrency: Math.max(1, p.autofix!.maxNumSeqs),
                  }))}
                >
                  apply
                </Button>
              </div>
            ) : null}

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="KV pool" value={`${count(p.maxTokens)} tokens`} sub={gib(p.availableKvBytes)} />
              <Stat label="Decode" value={`${r.throughput.decode.tokensPerSecond.toFixed(0)} tok/s`}
                    sub={`ITL ${seconds(r.throughput.decode.stepSeconds)}`} />
              <Stat label="Prefill" value={`${r.throughput.prefill.tokensPerSecond.toFixed(0)} tok/s`}
                    sub={`TTFT ${seconds(r.throughput.prefill.ttftSeconds)} · ${r.throughput.bound}-bound`} />
              <Stat {...quantCost(cfg.quant as QuantScheme)} />
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
          <CardContent className="pt-6">
            <Tabs defaultValue="run">
              <TabsList className="h-auto flex-wrap">
                <TabsTrigger value="run">Run it</TabsTrigger>
                <TabsTrigger value="memory">Memory</TabsTrigger>
                <TabsTrigger value="scaling">Scaling</TabsTrigger>
                <TabsTrigger value="host">Host</TabsTrigger>
                <TabsTrigger value="machines">Machines</TabsTrigger>
                <TabsTrigger value="speculative">Speculative</TabsTrigger>
                <TabsTrigger value="lora">Multi-LoRA</TabsTrigger>
                <TabsTrigger value="disagg">Disaggregated</TabsTrigger>
              </TabsList>

              <TabsContent value="run" className="mt-4 space-y-2">
                <p className="text-xs text-muted-foreground">
                  {p.fits ? 'These flags match the numbers above.' : 'Flags are emitted for the largest configuration that fits.'}
                </p>
                <CommandBlock command={r.command} />
              </TabsContent>

              <TabsContent value="memory" className="mt-4">
                <table className="w-full text-sm">
                  <tbody className="font-mono tabular-nums">
                    <Row label="model weights (per GPU)" bytes={p.weightBytesPerDevice} />
                    <Row label="kv cache required" bytes={p.requiredKvBytes} />
                    <Row label="kv pool reserved by engine" bytes={p.availableKvBytes} />
                    <Rule />
                    <Row label="cuda context" bytes={p.overhead.cudaContextBytes} />
                    <Row label="nccl buffers" bytes={p.overhead.commBytes} />
                    <Row label="cuda graphs" bytes={p.overhead.graphBytes} />
                    <Row label="activation peak" bytes={p.overhead.activationBytes} />
                    <Row label="sampling logits" bytes={p.overhead.logitsBytes} />
                    <Row label="allocator slack" bytes={p.overhead.fragmentationBytes} />
                    <Rule />
                    <Row label="usable vram per GPU" bytes={p.usableVramBytes} />
                    <Row label="free" bytes={p.freeBytes} />
                  </tbody>
                </table>
              </TabsContent>

              <TabsContent value="scaling" className="mt-4">
                <ScalingCurve req={req} current={cfg.concurrency} />
              </TabsContent>

              <TabsContent value="host" className="mt-4">
                <HostPanel result={r} />
              </TabsContent>

              <TabsContent value="machines" className="mt-4">
                <Machines req={req} />
              </TabsContent>

              <TabsContent value="speculative" className="mt-4"><Speculative req={req} result={r} /></TabsContent>
              <TabsContent value="lora" className="mt-4"><MultiLora result={r} /></TabsContent>
              <TabsContent value="disagg" className="mt-4"><Disaggregated req={req} /></TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

/**
 * The published accuracy cost of the selected scheme. An absent measurement is reported as
 * unmeasured, never as lossless — nobody has run the benchmark, which is not the same as it
 * being free.
 */
function quantCost(quant: QuantScheme): { label: string; value: string; sub: string } {
  const label = 'Quantization cost'
  const ms = qualityFor(quant)
  if (!ms.length) {
    return { label, value: 'unmeasured', sub: `no published perplexity for ${quant}` }
  }
  const pct = ms.map(pplPenalty).sort((a, b) => a - b)
  const lo = pct[0]!, hi = pct[pct.length - 1]!
  const fmt = (v: number) => `${(v * 100).toFixed(v * 100 < 1 ? 2 : 1)}%`
  const value = lo === hi ? `+${fmt(lo)} ppl` : `+${fmt(lo)}\u2013${fmt(hi)} ppl`
  const models = ms.length === 1 ? ms[0]!.model : `${ms.length} models`
  return { label, value, sub: `${models}, ${ms[0]!.dataset} — measured elsewhere, not here` }
}

function Row({ label, bytes }: { label: string; bytes: number }) {
  return (
    <tr>
      <td className="py-0.5 font-sans text-muted-foreground">{label}</td>
      <td className="py-0.5 text-right">{humanBytes(bytes)}</td>
    </tr>
  )
}

function Rule() {
  return (
    <tr>
      <td colSpan={2} className="py-1">
        <div className="h-px bg-border" />
      </td>
    </tr>
  )
}
