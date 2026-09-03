'use client'

import { useMemo, useState } from 'react'
import {
  size, listModels, resolveModel, parameterCounts, loraSizing, speculativeThroughput,
  emitSpeculativeFlags, planDisaggregated, gib, bytes as humanBytes, seconds, count,
  type SizingRequest, type SizingResult,
} from '@llmsize/core'
import { Field, Stat, CommandBlock } from '@/components/field'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

/** Named links, so the bandwidth in the disaggregation model is a fabric and not a magic number. */
const LINKS: { label: string; bytesPerSec: number }[] = [
  { label: 'NVLink 4 (900 GB/s)', bytesPerSec: 900e9 },
  { label: 'NVLink 3 (600 GB/s)', bytesPerSec: 600e9 },
  { label: 'InfiniBand NDR 400G (50 GB/s)', bytesPerSec: 50e9 },
  { label: 'InfiniBand HDR 200G (25 GB/s)', bytesPerSec: 25e9 },
  { label: 'Ethernet 100G (12.5 GB/s)', bytesPerSec: 12.5e9 },
  { label: 'Ethernet 25G (3.1 GB/s)', bytesPerSec: 3.125e9 },
]

const RANKS = [8, 16, 32, 64, 128, 256]

/** Snapshots ordered small-to-large, so the draft-model picker offers plausible drafts first. */
function modelsBySize(): { id: string; params: number }[] {
  return listModels()
    .map((id) => ({ id, params: parameterCounts(resolveModel(id)).total }))
    .sort((a, b) => a.params - b.params)
}

export function Speculative({ req, result }: { req: SizingRequest; result: SizingResult }) {
  const candidates = useMemo(() => modelsBySize(), [])
  const [draft, setDraft] = useState(candidates[0]!.id)
  const [k, setK] = useState(4)
  const [acceptance, setAcceptance] = useState(0.7)

  const draftStep = useMemo(() => {
    try {
      // The draft runs on the same devices as the target, so it is sized the same way.
      return size({ ...req, model: draft, concurrency: 1, quant: 'bf16' }).throughput.decode.stepSeconds
    } catch {
      return null
    }
  }, [req, draft])

  if (draftStep === null) {
    return <p className="text-sm text-muted-foreground">Could not size {draft} on this hardware.</p>
  }

  const cfg = { numSpeculativeTokens: k, acceptanceRate: acceptance, draftStepSeconds: draftStep }
  const est = speculativeThroughput(result.throughput, cfg)

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Speculation is a <strong>latency</strong> optimisation. A cycle costs {k} draft steps plus one
        target forward, and yields a truncated geometric number of tokens. When the draft is slow or
        wrong the speedup drops below 1 — this panel will tell you so rather than hide it.
      </p>

      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="Draft model" hint="smallest first">
          <Select value={draft} onValueChange={setDraft}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              {candidates.map((c) => (
                <SelectItem key={c.id} value={c.id}>{c.id} · {count(c.params)}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Field label="Speculative tokens" hint={String(k)}>
          <Slider aria-label="Speculative tokens" min={1} max={10} step={1} value={[k]} onValueChange={([v]) => setK(v!)} />
        </Field>
        <Field label="Acceptance rate" hint={`${(acceptance * 100).toFixed(0)}% — measure it, do not guess`}>
          <Slider aria-label="Acceptance rate" min={0.05} max={0.99} step={0.01} value={[acceptance]} onValueChange={([v]) => setAcceptance(v!)} />
        </Field>
      </div>

      <div className="grid gap-4 sm:grid-cols-4">
        <Stat label="Speedup" value={`${est.speedup.toFixed(2)}x`} tone={est.worthwhile ? 'good' : 'bad'}
          sub={est.worthwhile ? 'worth enabling' : 'net loss — draft too slow or too wrong'} />
        <Stat label="ITL" value={`${est.itlMs.toFixed(1)} ms`} sub={`was ${(result.throughput.decode.stepSeconds * 1000).toFixed(1)} ms`} />
        <Stat label="Tokens / cycle" value={est.tokensPerCycle.toFixed(2)} sub={`of ${k + 1} proposed`} />
        <Stat label="Draft step" value={seconds(draftStep)} sub="same GPUs as the target" />
      </div>

      {draftStep >= result.throughput.decode.stepSeconds ? (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm text-muted-foreground">
          This draft is not faster than the target it is meant to accelerate, so no acceptance rate
          can make speculation pay. Pick a smaller draft.
        </p>
      ) : null}

      <CommandBlock command={`${result.command} \\\n    ${emitSpeculativeFlags(draft, cfg).join(' ')}`} />
    </div>
  )
}

export function MultiLora({ result }: { result: SizingResult }) {
  const [maxLoras, setMaxLoras] = useState(4)
  const [rank, setRank] = useState(16)
  const p = result.plan
  const sizing = loraSizing(p.input.model, { maxLoras, maxLoraRank: rank }, p.input.parallel)
  const fits = sizing.totalBytesPerDevice <= p.freeBytes

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        A rank-<em>r</em> adapter stores <span className="font-mono">r(m+n)</span> parameters per
        projection, not <span className="font-mono">r·m·n</span>. Every slot is pre-allocated at the
        maximum rank whether the loaded adapter uses it or not, so the resident cost is fixed by
        <span className="font-mono"> --max-loras</span>, not by which adapters are live.
      </p>

      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Resident adapter slots (--max-loras)">
          <Input type="number" min={1} max={64} value={maxLoras}
            onChange={(e) => setMaxLoras(Math.max(1, Number(e.target.value) || 1))} />
        </Field>
        <Field label="Max rank (--max-lora-rank)">
          <Select value={String(rank)} onValueChange={(v) => setRank(Number(v))}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>{RANKS.map((r) => <SelectItem key={r} value={String(r)}>{r}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Per adapter" value={humanBytes(sizing.bytesPerAdapter)} sub={`${(sizing.paramsPerAdapter / 1e6).toFixed(1)}M params`} />
        <Stat label="Resident, per GPU" value={humanBytes(sizing.totalBytesPerDevice)} sub={`${maxLoras} slots at rank ${rank}`} />
        <Stat label="Free after adapters" value={gib(Math.max(0, p.freeBytes - sizing.totalBytesPerDevice))}
          tone={fits ? 'good' : 'bad'} sub={fits ? 'still fits' : 'over budget — lower --max-loras'} />
      </div>

      <CommandBlock command={`${result.command} \\\n    ${sizing.flags.join(' \\\n    ')}`} />
    </div>
  )
}

export function Disaggregated({ req }: { req: SizingRequest }) {
  const [prefillGpus, setPrefillGpus] = useState(2)
  const [prefillTp, setPrefillTp] = useState(1)
  const [decodeGpus, setDecodeGpus] = useState(2)
  const [decodeTp, setDecodeTp] = useState(1)
  const [promptTokens, setPromptTokens] = useState(4096)
  const [outputTokens, setOutputTokens] = useState(512)
  const [link, setLink] = useState(LINKS[2]!.bytesPerSec)

  const plan = useMemo(() => {
    try {
      return {
        ok: true as const,
        value: planDisaggregated({
          ...req,
          prefill: { gpus: prefillGpus, tp: prefillTp },
          decode: { gpus: decodeGpus, tp: decodeTp },
          context: Math.max(req.context ?? 8192, promptTokens + outputTokens),
          prefillConcurrency: Math.max(1, Math.floor(prefillGpus / prefillTp)),
          decodeConcurrency: req.concurrency ?? 32,
          promptTokens, outputTokens, transferBytesPerSec: link,
        }),
      }
    } catch (e) {
      return { ok: false as const, error: e as Error }
    }
  }, [req, prefillGpus, prefillTp, decodeGpus, decodeTp, promptTokens, outputTokens, link])

  if (!plan.ok) return <p className="text-sm text-destructive font-mono">{plan.error.message}</p>
  const d = plan.value

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Prefill is compute-bound and decode is bandwidth-bound, so splitting them lets each pool be
        parallelised on its own terms. The price is that every request&apos;s KV cache crosses the wire
        once — which, on anything short of NVLink, is usually the actual bottleneck.
      </p>

      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="Prefill GPUs"><Input type="number" min={1} value={prefillGpus} onChange={(e) => setPrefillGpus(Math.max(1, Number(e.target.value) || 1))} /></Field>
        <Field label="Prefill TP"><Input type="number" min={1} value={prefillTp} onChange={(e) => setPrefillTp(Math.max(1, Number(e.target.value) || 1))} /></Field>
        <Field label="Prompt tokens"><Input type="number" min={1} value={promptTokens} onChange={(e) => setPromptTokens(Math.max(1, Number(e.target.value) || 1))} /></Field>
        <Field label="Decode GPUs"><Input type="number" min={1} value={decodeGpus} onChange={(e) => setDecodeGpus(Math.max(1, Number(e.target.value) || 1))} /></Field>
        <Field label="Decode TP"><Input type="number" min={1} value={decodeTp} onChange={(e) => setDecodeTp(Math.max(1, Number(e.target.value) || 1))} /></Field>
        <Field label="Output tokens"><Input type="number" min={1} value={outputTokens} onChange={(e) => setOutputTokens(Math.max(1, Number(e.target.value) || 1))} /></Field>
        <Field label="Interconnect" className="sm:col-span-3">
          <Select value={String(link)} onValueChange={(v) => setLink(Number(v))}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>{LINKS.map((l) => <SelectItem key={l.label} value={String(l.bytesPerSec)}>{l.label}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
      </div>

      <div className="grid gap-4 sm:grid-cols-4">
        <Stat label="Bottleneck" value={d.balance.bottleneck} tone={d.balance.bottleneck === 'transfer' ? 'bad' : 'good'}
          sub="slowest stage at these pool sizes" />
        <Stat label="KV per request" value={gib(d.transfer.bytesPerRequest)} sub={`${seconds(d.transfer.secondsPerRequest)} on the wire`} />
        <Stat label="Link must sustain" value={`${(d.transfer.requiredBytesPerSec / 1e9).toFixed(1)} GB/s`}
          sub={`${(d.transfer.shareOfTtft * 100).toFixed(0)}% of TTFT is transfer`} />
        <Stat label="Decode per prefill" value={Number.isFinite(d.balance.decodePerPrefill) ? d.balance.decodePerPrefill.toFixed(2) : '∞'}
          sub="instances for a balanced pipeline" />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {([['Prefill pool', d.prefill], ['Decode pool', d.decode]] as const).map(([label, r]) => (
          <Card key={label}>
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <CardTitle className="text-sm">{label}</CardTitle>
                <Badge variant={r.plan.fits ? 'default' : 'destructive'}>{r.plan.fits ? 'fits' : 'does not fit'}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-1 font-mono text-xs tabular-nums">
              <div className="flex justify-between"><span className="font-sans text-muted-foreground">weights/GPU</span>{gib(r.plan.weightBytesPerDevice)}</div>
              <div className="flex justify-between"><span className="font-sans text-muted-foreground">kv pool</span>{gib(r.plan.availableKvBytes)}</div>
              <div className="flex justify-between"><span className="font-sans text-muted-foreground">free</span>{gib(r.plan.freeBytes)}</div>
              <div className="flex justify-between">
                <span className="font-sans text-muted-foreground">req/s per instance</span>
                {(label === 'Prefill pool' ? d.balance.prefillRequestsPerSec : d.balance.decodeRequestsPerSec).toFixed(2)}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {d.warnings.length ? (
        <div className="space-y-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
          {d.warnings.map((w) => (
            <p key={w.code} className="text-muted-foreground">
              <span className="mr-2 font-mono text-[10px]">[{w.code}]</span>{w.message}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  )
}
