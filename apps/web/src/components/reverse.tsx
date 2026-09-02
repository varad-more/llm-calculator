'use client'

import { useMemo } from 'react'
import { reverseLookup, listGpus, gib, count, type EngineName, type QuantScheme } from '@llmsize/core'
import { useUrlState } from '@/lib/url-state'
import { Field, CommandBlock } from '@/components/field'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

const ENGINES: EngineName[] = ['vllm', 'sglang', 'trtllm', 'llamacpp']
const QUANT_SETS: Record<string, QuantScheme[]> = {
  'bf16 · fp8 · int4': ['bf16', 'fp8', 'awq-int4'],
  'bf16 only': ['bf16'],
  'quantized only': ['fp8', 'awq-int4', 'gptq-int4'],
  'everything': ['bf16', 'fp8', 'int8', 'awq-int4', 'gptq-int4'],
}

const DEFAULTS = {
  gpu: 'h100-sxm-80',
  gpuCount: 1,
  engine: 'vllm' as string,
  concurrency: 1,
  quants: 'bf16 · fp8 · int4',
  fittingOnly: 1,
}

export function ReverseLookup() {
  const gpus = listGpus()
  const [cfg, set] = useUrlState(DEFAULTS)

  const rows = useMemo(() => reverseLookup({
    gpu: cfg.gpu,
    gpuCount: cfg.gpuCount,
    engine: cfg.engine as EngineName,
    concurrency: cfg.concurrency,
    quants: QUANT_SETS[cfg.quants] ?? QUANT_SETS['bf16 · fp8 · int4'],
    fittingOnly: cfg.fittingOnly === 1,
    limit: 40,
  }), [cfg])

  const gpu = gpus.find((g) => g.id === cfg.gpu)

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="pb-3"><CardTitle className="text-sm">Hardware you have</CardTitle></CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <Field label="GPU">
            <Select value={cfg.gpu} onValueChange={(v) => set('gpu', v)}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{gpus.map((g) => <SelectItem key={g.id} value={g.id}>{g.name}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="How many">
            <Input type="number" min={1} max={64} value={cfg.gpuCount}
              onChange={(e) => set('gpuCount', Math.max(1, Number(e.target.value) || 1))} />
          </Field>
          <Field label="Engine">
            <Select value={cfg.engine} onValueChange={(v) => set('engine', v)}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{ENGINES.map((e) => <SelectItem key={e} value={e}>{e}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="Quantizations to try">
            <Select value={cfg.quants} onValueChange={(v) => set('quants', v)}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{Object.keys(QUANT_SETS).map((k) => <SelectItem key={k} value={k}>{k}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="Concurrent requests">
            <Input type="number" min={1} value={cfg.concurrency}
              onChange={(e) => set('concurrency', Math.max(1, Number(e.target.value) || 1))} />
          </Field>
          <div className="flex items-center gap-2 sm:col-span-2 lg:col-span-5">
            <Switch id="fitting" checked={cfg.fittingOnly === 1} onCheckedChange={(v) => set('fittingOnly', v ? 1 : 0)} />
            <Label htmlFor="fitting" className="text-xs font-normal text-muted-foreground">
              Only show configurations that fit
            </Label>
          </div>
        </CardContent>
      </Card>

      <p className="text-sm text-muted-foreground">
        {rows.length} configuration{rows.length === 1 ? '' : 's'} on {cfg.gpuCount}x {gpu?.name ?? cfg.gpu}
        {' '}({gib((gpu?.vramBytes ?? 0) * cfg.gpuCount)} total), ranked by context × decode throughput.
        Every row went through the same allocator as the sizing page — these are predictions, not a lookup table.
      </p>

      <div className="overflow-x-auto rounded-md border">
        <table className="w-full min-w-[52rem] text-sm">
          <thead className="bg-muted/50 text-left text-xs text-muted-foreground">
            <tr>
              <th className="px-3 py-2 font-medium">Model</th>
              <th className="px-3 py-2 font-medium">Weights</th>
              <th className="px-3 py-2 font-medium">KV</th>
              <th className="px-3 py-2 font-medium">TP</th>
              <th className="px-3 py-2 text-right font-medium">Context</th>
              <th className="px-3 py-2 text-right font-medium">Weights/GPU</th>
              <th className="px-3 py-2 text-right font-medium">KV pool</th>
              <th className="px-3 py-2 text-right font-medium">Free</th>
              <th className="px-3 py-2 text-right font-medium">Decode</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r, i) => (
              <tr key={`${r.model}-${r.quant}-${r.kvDtype}-${r.context}-${r.tp}`}
                  className={`border-t ${r.fits ? '' : 'opacity-50'} ${i % 2 ? 'bg-muted/20' : ''}`}>
                <td className="px-3 py-1.5 font-sans">{r.model}</td>
                <td className="px-3 py-1.5">{r.quant}</td>
                <td className="px-3 py-1.5">{r.kvDtype}</td>
                <td className="px-3 py-1.5">{r.tp}</td>
                <td className="px-3 py-1.5 text-right">{r.context.toLocaleString()}</td>
                <td className="px-3 py-1.5 text-right">{gib(r.weightBytesPerDevice)}</td>
                <td className="px-3 py-1.5 text-right">{count(r.kvPoolTokens)} tok</td>
                <td className="px-3 py-1.5 text-right">{gib(r.freeBytes)}</td>
                <td className="px-3 py-1.5 text-right">{r.decodeTokensPerSecond.toFixed(0)} tok/s</td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr><td colSpan={9} className="px-3 py-6 text-center font-sans text-muted-foreground">
                Nothing fits. Add GPUs, quantize harder, or turn off &ldquo;only show configurations that fit&rdquo;
                to see how far off each candidate is.
              </td></tr>
            ) : null}
          </tbody>
        </table>
      </div>

      {rows[0] ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Best fit</CardTitle>
            <p className="text-xs text-muted-foreground">{rows[0].model} at {rows[0].quant}, {rows[0].context.toLocaleString()} context</p>
          </CardHeader>
          <CardContent><CommandBlock command={rows[0].command} /></CardContent>
        </Card>
      ) : null}
    </div>
  )
}
