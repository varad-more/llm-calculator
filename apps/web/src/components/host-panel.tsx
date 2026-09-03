'use client'

import { useState } from 'react'
import { weightBytes, bytes as humanBytes, seconds, type SizingResult, type QuantScheme } from '@llmsize/core'
import { Field, Stat } from '@/components/field'
import { Input } from '@/components/ui/input'

/**
 * What the config costs outside the GPU: bytes on disk, bytes over the wire, the time both take,
 * and the interconnect the tensor-parallel all-reduce is actually riding on.
 *
 * The checkpoint you download is not the thing that lands in VRAM. fp8 and int8 are usually
 * quantized at load from a bf16 checkpoint, so the download is twice the resident size; awq,
 * gptq, mxfp4 and gguf ship pre-quantized and the two match.
 */
export function HostPanel({ result }: { result: SizingResult }) {
  const [gbps, setGbps] = useState(1)
  const [diskGbPerSec, setDiskGbPerSec] = useState(2)

  const p = result.plan
  const spec = p.input.model
  const served = p.input.quant as QuantScheme
  const ckptQuant = (spec.checkpointQuant ?? 'bf16') as QuantScheme

  // The measured file sizes when we have them; otherwise the checkpoint's own dtype, derived.
  const measured = spec.measuredWeightBytes
  const checkpointBytes = measured ?? weightBytes(spec, { quant: ckptQuant }).totalBytes
  const servedBytes = p.weights.totalBytes
  const quantizedAtLoad = Math.abs(checkpointBytes - servedBytes) / checkpointBytes > 0.01

  const downloadSeconds = checkpointBytes / ((gbps * 1e9) / 8)
  const loadSeconds = checkpointBytes / (diskGbPerSec * 1e9)

  const t = result.throughput
  const link = p.input.gpu.interconnect
  const devices = Math.max(1, p.input.parallel.tp) * Math.max(1, p.input.parallel.pp)
  const commShare = t.decode.stepSeconds > 0 ? t.decode.commSeconds / t.decode.stepSeconds : 0
  const prefillShare = t.prefill.ttftSeconds > 0 ? t.prefill.commSeconds / t.prefill.ttftSeconds : 0

  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat
          label="Checkpoint on disk"
          value={humanBytes(checkpointBytes)}
          sub={measured ? `${ckptQuant} · measured from file sizes` : `${ckptQuant} · derived`}
        />
        <Stat label="Weights in VRAM" value={humanBytes(servedBytes)} sub={`${served} · ${humanBytes(p.weightBytesPerDevice)} on each of ${devices}`} />
        <Stat label="Host RAM to plan for" value={humanBytes(checkpointBytes)} sub="page cache, reclaimable" />
      </div>

      {quantizedAtLoad ? (
        <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
          You download <span className="font-mono">{humanBytes(checkpointBytes)}</span> of{' '}
          <span className="font-mono">{ckptQuant}</span> and serve{' '}
          <span className="font-mono">{humanBytes(servedBytes)}</span> of{' '}
          <span className="font-mono">{served}</span> — the quantization happens at load, so disk and
          bandwidth are sized by the checkpoint, not by what ends up resident.
        </p>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Your download link" hint={`${gbps} Gbit/s`}>
          <Input type="number" min={0.1} step={0.1} value={gbps}
                 onChange={(e) => setGbps(Math.max(0.1, Number(e.target.value) || 0.1))} />
        </Field>
        <Field label="Your disk read rate" hint={`${diskGbPerSec} GB/s`}>
          <Input type="number" min={0.05} step={0.1} value={diskGbPerSec}
                 onChange={(e) => setDiskGbPerSec(Math.max(0.05, Number(e.target.value) || 0.05))} />
        </Field>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Stat label="First download" value={seconds(downloadSeconds)} sub={`${humanBytes(checkpointBytes)} at ${gbps} Gbit/s`} />
        <Stat label="Cold start from disk" value={seconds(loadSeconds)} sub={`re-read on every restart at ${diskGbPerSec} GB/s`} />
      </div>

      <div className="space-y-2 border-t pt-4">
        <p className="text-sm font-medium">Interconnect</p>
        {devices === 1 ? (
          <p className="text-xs text-muted-foreground">
            Single device — no all-reduce, so the link does not enter the arithmetic.
          </p>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-3">
              <Stat label="Link" value={link.kind} sub={`${(link.bidirectionalBytesPerSec / 1e9).toFixed(0)} GB/s bidirectional`} />
              <Stat label="All-reduce per decode step" value={seconds(t.decode.commSeconds)}
                    sub={`${(commShare * 100).toFixed(1)}% of the ${t.decode.itlMs.toFixed(1)} ms step`} />
              <Stat label="All-reduce per prefill chunk" value={seconds(t.prefill.commSeconds)}
                    sub={`of ${seconds(t.prefill.ttftSeconds)} TTFT`} />
            </div>
            <p className="text-xs text-muted-foreground">
              Ring all-reduce over {devices} ranks, once per layer. Prefill moves a whole chunk of
              activations and decode moves one token per sequence, which is why the link shows up in
              time-to-first-token ({(prefillShare * 100).toFixed(1)}%) long before it shows up between
              tokens ({(commShare * 100).toFixed(1)}%). On a true PCIe card the prefill share is where
              tensor parallelism stops paying for itself.
            </p>
          </>
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        Disk and download are exact byte counts. The two rates are yours to set — nothing here
        assumes a network or a disk on your behalf. Host RAM is the page cache the loader wants
        while it mmaps the checkpoint; it is reclaimable, so it is a comfort figure, not a floor.
      </p>
    </div>
  )
}
