'use client'

import { useMemo } from 'react'
import {
  CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { size, type SizingRequest } from '@llmsize/core'

// Geometric ladder: batch effects are multiplicative, so even spacing here reads as a log axis.
const LADDER = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

interface Point {
  c: number
  tps: number
  perUser: number
  itl: number
  ttft: number
  kvGib: number
  fits: boolean
}

/**
 * Throughput and latency as a function of concurrency, run through the real allocator at every
 * point — so the curve stops where the KV pool actually runs out, not where a formula says it
 * should. Total tok/s saturates as the batch stops hiding the weight stream; per-sequence tok/s
 * falls the whole way. The crossing of those two is the batch-size decision.
 */
export function ScalingCurve({ req, current }: { req: SizingRequest; current: number }) {
  const points = useMemo(() => {
    const out: Point[] = []
    let misses = 0
    for (const c of LADDER) {
      let r
      try { r = size({ ...req, concurrency: c }) } catch { continue }
      out.push({
        c,
        tps: r.throughput.decode.tokensPerSecond,
        perUser: 1 / r.throughput.decode.stepSeconds,
        itl: r.throughput.decode.itlMs,
        ttft: r.throughput.prefill.ttftSeconds * 1000,
        kvGib: r.plan.requiredKvBytes / 2 ** 30,
        fits: r.plan.fits,
      })
      // Two points past the cliff is enough to draw it; the rest are the same answer.
      if (!r.plan.fits && ++misses >= 2) break
    }
    return out
  }, [req])

  const lastFit = [...points].reverse().find((p) => p.fits)
  const peak = points.filter((p) => p.fits).reduce((a, p) => Math.max(a, p.tps), 0)

  return (
    <div className="space-y-3">
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={points} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="c" type="category" tickLine={false} axisLine={false}
            stroke="var(--muted-foreground)" className="text-xs"
            label={{ value: 'concurrent sequences', position: 'insideBottom', offset: -2, fontSize: 10, fill: 'var(--muted-foreground)' }}
          />
          <YAxis
            yAxisId="tps" tickLine={false} axisLine={false} width={52}
            stroke="var(--chart-1)" className="text-xs"
            tickFormatter={(v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0))}
          />
          <YAxis
            yAxisId="itl" orientation="right" tickLine={false} axisLine={false} width={46}
            stroke="var(--chart-3)" className="text-xs"
            tickFormatter={(v: number) => `${v.toFixed(0)}ms`}
          />
          <Tooltip
            cursor={{ stroke: 'var(--border)' }}
            content={({ payload, label }) => {
              const p = payload?.[0]?.payload as Point | undefined
              if (!p) return null
              return (
                <div className="space-y-0.5 rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                  <div className="font-medium">{label} concurrent</div>
                  <Cell k="total" v={`${p.tps.toFixed(0)} tok/s`} />
                  <Cell k="per sequence" v={`${p.perUser.toFixed(1)} tok/s`} />
                  <Cell k="ITL" v={`${p.itl.toFixed(1)} ms`} />
                  <Cell k="TTFT" v={`${p.ttft.toFixed(0)} ms`} />
                  <Cell k="kv needed" v={`${p.kvGib.toFixed(1)} GiB`} />
                  {!p.fits ? <div className="pt-0.5 text-destructive">does not fit</div> : null}
                </div>
              )
            }}
          />
          <Line yAxisId="tps" type="monotone" dataKey="tps" stroke="var(--chart-1)" strokeWidth={2} dot={false} isAnimationActive={false} name="tok/s" />
          <Line yAxisId="itl" type="monotone" dataKey="itl" stroke="var(--chart-3)" strokeWidth={2} strokeDasharray="4 3" dot={false} isAnimationActive={false} name="ITL" />
          {lastFit ? (
            <ReferenceLine
              yAxisId="tps" x={lastFit.c} stroke="var(--destructive)" strokeWidth={2}
              label={{ value: 'kv pool full', position: 'insideTopRight', fontSize: 10, fill: 'var(--destructive)' }}
            />
          ) : null}
          <ReferenceLine
            yAxisId="tps" x={nearest(current)} stroke="var(--muted-foreground)" strokeDasharray="3 3"
            label={{ value: 'you', position: 'insideTopLeft', fontSize: 10, fill: 'var(--muted-foreground)' }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs">
        <Legend color="var(--chart-1)" label="total throughput" />
        <Legend color="var(--chart-3)" label="inter-token latency" dashed />
      </div>

      <p className="text-xs text-muted-foreground">
        {lastFit ? (
          <>
            The KV pool holds this workload up to <span className="font-mono">{lastFit.c}</span> concurrent
            sequences at a mean of <span className="font-mono">{(req.avgSeqLen ?? req.context).toLocaleString()}</span> tokens,
            peaking near <span className="font-mono">{peak.toFixed(0)} tok/s</span> at{' '}
            <span className="font-mono">{lastFit.itl.toFixed(0)} ms</span> between tokens. Every point is a full
            allocator run, so the wall is where memory actually ends.
          </>
        ) : (
          <>This configuration does not fit at any concurrency on the ladder. Lower the context, quantize, or add GPUs.</>
        )}{' '}
        Roofline, predicted: no queueing, no scheduler, no batching jitter.
      </p>
    </div>
  )
}

function nearest(c: number): number {
  return LADDER.reduce((a, b) => (Math.abs(b - c) < Math.abs(a - c) ? b : a))
}

function Cell({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center gap-4">
      <span className="text-muted-foreground">{k}</span>
      <span className="ml-auto font-mono tabular-nums">{v}</span>
    </div>
  )
}

function Legend({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-0.5 w-4" style={{ background: dashed ? `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 7px)` : color }} />
      <span className="text-muted-foreground">{label}</span>
    </span>
  )
}
