'use client'

import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { gib } from '@llmsize/core'

const GiB = 2 ** 30

export interface Segment {
  key: string
  label: string
  bytes: number
  color: string
}

export const SEGMENT_COLORS = {
  weights: 'var(--chart-1)',
  kv: 'var(--chart-2)',
  activations: 'var(--chart-3)',
  overhead: 'var(--chart-4)',
  free: 'var(--chart-5)',
} as const

/**
 * One stacked horizontal bar per GPU, with a hard capacity line at the device's usable VRAM.
 * Anything past the line is the amount by which the configuration is over budget, which is
 * the number people actually need to see.
 */
export function MemoryBar({ segments, capacityBytes, budgetBytes }: {
  segments: Segment[]
  capacityBytes: number
  budgetBytes: number
}) {
  const row: Record<string, number | string> = { name: 'per GPU' }
  for (const s of segments) row[s.key] = s.bytes / GiB
  const total = segments.reduce((a, s) => a + s.bytes, 0)
  const max = Math.max(total, capacityBytes) / GiB

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={96}>
        <BarChart data={[row]} layout="vertical" margin={{ top: 8, right: 8, bottom: 8, left: 8 }} barSize={44}>
          <XAxis
            type="number" domain={[0, max * 1.02]}
            tickFormatter={(v: number) => `${v.toFixed(0)}`}
            tickLine={false} axisLine={false}
            className="text-xs" stroke="var(--muted-foreground)"
          />
          <YAxis type="category" dataKey="name" hide />
          <Tooltip
            cursor={false}
            content={({ payload }) => {
              if (!payload?.length) return null
              return (
                <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                  {payload.map((p) => (
                    <div key={String(p.dataKey)} className="flex items-center gap-2">
                      <span className="inline-block size-2 rounded-full" style={{ background: p.color }} />
                      <span className="text-muted-foreground">{segments.find((s) => s.key === p.dataKey)?.label}</span>
                      <span className="ml-auto font-mono tabular-nums">{Number(p.value).toFixed(2)} GiB</span>
                    </div>
                  ))}
                </div>
              )
            }}
          />
          {segments.map((s) => (
            <Bar key={s.key} dataKey={s.key} stackId="mem" fill={s.color} isAnimationActive={false}>
              <Cell fill={s.color} />
            </Bar>
          ))}
          <ReferenceLine
            x={budgetBytes / GiB} stroke="var(--muted-foreground)" strokeDasharray="3 3"
            label={{ value: 'utilization budget', position: 'insideTopRight', fontSize: 10, fill: 'var(--muted-foreground)' }}
          />
          <ReferenceLine
            x={capacityBytes / GiB} stroke="var(--destructive)" strokeWidth={2}
            label={{ value: 'VRAM', position: 'insideTopRight', fontSize: 10, fill: 'var(--destructive)' }}
          />
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 text-xs">
        {segments.map((s) => (
          <span key={s.key} className="flex items-center gap-1.5">
            <span className="inline-block size-2 rounded-full" style={{ background: s.color }} />
            <span className="text-muted-foreground">{s.label}</span>
            <span className="font-mono tabular-nums">{gib(s.bytes)}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
