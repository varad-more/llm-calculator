'use client'

import { useMemo, useState } from 'react'
import { compareMachines, gib, type SizingRequest, type MachineOption } from '@llmsize/core'
import { Field } from '@/components/field'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { CommandBlock } from '@/components/field'

/** Blank means "no target", not zero. */
function num(s: string): number | undefined {
  const v = Number(s)
  return s.trim() === '' || !Number.isFinite(v) || v <= 0 ? undefined : v
}

const money = (v: number) =>
  !Number.isFinite(v) ? '—' : v < 0.01 ? `$${v.toFixed(4)}` : v < 1 ? `$${v.toFixed(3)}` : `$${v.toFixed(2)}`

export function Machines({ req }: { req: SizingRequest }) {
  const [ttft, setTtft] = useState('')
  const [perUser, setPerUser] = useState('')
  const [rates, setRates] = useState<Record<string, number>>({})
  const [open, setOpen] = useState<string | null>(null)

  const slo = useMemo(() => {
    const maxTtftSeconds = num(ttft)
    const minTokensPerSecondPerUser = num(perUser)
    return maxTtftSeconds || minTokensPerSecondPerUser
      ? { maxTtftSeconds: maxTtftSeconds ? maxTtftSeconds / 1000 : undefined, minTokensPerSecondPerUser }
      : undefined
  }, [ttft, perUser])

  const rows = useMemo(
    () => compareMachines({ req, slo, usdPerHour: rates }),
    [req, slo, rates],
  )

  if (!rows.length) {
    return <p className="text-sm text-muted-foreground">No instance in the catalogue can size this model.</p>
  }

  const priced = rows[0]!.instance
  const winner = rows.find((r) => r.fits && !r.sloMisses.length)

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <Field label="Max TTFT" hint="ms, blank for none">
          <Input inputMode="numeric" placeholder="e.g. 500" value={ttft} onChange={(e) => setTtft(e.target.value)} />
        </Field>
        <Field label="Min speed per user" hint="tok/s, blank for none">
          <Input inputMode="numeric" placeholder="e.g. 30" value={perUser} onChange={(e) => setPerUser(e.target.value)} />
        </Field>
        <div className="flex items-end text-xs text-muted-foreground">
          {slo
            ? winner
              ? <p>Cheapest machine that holds this SLO: <span className="font-mono text-foreground">{winner.instance.id}</span> at {money(winner.usdPerMillionOutputTokens)}/1M output tokens.</p>
              : <p className="text-amber-600 dark:text-amber-400">Nothing in the catalogue holds this SLO. Rows below are ordered by how close they get.</p>
            : <p>Set a latency target to narrow this to machines that can hold it.</p>}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs text-muted-foreground">
              <th className="py-2 pr-3 font-medium">Instance</th>
              <th className="py-2 pr-3 font-medium">Layout</th>
              <th className="py-2 pr-3 text-right font-medium">$/hour</th>
              <th className="py-2 pr-3 text-right font-medium">tok/s</th>
              <th className="py-2 pr-3 text-right font-medium">TTFT</th>
              <th className="py-2 pr-3 text-right font-medium">per user</th>
              <th className="py-2 pr-3 text-right font-medium">$/1M in</th>
              <th className="py-2 pr-3 text-right font-medium">$/1M out</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {rows.map((r) => (
              <Row key={r.instance.id} r={r} open={open === r.instance.id}
                   onToggle={() => setOpen(open === r.instance.id ? null : r.instance.id)}
                   rate={rates[r.instance.id]}
                   onRate={(v) => setRates((s) => {
                     const next = { ...s }
                     if (v === undefined) delete next[r.instance.id]; else next[r.instance.id] = v
                     return next
                   })} />
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs leading-relaxed text-muted-foreground">
        Prices are <span className="font-medium">{priced.priceBasis}</span> in{' '}
        <span className="font-mono">{priced.region}</span>, read on{' '}
        <span className="font-mono">{priced.priceRetrieved}</span> from{' '}
        <a href={priced.price_source_url} className="underline underline-offset-2">the AWS price list</a>.
        They move — click any rate to substitute your own spot or savings-plan price. A machine bills
        whether or not it is busy, so every figure here is the price at full saturation: at 40%
        utilization multiply by 2.5, which is the same as typing 2.5x the rate.
      </p>
      <p className="text-xs leading-relaxed text-muted-foreground">
        <span className="font-medium text-foreground">tok/s</span> is the whole machine — all replicas.{' '}
        <span className="font-medium text-foreground">TTFT</span> and{' '}
        <span className="font-medium text-foreground">per user</span> are one replica&apos;s, because one
        replica serves a given request. Only the cheapest size at each (family, GPU count) is listed: a
        g5.16xlarge has the same single A10G as a g5.xlarge and costs four times as much.
      </p>
    </div>
  )
}

function Row({ r, open, onToggle, rate, onRate }: {
  r: MachineOption
  open: boolean
  onToggle: () => void
  rate?: number
  onRate: (v: number | undefined) => void
}) {
  const missed = r.sloMisses.length > 0
  const dim = !r.fits || missed ? 'opacity-45' : ''
  return (
    <>
      <tr className={`cursor-pointer border-b hover:bg-muted/40 ${dim}`} onClick={onToggle}>
        <td className="py-1.5 pr-3 font-mono text-xs">
          {r.instance.id}
          {!r.fits ? <Badge variant="destructive" className="ml-2 align-middle text-[10px]">no fit</Badge> : null}
        </td>
        <td className="py-1.5 pr-3 font-mono text-xs text-muted-foreground">
          {r.replicas > 1 ? `${r.replicas} x tp${r.tp}` : `tp${r.tp}`}
        </td>
        <td className="py-1.5 pr-3 text-right font-mono text-xs">
          <input
            aria-label={`Hourly rate for ${r.instance.id}`}
            className={`w-20 rounded border bg-transparent px-1 text-right ${rate !== undefined ? 'border-foreground/40' : 'border-transparent hover:border-border'}`}
            inputMode="decimal"
            value={rate ?? r.instance.usdPerHour}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => {
              const v = Number(e.target.value)
              onRate(Number.isFinite(v) && v > 0 && e.target.value.trim() !== '' ? v : undefined)
            }}
          />
        </td>
        <td className="py-1.5 pr-3 text-right font-mono text-xs">{r.decodeTokensPerSecond.toFixed(0)}</td>
        <td className="py-1.5 pr-3 text-right font-mono text-xs">{(r.ttftSeconds * 1000).toFixed(0)} ms</td>
        <td className="py-1.5 pr-3 text-right font-mono text-xs">{(1000 / r.itlMs).toFixed(0)} tok/s</td>
        <td className="py-1.5 pr-3 text-right font-mono text-xs">{money(r.usdPerMillionInputTokens)}</td>
        <td className="py-1.5 pr-3 text-right font-mono text-xs font-medium">{money(r.usdPerMillionOutputTokens)}</td>
      </tr>
      {missed || !r.fits ? (
        <tr className={`border-b ${dim}`}>
          <td colSpan={8} className="pb-1.5 pl-2 text-[11px] text-muted-foreground">
            {!r.fits ? 'does not fit at any tensor-parallel degree this machine can form' : r.sloMisses.join(' · ')}
          </td>
        </tr>
      ) : null}
      {open ? (
        <tr className="border-b bg-muted/30">
          <td colSpan={8} className="space-y-2 p-3 text-xs">
            <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-4">
              <Detail label="GPUs" value={`${r.instance.gpuCount} x ${r.replica.plan.input.gpu.name}`} />
              <Detail label="Replicas" value={`${r.replicas} x tp${r.tp}, ${r.concurrencyPerReplica} seqs each`} />
              <Detail label="KV pool" value={`${(r.replica.plan.maxTokens / 1000).toFixed(0)}k tokens per replica`} />
              <Detail label="Host" value={`${r.instance.vcpus} vCPU · ${gib(r.instance.hostRamBytes, 0)} RAM · ${(r.instance.localDiskBytes / 1e12).toFixed(1)} TB NVMe · ${r.instance.networkGbps} Gbps`} />
            </div>
            {!r.hostRamHoldsCheckpoint ? (
              <p className="text-amber-600 dark:text-amber-400">
                Host RAM is smaller than the checkpoint, so the page cache cannot hold a whole copy and
                every restart re-reads it from disk or the network.
              </p>
            ) : null}
            <CommandBlock command={r.replica.command} />
          </td>
        </tr>
      ) : null}
    </>
  )
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-muted-foreground">{label}</span>
      <div className="font-mono">{value}</div>
    </div>
  )
}
