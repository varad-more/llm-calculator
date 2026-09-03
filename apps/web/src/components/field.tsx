'use client'

import { Label } from '@/components/ui/label'

export function Field({ label, hint, children, className }: {
  label: string
  hint?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <Label className={`block space-y-1.5 text-xs ${className ?? ''}`}>
      <span className="flex items-center gap-1">
        {label}
        {hint ? <span className="ml-1 font-normal text-muted-foreground">{hint}</span> : null}
      </span>
      {children}
    </Label>
  )
}

export function Stat({ label, value, sub, tone }: {
  label: string
  value: string
  sub?: string
  tone?: 'good' | 'bad'
}) {
  const color = tone === 'good' ? 'text-emerald-600 dark:text-emerald-400'
    : tone === 'bad' ? 'text-destructive'
    : ''
  return (
    <div className="rounded-md border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`mt-0.5 font-mono text-lg tabular-nums ${color}`}>{value}</div>
      {sub ? <div className="text-xs text-muted-foreground">{sub}</div> : null}
    </div>
  )
}

/** A shell command with a copy button. The whole point of the tool is that this line runs. */
export function CommandBlock({ command }: { command: string }) {
  return (
    <div className="relative">
      <pre className="overflow-x-auto rounded-md bg-muted p-3 pr-16 font-mono text-xs leading-relaxed">{command}</pre>
      <button
        type="button"
        className="absolute right-2 top-2 rounded border bg-background px-2 py-1 text-[11px] text-muted-foreground hover:text-foreground"
        onClick={(e) => {
          navigator.clipboard?.writeText(command)
          const b = e.currentTarget
          b.textContent = 'copied'
          setTimeout(() => { b.textContent = 'copy' }, 1200)
        }}
      >
        copy
      </button>
    </div>
  )
}
