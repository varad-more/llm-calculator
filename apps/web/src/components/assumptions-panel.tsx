'use client'

import { defaultAssumptions } from '@llmsize/core'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

/**
 * Always visible, never buried. Every empirical constant the prediction rests on, with its
 * rationale, its source and an honest confidence label — and every one editable, because a
 * number you cannot argue with is a number you cannot trust.
 */
export function AssumptionsPanel({ values, onChange }: {
  values: Record<string, number>
  onChange: (key: string, value: number) => void
}) {
  const defaults = defaultAssumptions()
  return (
    <div className="space-y-3">
      {Object.entries(defaults).map(([key, a]) => {
        const current = values[key] ?? a.value
        const changed = current !== a.value
        return (
          <div key={key} className="grid grid-cols-[1fr_7rem] items-start gap-3">
            <div className="min-w-0">
              <Label htmlFor={`assume-${key}`} className="flex-wrap gap-x-2 font-mono text-xs">
                <span className="min-w-0 break-all">{key}</span>
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
              </Label>
              <p className="mt-1 text-xs leading-snug text-muted-foreground">{a.rationale}</p>
              <a
                href={a.source_url} target="_blank" rel="noreferrer"
                className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
              >
                source
              </a>
            </div>
            <div>
              <Input
                id={`assume-${key}`} type="number" inputMode="decimal"
                step={a.value < 1 ? 0.01 : 1}
                value={current}
                onChange={(e) => onChange(key, Number(e.target.value))}
                className={'font-mono text-xs tabular-nums ' + (changed ? 'border-amber-500' : '')}
              />
              <div className="mt-1 text-right text-[10px] text-muted-foreground">
                {changed ? <button className="underline" onClick={() => onChange(key, a.value)}>reset</button> : a.unit}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
