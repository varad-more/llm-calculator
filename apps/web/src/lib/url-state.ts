'use client'

import { useEffect, useRef, useState } from 'react'

type Scalar = string | number

/**
 * Config state mirrored into the query string, so a sizing is a link you can paste.
 *
 * Hydration-safe on purpose: the first client render uses `defaults`, which is exactly what
 * the statically exported HTML contains. The URL is only applied in an effect after mount,
 * so there is never a server/client mismatch. Only values that differ from the default are
 * written back, which keeps shared links readable.
 */
export function useUrlState<T extends Record<string, Scalar>>(defaults: T) {
  const base = useRef(defaults)
  const [state, setState] = useState<T>(defaults)
  const mounted = useRef(false)

  useEffect(() => {
    const q = new URLSearchParams(window.location.search)
    const next = { ...base.current }
    for (const key of Object.keys(base.current) as (keyof T)[]) {
      const raw = q.get(String(key))
      if (raw === null) continue
      if (typeof base.current[key] === 'number') {
        const n = Number(raw)
        if (Number.isFinite(n)) next[key] = n as T[keyof T]
      } else {
        next[key] = raw as T[keyof T]
      }
    }
    mounted.current = true
    setState(next)
  }, [])

  useEffect(() => {
    if (!mounted.current) return
    const q = new URLSearchParams()
    for (const [k, v] of Object.entries(state)) {
      if (v !== base.current[k]) q.set(k, String(v))
    }
    const s = q.toString()
    window.history.replaceState(null, '', s ? `?${s}` : window.location.pathname)
  }, [state])

  function set<K extends keyof T>(key: K, value: T[K]) {
    setState((c) => ({ ...c, [key]: value }))
  }

  return [state, set, setState] as const
}
