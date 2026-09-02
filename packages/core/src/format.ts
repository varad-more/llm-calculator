// The ONLY place bytes turn into human units. Everything upstream is bytes.
const GiB = 2 ** 30
const MiB = 2 ** 20

/** Bytes -> "12.34 GiB". @see docs/MATH.md#units */
export function gib(bytes: number, digits = 2): string {
  return `${(bytes / GiB).toFixed(digits)} GiB`
}

/** Bytes -> "123.4 MiB" or "1.23 GiB", whichever reads better. @see docs/MATH.md#units */
export function bytes(n: number): string {
  return Math.abs(n) >= GiB ? gib(n) : `${(n / MiB).toFixed(1)} MiB`
}

/** Seconds -> "12.3 ms" / "1.23 s". @see docs/MATH.md#units */
export function seconds(s: number): string {
  return s < 1 ? `${(s * 1000).toFixed(1)} ms` : `${s.toFixed(2)} s`
}

/** 1234567 -> "1.23M". @see docs/MATH.md#units */
export function count(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}
