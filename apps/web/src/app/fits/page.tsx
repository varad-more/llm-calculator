import { ReverseLookup } from '@/components/reverse'

export const metadata = {
  title: 'What fits on my GPU',
  description:
    'Given the GPUs you have, enumerate every model, quantization and context length that actually ' +
    'fits — sized through the same engine allocator, with the serving command for each.',
  alternates: { canonical: '/fits/' },
}

export default function FitsPage() {
  return (
    <main className="mx-auto max-w-7xl px-4 py-8 md:px-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">What fits on my GPU</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          The sizing page answers &ldquo;does this config fit&rdquo;. This one runs the question backwards:
          given the hardware, sweep every model, quantization, KV dtype, context and tensor-parallel
          degree through the same allocator and rank what survives.
        </p>
      </header>
      <ReverseLookup />
    </main>
  )
}
