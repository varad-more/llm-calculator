import { notFound } from 'next/navigation'
import Link from 'next/link'
import { Sizer } from '@/components/sizer'
import { allSlugs, modelForSlug, specFor, describe } from '@/lib/models'
import { size, gib, count } from '@llmsize/core'

// The only reason this app is Next and not Vite: one static page per model, pre-rendered
// with real numbers, for the "<model> vram requirements" searches people actually run.
export function generateStaticParams() {
  return allSlugs().map((slug) => ({ slug }))
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const id = modelForSlug(slug)
  if (!id) return {}
  const spec = specFor(id)
  const h100 = size({ model: id, gpu: 'h100-sxm-80', engine: 'vllm', context: 8192, concurrency: 1 })
  return {
    title: `${id} VRAM requirements and vLLM config — llmsize`,
    description:
      `${id} needs ${gib(h100.plan.weights.totalBytes)} for weights at bf16 (${count(spec.numLayers)} layers, ` +
      `${describe(spec)}). Size it for any GPU, quantization and context, and get the vLLM flags.`,
  }
}

export default async function ModelPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params
  const id = modelForSlug(slug)
  if (!id) notFound()
  const spec = specFor(id)

  // Pre-computed at build time so the page has real content before any JS runs.
  const cards = (['h100-sxm-80', 'a100-sxm-80', 'l40s-48', 'rtx4090-24'] as const).map((gpu) => {
    const r = size({ model: id, gpu, engine: 'vllm', context: 8192, concurrency: 1 })
    const q = size({ model: id, gpu, engine: 'vllm', context: 8192, concurrency: 1, quant: 'awq-int4', kvDtype: 'fp8' })
    return { gpu, name: r.plan.input.gpu.name, bf16: r.plan, int4: q.plan }
  })

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 md:px-8">
      <nav className="mb-6 text-xs text-muted-foreground">
        <Link href="/" className="hover:text-foreground">llmsize</Link> / {id}
      </nav>

      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">{id} — VRAM and serving config</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          {count(spec.numLayers)} layers · {describe(spec)} · vocab {spec.vocabSize.toLocaleString()} ·
          native context {spec.maxPositionEmbeddings.toLocaleString()}
        </p>
      </header>

      <section className="mb-8 overflow-x-auto">
        <table className="w-full min-w-[38rem] text-sm">
          <thead className="text-left text-xs text-muted-foreground">
            <tr>
              <th className="pb-2 font-medium">GPU</th>
              <th className="pb-2 font-medium">bf16 weights/GPU</th>
              <th className="pb-2 font-medium">Fits on 1 (8k ctx)</th>
              <th className="pb-2 font-medium">int4 + fp8 KV</th>
              <th className="pb-2 font-medium">Fits on 1</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {cards.map((c) => (
              <tr key={c.gpu} className="border-t">
                <td className="py-2 font-sans">{c.name}</td>
                <td className="py-2">{gib(c.bf16.weightBytesPerDevice)}</td>
                <td className="py-2">{c.bf16.fits ? 'yes' : `no (needs ${c.bf16.autofix?.maxModelLen ?? 0} ctx)`}</td>
                <td className="py-2">{gib(c.int4.weightBytesPerDevice)}</td>
                <td className="py-2">{c.int4.fits ? 'yes' : `no (needs ${c.int4.autofix?.maxModelLen ?? 0} ctx)`}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-muted-foreground">
          Predicted, not measured. Weight bytes are read from the checkpoint&apos;s own file sizes where the
          quantization matches; everything else is computed. See docs/MATH.md.
        </p>
      </section>

      <Sizer initialModel={id} />
    </main>
  )
}
