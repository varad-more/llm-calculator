import Link from 'next/link'
import { Sizer } from '@/components/sizer'
import { allSlugs, modelForSlug } from '@/lib/models'
import { allValidations } from '@llmsize/core'

export const metadata = {
  title: { absolute: 'llmsize — LLM inference sizing and serving-config generator' },
  description:
    'Predict engine-accurate memory allocation and throughput for vLLM, SGLang, TensorRT-LLM and llama.cpp, then emit the exact flags to run it.',
  alternates: { canonical: '/' },
}

export default function Home() {
  const validated = allValidations().length
  return (
    <main className="mx-auto max-w-7xl px-4 py-8 md:px-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">llmsize</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          Inference sizing that models what the serving engine actually allocates — per-layer KV
          dispatch for GQA, MLA and sliding-window models, paged block rounding, CUDA-graph and
          activation overhead — and emits the flags to run the result.
        </p>
        <p className="mt-2 text-xs text-muted-foreground">
          {validated === 0 ? (
            <>
              <span className="rounded bg-amber-500/10 px-1.5 py-0.5 font-medium text-amber-600 dark:text-amber-400">
                predicted, not measured
              </span>{' '}
              No engine startup logs have been contributed yet, so every number here is a
              prediction. If you have a GPU, a pasted startup log is worth more than any argument.
            </>
          ) : (
            <>{validated} validated (model, GPU, engine) triples — see docs/VALIDATION.md for measured error.</>
          )}
        </p>
      </header>

      <Sizer />

      <section className="mt-12 border-t pt-6">
        <h2 className="text-sm font-medium">Per-model pages</h2>
        <div className="mt-3 flex flex-wrap gap-2">
          {allSlugs().map((slug) => (
            <Link
              key={slug} href={`/${slug}/`}
              className="rounded-md border px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {modelForSlug(slug)}
            </Link>
          ))}
        </div>
      </section>
    </main>
  )
}
