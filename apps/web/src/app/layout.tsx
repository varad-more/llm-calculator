import type { Metadata } from 'next'
import Link from 'next/link'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] })
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] })

export const metadata: Metadata = {
  title: {
    default: 'llmsize — LLM inference sizing and serving-config generator',
    template: '%s — llmsize',
  },
  description:
    'Predict engine-accurate memory allocation and throughput for vLLM, SGLang, TensorRT-LLM and ' +
    'llama.cpp, then emit the exact flags to run it.',
}

export default function RootLayout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur">
          <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3 md:px-8">
            <Link href="/" className="text-sm font-semibold tracking-tight">llmsize</Link>
            <nav className="flex items-center gap-4 text-xs text-muted-foreground">
              <Link href="/" className="hover:text-foreground">Size a config</Link>
              <Link href="/fits/" className="hover:text-foreground">What fits my GPU</Link>
              <Link href="/explained/" className="hover:text-foreground">How it works</Link>
            </nav>
            <a
              href="https://github.com/varad-more/llmsize"
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
            >
              source
            </a>
          </div>
        </header>
        <div className="flex-1">{children}</div>
        <footer className="mt-16 border-t">
          <div className="mx-auto max-w-7xl px-4 py-6 text-xs text-muted-foreground md:px-8">
            Every formula is in <span className="font-mono">docs/MATH.md</span> with a derivation and a
            citation; every empirical constant is in <span className="font-mono">data/assumptions.json</span>{' '}
            with a source and an honest confidence label. Inference only — training and optimizer memory
            are out of scope.
          </div>
        </footer>
      </body>
    </html>
  )
}
