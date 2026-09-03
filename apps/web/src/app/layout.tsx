import type { Metadata } from 'next'
import Link from 'next/link'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] })
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] })
const description =
  'Predict engine-aware GPU memory and throughput for vLLM, SGLang, TensorRT-LLM and ' +
  'llama.cpp, then generate the exact serving flags.'

export const metadata: Metadata = {
  metadataBase: new URL('https://varad-more.github.io/llm-calculator/'),
  title: {
    default: 'llmsize — LLM inference sizing and serving-config generator',
    template: '%s — llmsize',
  },
  description,
  applicationName: 'llmsize',
  keywords: ['LLM inference', 'GPU memory', 'VRAM calculator', 'vLLM', 'SGLang', 'TensorRT-LLM', 'llama.cpp'],
  authors: [{ name: 'Varad More', url: 'https://github.com/varad-more' }],
  creator: 'Varad More',
  openGraph: {
    type: 'website',
    siteName: 'llmsize',
    title: 'llmsize — LLM inference sizing and serving-config generator',
    description,
  },
  twitter: {
    card: 'summary',
    title: 'llmsize — LLM inference sizing and serving-config generator',
    description,
  },
  robots: { index: true, follow: true },
}

export default function RootLayout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 md:flex-nowrap md:gap-6 md:px-8">
            <Link href="/" className="text-sm font-semibold tracking-tight">llmsize</Link>
            <nav className="order-last flex w-full items-center gap-4 overflow-x-auto text-xs text-muted-foreground md:order-none md:w-auto">
              <Link href="/" className="hover:text-foreground">Size a config</Link>
              <Link href="/fits/" className="hover:text-foreground">What fits my GPU</Link>
              <Link href="/explained/" className="hover:text-foreground">How it works</Link>
            </nav>
            <a
              href="https://github.com/varad-more/llm-calculator"
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
            >
              source
            </a>
          </div>
        </header>
        <div className="flex-1">{children}</div>
        <footer className="mt-16 border-t">
          <div className="mx-auto max-w-7xl px-4 py-10 md:px-8">
            <div className="grid gap-8 md:grid-cols-[1.6fr_1fr_1fr]">
              <div>
                <p className="text-sm font-semibold tracking-tight">llmsize</p>
                <p className="mt-2 max-w-md text-xs leading-relaxed text-muted-foreground">
                  Inference sizing and serving-config generation for LLMs. Engine-accurate memory
                  allocation, roofline throughput, and the exact flags to run it — computed by pure
                  functions with zero runtime dependencies, in your browser.
                </p>
              </div>

              <nav className="flex flex-col gap-2 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Pages</span>
                <Link href="/" className="hover:text-foreground">Size a config</Link>
                <Link href="/fits/" className="hover:text-foreground">What fits my GPU</Link>
                <Link href="/explained/" className="hover:text-foreground">How it works</Link>
              </nav>

              <div className="flex flex-col gap-2 text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Source</span>
                <a href="https://github.com/varad-more/llm-calculator" className="hover:text-foreground">
                  github.com/varad-more/llm-calculator
                </a>
                <span className="font-mono">docs/MATH.md</span>
                <span className="font-mono">data/assumptions.json</span>
              </div>
            </div>

            <div className="mt-8 flex flex-col gap-3 border-t pt-6 text-xs text-muted-foreground md:flex-row md:items-center">
              <p>
                Built by{' '}
                <a
                  href="https://github.com/varad-more"
                  className="font-medium text-foreground underline underline-offset-2"
                >
                  Varad More
                </a>
                . Apache-2.0. Contributions — especially{' '}
                <Link href="/explained/#assumptions" className="underline underline-offset-2 hover:text-foreground">
                  real engine logs
                </Link>{' '}
                — welcome.
              </p>
              <p className="md:ml-auto">
                Every number here is <span className="font-medium">predicted</span> until an engine log
                says otherwise. Inference only.
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  )
}
