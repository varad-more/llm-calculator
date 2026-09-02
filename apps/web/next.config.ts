import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // Static export. Next earns its place here only because of the per-model routes
  // generated below (/llama-3-1-70b-instruct-vram) — pure search surface, no server.
  output: 'export',
  images: { unoptimized: true },
  transpilePackages: ['@llmsize/core'],
  trailingSlash: true,
}

export default nextConfig
