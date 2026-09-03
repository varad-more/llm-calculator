import type { MetadataRoute } from 'next'

const site = 'https://varad-more.github.io/llm-calculator'
export const dynamic = 'force-static'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: '*', allow: '/' },
    sitemap: `${site}/sitemap.xml`,
  }
}
