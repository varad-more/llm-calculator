import type { MetadataRoute } from 'next'

const site = 'https://varadmore.me/llm-calculator'
export const dynamic = 'force-static'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: '*', allow: '/' },
    sitemap: `${site}/sitemap.xml`,
  }
}
