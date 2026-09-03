import type { MetadataRoute } from 'next'
import { allSlugs } from '@/lib/models'

const site = 'https://varadmore.me/llm-calculator'
export const dynamic = 'force-static'

export default function sitemap(): MetadataRoute.Sitemap {
  return ['', 'fits', 'explained', ...allSlugs()].map((path) => ({
    url: `${site}/${path}${path ? '/' : ''}`,
    changeFrequency: 'monthly',
  }))
}
