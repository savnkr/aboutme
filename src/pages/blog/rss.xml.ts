import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = (await getCollection('blog', ({ data }) => !data.draft)).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: 'Sawan Kumar · blog',
    description: 'Notes on kernels, Gaussian processes, and probability.',
    site: context.site ?? 'https://savnkr.github.io',
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description,
      link: `/aboutme/blog/${post.id}/`,
    })),
  });
}
