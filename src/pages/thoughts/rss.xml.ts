import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import type { APIContext } from 'astro';

export async function GET(context: APIContext) {
  const posts = (await getCollection('thoughts', ({ data }) => !data.draft)).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: 'Sawan Kumar · thoughts',
    description: 'Notes on making, reading, and ordinary life.',
    site: context.site ?? 'https://savnkr.github.io',
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description,
      link: `/aboutme/thoughts/${post.id}/`,
    })),
  });
}
