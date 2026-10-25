// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import remarkDirective from 'remark-directive';
import remarkBlocks from './src/lib/remark-blocks.mjs';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://savnkr.github.io',
  base: '/aboutme',
  trailingSlash: 'always',
  prefetch: {
    prefetchAll: false,
    defaultStrategy: 'hover',
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkDirective, remarkBlocks],
      rehypePlugins: [rehypeKatex],
    }),
    shikiConfig: { theme: 'github-light' },
  },
  integrations: [mdx(), sitemap()],
});
