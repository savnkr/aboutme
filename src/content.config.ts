import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const papers = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/papers' }),
  schema: z.object({
    title: z.string(),
    authors: z.array(z.string()),
    equal: z.array(z.string()).default([]),
    venue: z.string(),
    year: z.number(),
    summary: z.string(),
    theme: z.enum(['functions', 'systems', 'machines']),
    featured: z.boolean().default(false),
    order: z.number(),
    paper: z.string().optional(),
    pdf: z.string().optional(),
    code: z.string().optional(),
    poster: z.string().optional(),
    bibtex: z.string().optional(),
    tldr: z.string().optional(),
    thumbnail: z.string(),
    video: z.string().optional(),
    project: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    status: z.string(),
    year: z.number(),
    tags: z.array(z.string()),
    figure: z.enum(['flow', 'robot', 'sphere']).optional(),
    teaser: z.boolean().default(false),
    order: z.number(),
    impact: z.string().optional(),
    role: z.string().optional(),
    stack: z.array(z.string()).default([]),
    cover: z.string().optional(),
    video: z.string().optional(),
    repo: z.string().optional(),
    demo: z.string().optional(),
    paper: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

const post = z.object({
  title: z.string(),
  description: z.string(),
  date: z.coerce.date(),
  updated: z.coerce.date(),
  tags: z.array(z.string()).default([]),
  cover: z.string(),
  draft: z.boolean().default(false),
  interactive: z.enum(['gp']).optional(),
});

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: post,
});

const thoughts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/thoughts' }),
  schema: post,
});

export const collections = { papers, projects, blog, thoughts };
