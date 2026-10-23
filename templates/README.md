# Adding papers and projects

Each paper or project is a single Markdown file.

| To add a… | Copy | To | Media goes in |
|---|---|---|---|
| Paper | `templates/paper.md` | `src/content/papers/<name>.md` | `public/papers/` |
| Project | `templates/project.md` | `src/content/projects/<name>.md` | `public/projects/<name>/` |

- Keep `draft: true` while you're writing. The entry stays off the site until you set `draft: false`.
- Run `npm run dev` and open http://localhost:4321/aboutme/ to preview.
- Home-page text you'll edit most often lives in `src/lib/site.ts`:
  - `news`: dated updates
  - `story`: the four scroll chapters
  - `values`: "What I believe"
  - `site.status`: the "open to roles" badge
