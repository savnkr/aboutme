---
# HOW TO USE: copy this file to src/content/papers/<short-name>.md and fill it in.
# Lines starting with # are comments. Delete optional fields you don't need.

title: "Full paper title"
authors: ["Sawan Kumar", "Co-author One", "Souvik Chakraborty"]   # your name is bolded automatically
venue: "NeurIPS"                 # short venue name, e.g. ICLR, CMAME, arXiv
year: 2027
summary: "One or two sentences on what the paper does. Shown on the featured card."
theme: machines                  # functions | systems | machines  (filters on the Research page)
order: 1                         # 1 = top of the list; renumber the others if needed
featured: false                  # true = the big green card on the home page (keep only one)
draft: true                      # true = hidden from the site; set false when it's public

# Optional, but they make a big difference to readers and recruiters:
tldr: "The one-line takeaway, in plain words a non-specialist could repeat."
thumbnail: /papers/<short-name>.png   # put the image (or .mp4) in public/papers/
paper: https://arxiv.org/abs/XXXX.XXXXX
pdf: https://arxiv.org/pdf/XXXX.XXXXX
code: https://github.com/savnkr/<repo>
poster: /papers/<short-name>-poster.pdf
video: https://youtube.com/...
project: /aboutme/projects/<project-name>/    # link to a project page on this site
bibtex: |
  @inproceedings{kumar2027short,
    title={Full paper title},
    author={Kumar, Sawan and ...},
    booktitle={...},
    year={2027}
  }
---
