---
# HOW TO USE: copy this file to src/content/projects/<short-name>.md and fill it in.
# The page URL becomes /projects/<short-name>/. Lines starting with # are comments.

title: "Project name"
summary: "One sentence: what it is."
impact: "One sentence: why it matters. This is what recruiters read first."
status: "In progress"            # In progress | Done | Paper under review | ...
year: 2027
order: 3                         # position on the Projects page and home page
tags: [robotics, diffusion, uncertainty]
draft: true                      # true = hidden from the site; set false when it's ready

# Media: a looping demo sells a project better than anything else.
# Put files in public/projects/<short-name>/ and reference them like below.
cover: /projects/<short-name>/demo.gif        # card image + page hero (gif, png, webp)
video: /projects/<short-name>/demo.mp4        # optional; replaces the cover on the project page
# figure: robot                               # or a built-in drawing (flow | robot) if you have no media yet

role: "What you did: designed the model, built the sim, ran the experiments."
stack: [PyTorch, JAX, MuJoCo]
repo: https://github.com/savnkr/<repo>
demo: https://<live-demo-url>
paper: https://arxiv.org/abs/XXXX.XXXXX
---

## The problem

Two or three sentences a smart non-expert can follow. What goes wrong today?

## What I built

The idea, a figure or GIF, and the one design decision you're proud of.

![Short description of the image](/aboutme/projects/<short-name>/figure.png)

## Results

The number or the side-by-side that makes the point. Be honest about limits.

## What I learned

One paragraph. This is where your personality shows.
