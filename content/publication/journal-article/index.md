---
title: "Towards Gaussian Process for operator learning: An uncertainty aware resolution independent operator learning algorithm for computational mechanics"
authors:
- Sawan Kumar
- Rajdip Nayek
- Souvik Chakraborty
# author_notes:
# - "Equal contribution"
# - "Equal contribution"
date: "2024-12-28T00:00:00Z"
# doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2025-02-05T00:00:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["article-journal"]

# Publication name and optional abbreviated publication name.
publication: "*Computer Methods in Applied Mechanics and Engineering, 1*(1)"
publication_short: ""

abstract: The growing demand for accurate, efficient, and scalable solutions in computational mechanics highlights the need for advanced operator learning algorithms that can efficiently handle large datasets while providing reliable uncertainty quantification. This paper introduces a novel Gaussian Process (GP) based neural operator for solving parametric differential equations. The approach proposed leverages the expressive capability of deterministic neural operators and the uncertainty awareness of conventional GP. In particular, we propose a “neural operator-embedded kernel” wherein the GP kernel is formulated in the latent space learned using a neural operator. Further, we exploit stochastic dual descent (SDD) algorithm for simultaneously training the neural operator parameters and the GP hyperparameters. Our approach addresses the (a) resolution dependence and (b) cubic complexity of traditional GP models, allowing for input-resolution independence and scalability in high-dimensional and non-linear parametric systems, such as those encountered in computational mechanics. We apply our method to a range of non-linear parametric partial differential equations (PDEs) and demonstrate its superiority in both computational efficiency and accuracy compared to standard GP models and wavelet neural operators. Our experimental results highlight the efficacy of this framework in solving complex PDEs while maintaining robustness in uncertainty estimation, positioning it as a scalable and reliable operator-learning algorithm for computational mechanics.

# Summary. An optional shortened abstract.
summary: This paper presents a GP-based neural operator for parametric differential equations, achieving high accuracy, scalability, and robust uncertainty estimation, outperforming traditional models.

tags:
- Source Themes
featured: false

links:
- name: "CMAME"
  url: ""
url_pdf: 'https://doi.org/10.1016/j.cma.2024.117664'
url_code: 'https://github.com/HugoBlox/hugo-blox-builder'
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image: 
  # caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/jdD8gXaTZsc)'
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: example
draft: true
---

{{% callout note %}}
Click the *Cite* button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the *Slides* button to check out the example.
{{% /callout %}}

<!-- Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images](https://docs.hugoblox.com/content/writing-markdown-latex/). -->
