---
title: "Neural Operator induced Gaussian Process framework for probabilistic solution of parametric partial differential equations"
authors:
- Sawan Kumar
- Rajdip Nayek
- Souvik Chakraborty
date: "2024-04-29T00:00:00Z"
doi: "10.48550/arXiv.2404.15618"

# Schedule page publish date (NOT publication's date).
publishDate: "2024-04-29T00:00:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["preprint"]

# Publication name and optional abbreviated publication name.
publication: "arXiv"
publication_short: "arXiv"

abstract: The study of neural operators has paved the way for the development of efficient approaches for solving partial differential equations (PDEs) compared with traditional methods. However, most of the existing neural operators lack the capability to provide uncertainty measures for their predictions, a crucial aspect, especially in data-driven scenarios with limited available data. In this work, we propose a novel Neural Operator-induced Gaussian Process (NOGaP), which exploits the probabilistic characteristics of Gaussian Processes (GPs) while leveraging the learning prowess of operator learning. The proposed framework leads to improved prediction accuracy and offers a quantifiable measure of uncertainty. The proposed framework is extensively evaluated through experiments on various PDE examples, including Burger’s equation, Darcy flow, non-homogeneous Poisson, and wave-advection equations. Furthermore, a comparative study with state-of-the-art operator learning algorithms is presented to highlight the advantages of NOGaP. The results demonstrate superior accuracy and expected uncertainty characteristics, suggesting the promising potential of the proposed framework.

# Summary. An optional shortened abstract.
summary: A novel framework, Neural Operator-induced Gaussian Process (NOGaP), is proposed for solving PDEs with improved prediction accuracy and quantifiable uncertainty, evaluated on various examples like Burger’s equation and Darcy flow.

tags:
- Neural Operators
- Gaussian Process
- PDEs
- Uncertainty Quantification

featured: true

links:
- name: arXiv
  url: https://arxiv.org/abs/2404.15618
url_pdf: https://arxiv.org/pdf/2404.15618
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  # caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/s9CC2SKySJM)'
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
- internal-project

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: example
draft: false
---
<!-- 
This work is driven by the results in my [previous paper](/publication/conference-paper/) on LLMs.

{{% callout note %}}
Create your slides in Markdown - click the *Slides* button to check out the example.
{{% /callout %}}

Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images](https://docs.hugoblox.com/content/writing-markdown-latex/). -->
