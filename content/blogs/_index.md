---
title: Blogs
cms_exclude: true
#url: talk

# View
view: card #article-grid
# cover: "blog_cover"  # Update with your image path
# Optional cover image (relative to `assets/media/` folder).
# featured:
#   image:
#     caption: ''
#     filename: 'static/uploads/blog_cov.jpg'

banner:
  image: 'stacked-peaks.svg'
  caption: 'Image credit: '

sections:
  - block: hero
    design:
      spacing:
        # Customize the section spacing. Order is top, right, bottom, left.
        # padding: ['180px', '0', '180px', '0']
        padding: ['0', '0', '0', '0']
      background:
        # Choose a color such as from https://html-color-codes.info
        color: 'black'
        # Text color (true=light, false=dark, or remove for the dynamic theme color).
        text_color_light: true
        video:
          # Name of video in `assets/media/`.
          filename: hero.mp4
          # Post-processing: flip the video horizontally?
          caption: 'Image credit: [**Louis Tiao**](https://raw.githubusercontent.com/ltiao/tiao.io)'
          flip: false
---
Welcome to my blog where I discuss various topics related to machine learning and AI
