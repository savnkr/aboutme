# Life Section - Quick Guide

This is your creative space for sketches, writings, and personal expressions!

## 📁 Structure

```
content/life/
├── _index.md          # Main Life page
├── sketches/
│   └── index.md       # Sketches gallery page
└── writings/
    └── index.md       # Writings collection page
```

## 🎨 Adding Sketches

### Option 1: Using a gallery (recommended)
1. Create a folder: `content/life/sketches/`
2. Add your sketch images to that folder (e.g., `sketch1.jpg`, `sketch2.png`)
3. The images will automatically appear in a grid

### Option 2: Adding individual sketches
Create new folders for each sketch collection:

```
content/life/sketches/
├── nature-sketches/
│   ├── index.md
│   ├── tree.jpg
│   └── mountain.jpg
└── portrait-sketches/
    ├── index.md
    └── portrait1.jpg
```

In each `index.md`:
```markdown
---
title: "Nature Sketches"
date: 2025-11-10
summary: "My nature-inspired drawings"
tags: ['sketches', 'nature']
---

![Tree Sketch](tree.jpg)
*Caption for your sketch*

![Mountain Sketch](mountain.jpg)
*Another beautiful caption*
```

## ✍️ Adding Writings

Create new markdown files in `content/life/writings/`:

```
content/life/writings/
├── my-first-poem/
│   └── index.md
└── short-story/
    └── index.md
```

Example `index.md`:
```markdown
---
title: "My First Poem"
date: 2025-11-10
summary: "A poem about life"
tags: ['poetry', 'creative writing']
---

## Title of Your Poem

*Your beautiful poem goes here...*

---

Reflections on this piece...
```

## 🎭 Adding Mixed Content

You can create entries that combine both sketches and writings:

```
content/life/journey-2025/
├── index.md
├── sketch1.jpg
└── sketch2.jpg
```

```markdown
---
title: "My 2025 Journey"
date: 2025-11-10
summary: "Sketches and thoughts from this year"
---

## Reflections

Some writing here...

![Sketch](sketch1.jpg)

More thoughts...

![Another sketch](sketch2.jpg)
```

## 🎨 Using the Aesthetic Styles

The custom CSS includes special classes you can use:

```markdown
<div class="writing-piece">

Your writing here with beautiful styling!

</div>

<hr class="life-divider">

<blockquote class="life-quote">
A beautiful quote or important thought
</blockquote>
```

## 📸 Image Best Practices

- **Format**: JPG or PNG
- **Size**: Optimize images (max 2MB recommended)
- **Dimensions**: 1200x800px works well for sketches
- **File names**: Use descriptive names without spaces (e.g., `sunset-sketch.jpg`)

## 🚀 Quick Start

1. Upload your sketches to `content/life/sketches/`
2. Add descriptions in `content/life/sketches/index.md`
3. View your Life section at: `yoursite.com/life/`

Happy creating! 🎨✨
