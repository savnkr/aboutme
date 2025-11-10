# Upload Your Images Here

## Directory Structure

```
static/uploads/life/
├── sketches/          # Put your sketch images here
└── other/             # Other creative content images
```

## How to Use

1. **Upload your sketches** to `static/uploads/life/sketches/`
2. **Reference them in your markdown files** using:
   - Relative path: `![Description](sketches/your-sketch.jpg)`
   - Or absolute path: `![Description](/uploads/life/sketches/your-sketch.jpg)`

## Image Guidelines

- **Formats**: JPG, PNG, WebP
- **Naming**: Use descriptive names (e.g., `sunset-sketch-2025.jpg`)
- **Size**: Optimize images (recommended < 2MB each)
- **Dimensions**: 1200px-2000px width is ideal for web viewing

## Example Usage in Content

In your `content/life/sketches/index.md`:

```markdown
![My First Sketch](/uploads/life/sketches/sketch-001.jpg)
*Description of your sketch*
```

Or using Hugo's figure shortcode for more control:

```markdown
{{< figure src="/uploads/life/sketches/sketch-001.jpg" 
    title="My First Sketch" 
    caption="Drawn in 2025" 
    width="600" >}}
```
