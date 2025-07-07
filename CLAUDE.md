# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install dependencies and sync environment
rye sync

# Run the application locally
rye run streamlit run main.py

# Run linter (ruff is installed as dev dependency)
rye run ruff check src/

# Format code with ruff
rye run ruff format src/
```

## Architecture Overview

This is a **Streamlit-based web application** that converts images to pixel art. The core conversion algorithm is implemented in Rust (`pixelart-modules`) for performance, while the web interface and image processing pipeline are in Python.

### Key Components

1. **main.py**: Entry point that initializes the Streamlit app with page configuration
2. **src/draw.py**: Main UI component that orchestrates the entire conversion pipeline
3. **src/convert.py**: Wrapper around the Rust-based `pixelart-modules` for image conversion
4. **src/filters.py**: Image preprocessing filters (DoG, morphology, Kuwahara, median)
5. **src/ai.py**: KMeans-based AI palette generation
6. **pages/**: Additional Streamlit pages for color samples and documentation

### Color Palette System

- Predefined palettes stored as CSV files in `color/` directory (RGB values)
- AI palette generation using KMeans clustering on input image
- Custom palette creation with hex color input support
- Palettes: pyxel, cold, gold, pale, pastel, rainbow, warm

### Image Processing Pipeline

1. **Input**: Accepts jpg, jpeg, png, webp, jfif formats
2. **Preprocessing**: Optional filters (DoG, morphology, etc.)
3. **Enhancement**: Saturation, brightness, contrast, sharpness adjustments
4. **Conversion**: Mosaic/pixelation with color palette mapping (via Rust module)
5. **Post-processing**: Alpha channel handling, transparency options

### Performance Considerations

- Images exceeding FullHD resolution (2,073,600 pixels) are automatically resized
- Core conversion uses Rust for speed
- Garbage collection is explicitly called after processing

### External Dependencies

- **pixelart-modules**: Rust library for core conversion (https://github.com/akazdayo/pixelart-modules)
- This is installed via pip and provides the `convert_pixelart` function

### Testing

Currently no automated tests are implemented (test/ directory is empty).