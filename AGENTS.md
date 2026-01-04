# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Build/Lint/Test Commands

### Environment Setup

```bash
# Install dependencies and sync environment (requires rye)
rye sync

# Alternative: pip install
pip install -r requirements.txt
```

### Running the Application

```bash
# Run the Streamlit application
rye run streamlit run main.py

# Alternative without rye
streamlit run main.py
```

### Linting and Formatting

```bash
# Run linter on all source files
rye run ruff check src/ pages/ main.py

# Run linter on a specific file
rye run ruff check src/convert.py

# Auto-fix linting issues
rye run ruff check --fix src/

# Format all code
rye run ruff format src/ pages/ main.py

# Format a specific file
rye run ruff format src/convert.py
```

### Type Checking

```bash
# Run pyright for type checking
rye run pyright src/
```

### Testing

Currently no automated tests are implemented. The `test/` directory is empty.

```bash
# When tests exist, they would be run with:
rye run pytest

# Run a single test file
rye run pytest tests/test_convert.py

# Run a specific test function
rye run pytest tests/test_convert.py::test_resize_image -v
```

## Code Style Guidelines

### Import Order

Organize imports in three groups with no blank lines between groups:

1. Standard library imports
2. Third-party imports
3. Local/project imports

```python
import csv
import os
import gc
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import src.convert as convert
import src.filters as filters
from numpy.typing import NDArray
```

Common aliases used in this project:
- `numpy` as `np`
- `streamlit` as `st`
- `cv2` (OpenCV) - no alias
- `pandas` as `pd`

### Type Hints

Type hints are used sparingly. Apply them in these cases:

- `__init__` methods: `def __init__(self) -> None:`
- Functions returning complex types: `def convert(...) -> NDArray[np.uint64]:`
- Use `typing.cast` when needed for type safety

```python
from numpy.typing import NDArray
from typing import cast

def __init__(self) -> None:
    pass

def convert(self, img, option, custom=None) -> NDArray[np.uint64]:
    changed = cast(NDArray[np.uint64], pm.convert(img, np.array(...)))
    return changed
```

### Naming Conventions

| Element    | Convention     | Examples                                    |
|------------|----------------|---------------------------------------------|
| Functions  | snake_case     | `file_dir()`, `hex_to_rgb()`, `get_image()` |
| Variables  | snake_case     | `color_palette`, `img_array`, `rgb_values`  |
| Classes    | PascalCase     | `Web`, `Convert`, `EdgeFilter`, `AI`        |
| Constants  | snake_case     | `warning_message` (not UPPER_SNAKE_CASE)    |
| Files      | snake_case     | `color_sample.py`, `how_to_use.py`          |

Common abbreviations:
- `img` for image
- `conv` for convert/converter
- `fdir` for file directory

### Formatting

- **Indentation**: 4 spaces
- **Line length**: ~88-100 characters (ruff default)
- **Quotes**: Double quotes preferred (`"string"`)
- **Trailing commas**: Use in multi-line structures

```python
st.set_page_config(
    page_title="Pixelart-Converter",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded",
)
```

### Error Handling

Error handling is minimal in this codebase. Use explicit ValueError for critical errors:

```python
if not custom:
    raise ValueError("Custom Palette is empty.")
```

For image processing, use conditional checks rather than try/except:

```python
# Check for alpha channel
if image.shape[2] == 4:
    # Handle RGBA image
else:
    # Handle RGB image
```

### Docstrings

Use triple double-quotes with Japanese comments where appropriate:

```python
def add_grid(self, image, grid_size, line_color=(0, 0, 0), line_thickness=1, opacity=0.5):
    """
    Add a grid mask to the image

    Args:
        image: Input image (numpy array)
        grid_size: Grid size in pixels
        line_color: Grid line color (B, G, R)
        line_thickness: Line thickness
        opacity: Grid opacity (0.0-1.0)

    Returns:
        Image with grid overlay
    """
```

### Class Structure

Classes are used to group related functions. Initialize with empty `__init__`:

```python
class EdgeFilter:
    def __init__(self) -> None:
        pass

    def canny(self, image, th1, th2):
        # Implementation
        pass
```

## Architecture Notes

### Project Structure

```
pixelart/
├── main.py           # Entry point
├── src/
│   ├── draw.py       # Web UI class (Streamlit interface)
│   ├── run.py        # Main processing pipeline
│   ├── convert.py    # Image conversion (wraps Rust module)
│   ├── filters.py    # Image preprocessing filters
│   └── ai.py         # KMeans color palette generation
├── pages/            # Additional Streamlit pages
├── color/            # CSV color palettes (RGB values)
└── sample/           # Sample images
```

### Key Dependencies

- **pixelart-modules**: Rust library for core conversion (`pip install pixelart-modules`)
- **streamlit**: Web framework for the UI
- **opencv-python-headless**: Image processing (use `cv2`)
- **scikit-learn**: KMeans clustering for AI palette

### Image Processing Pipeline

1. Input validation and resize (if > FullHD resolution)
2. Apply preprocessing filters (DoG, morphology, Kuwahara, median)
3. Apply image enhancements (saturation, brightness, contrast, sharpness)
4. Mosaic/pixelation
5. Color palette mapping via Rust module
6. Post-processing (alpha channel, grid overlay, dithering)

### Color Format Notes

- OpenCV uses BGR format (not RGB)
- Color palettes stored as RGB in CSV files
- Alpha channel handling: Check `image.shape[2] == 4` for RGBA

### Performance Considerations

- Large images (>2,073,600 pixels) are automatically resized
- Call `gc.collect()` after heavy processing
- Core conversion uses Rust for performance

## Common Patterns

### Alpha Channel Handling

```python
# Separate alpha channel before processing
has_alpha = len(image.shape) == 3 and image.shape[2] == 4
if has_alpha:
    bgr = image[:, :, :3].copy()
    alpha = image[:, :, 3]
else:
    bgr = image.copy()

# ... process bgr ...

# Merge alpha channel back
if has_alpha:
    return np.dstack([result, alpha])
else:
    return result
```

### Hex to RGB Conversion

```python
def hex_to_rgb(self, hex_code):
    hex_code = hex_code.replace("#", "")
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return [r, g, b]
```

## Python Version

Requires Python >= 3.12 (specified in pyproject.toml).
