"""Headless image-processing pipeline (no Streamlit dependency)."""

from __future__ import annotations

import base64
import gc
from dataclasses import dataclass, field
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .ai import AI
from .convert import Convert
from .filters import Dithering, EdgeFilter, GridMask, ImageEnhancer

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def cv_to_base64(img: NDArray[np.uint8]) -> str:
    """Encode an OpenCV image (BGR/BGRA) to a PNG base64 string."""
    _, encoded = cv2.imencode(".png", img)
    return base64.b64encode(encoded).decode("ascii")


def _hex_to_rgb(hex_str: str) -> list[int]:
    """Convert a ``#rrggbb`` hex string to an ``[r, g, b]`` list."""
    hex_str = hex_str.lstrip("#")
    return [int(hex_str[i : i + 2], 16) for i in (0, 2, 4)]


# ---------------------------------------------------------------------------
# Processing configuration
# ---------------------------------------------------------------------------

MAX_PIXELS = 2_073_600  # FullHD


@dataclass
class ProcessingOptions:
    """All knobs exposed by the UI, collected into a plain data object.

    Every field has a sensible default so callers only need to override
    what they care about.
    """

    # colour-palette ---------------------------------------------------------
    palette: str = "pyxel"
    """Palette name (e.g. ``"pyxel"``, ``"cold"``) or ``"Custom"`` / ``"AI"``."""

    custom_palette: list[list[int]] = field(default_factory=list)
    """RGB rows when *palette* is ``"Custom"`` or ``"AI"``."""

    no_convert: bool = False
    """Skip colour-palette mapping entirely."""

    # AI palette -------------------------------------------------------------
    ai_colors: int = 8
    ai_iterations: int = 150

    # mosaic / pixelation ----------------------------------------------------
    mosaic_mode: str = "slider"
    """``"slider"`` or ``"grid"``."""

    slider_ratio: float = 0.3
    grid_size: int = 256

    no_expand: bool = False
    """If *True*, do **not** resize back to original dimensions after mosaic."""

    # pre-processing filters -------------------------------------------------
    dog_filter: bool = False
    scratch: bool = False
    morphology: bool = False
    kuwahara: bool = False
    median: bool = False
    delete_transparent: bool = False

    # enhancements -----------------------------------------------------------
    saturation: float = 1.0
    brightness: float = 1.0
    contrast: float = 1.0
    sharpness: float = 1.0

    # dithering --------------------------------------------------------------
    dithering: bool = False
    dithering_method: str = "Floyd-Steinberg"
    dither_matrix_size: int = 4
    dither_intensity: float = 1.0

    # post-processing --------------------------------------------------------
    decrease_color: bool = False
    delete_alpha: bool = False

    # grid overlay -----------------------------------------------------------
    enable_grid: bool = False
    grid_line_color: tuple[int, int, int] = (0, 0, 0)
    grid_line_thickness: int = 1
    grid_opacity: float = 0.5


# ---------------------------------------------------------------------------
# Processing result
# ---------------------------------------------------------------------------


@dataclass
class ProcessingResult:
    """Return value of :func:`process_image`."""

    image: NDArray[np.uint8]
    """Final processed image (RGB or RGBA)."""

    was_resized: bool
    """Whether the input was down-scaled because it exceeded *MAX_PIXELS*."""

    ai_hex_colors: list[str] | None = None
    """Hex colour strings produced by the AI palette (if used)."""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_image(
    img: NDArray[np.uint8],
    options: ProcessingOptions | None = None,
) -> ProcessingResult:
    """Run the full pixel-art conversion pipeline on *img*.

    Parameters
    ----------
    img:
        Input image in RGB or RGBA format.
    options:
        Processing knobs. ``None`` means use all defaults.

    Returns
    -------
    ProcessingResult
        Contains the processed image, a resize flag, and optional AI colours.
    """
    if options is None:
        options = ProcessingOptions()

    conv = Convert()
    edges = EdgeFilter()
    enhance = ImageEnhancer()
    grid_mask_tool = GridMask()

    # --- greyscale guard ---------------------------------------------------
    if img.ndim == 2:
        img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    # --- resize if too large -----------------------------------------------
    height, width = img.shape[:2]
    was_resized = height * width > MAX_PIXELS
    if was_resized:
        img = conv.resize_image(img)

    cimg: NDArray[Any] = img.copy()

    # --- enhancements ------------------------------------------------------
    if options.saturation != 1.0:
        cimg = enhance.saturation(cimg, options.saturation)
    if options.brightness != 1.0:
        cimg = enhance.brightness(cimg, options.brightness)
    if options.contrast != 1.0:
        cimg = enhance.contrast(cimg, options.contrast)
    if options.sharpness != 1.0:
        cimg = enhance.sharpness(cimg, options.sharpness)

    # --- pre-processing filters --------------------------------------------
    if options.delete_transparent:
        cimg = conv.delete_transparent_color(cimg)
    if options.scratch:
        cimg = edges.dog(cimg, True)
    if options.median:
        cimg = edges.median(cimg, 15)
    if options.kuwahara:
        cimg = edges.apply_kuwahara(cimg)
    if options.dog_filter and not options.scratch:
        cimg = edges.dog(cimg)
    if options.morphology and not options.scratch:
        cimg = edges.morphology_erode(cimg)

    # --- mosaic / pixelation -----------------------------------------------
    if options.mosaic_mode == "slider":
        cimg = enhance.slider_mosaic(cimg, options.slider_ratio)
    else:
        cimg = enhance.grid_mosaic(cimg, options.grid_size)

    # --- dithering ---------------------------------------------------------
    if options.dithering:
        dither = Dithering()
        if options.dithering_method == "Floyd-Steinberg":
            cimg = dither.floyd_steinberg(cimg, options.dither_intensity)
        elif options.dithering_method == "Ordered":
            cimg = dither.ordered_dither(
                cimg, options.dither_matrix_size, options.dither_intensity
            )
        elif options.dithering_method == "Atkinson":
            cimg = dither.atkinson(cimg, options.dither_intensity)

    # --- colour conversion -------------------------------------------------
    ai_hex_colors: list[str] | None = None
    if not options.no_convert:
        if options.palette in ("Custom", "AI"):
            if options.palette == "AI":
                ai_tool = AI()
                ai_hex_colors = ai_tool.get_color(
                    cimg, options.ai_colors, options.ai_iterations
                )
                ai_palette = [_hex_to_rgb(h) for h in ai_hex_colors]
                cimg = conv.convert(cimg, "Custom", ai_palette)
            else:
                cimg = conv.convert(cimg, "Custom", options.custom_palette or None)
        else:
            cimg = conv.convert(cimg, options.palette)

    # --- expand back -------------------------------------------------------
    if not options.no_expand:
        cimg = cast(
            NDArray[np.uint8],
            cv2.resize(cimg, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST),
        )

    # --- grid overlay ------------------------------------------------------
    if options.enable_grid:
        if options.mosaic_mode == "grid":
            computed_grid = max(1, cimg.shape[0] // options.grid_size)
        else:
            computed_grid = max(1, int(img.shape[0] * options.slider_ratio))
        cimg = grid_mask_tool.add_grid(
            cimg,
            computed_grid,
            options.grid_line_color,
            options.grid_line_thickness,
            options.grid_opacity,
        )

    # --- post-processing ---------------------------------------------------
    if options.decrease_color:
        cimg = enhance.decrease(cimg)
    if options.delete_alpha:
        cimg = conv.delete_alpha(cimg)

    # --- dtype guard -------------------------------------------------------
    if cimg.dtype != np.uint8:
        cimg = cast(NDArray[np.uint8], cv2.convertScaleAbs(cimg))

    gc.collect()

    return ProcessingResult(
        image=cimg,
        was_resized=was_resized,
        ai_hex_colors=ai_hex_colors,
    )
