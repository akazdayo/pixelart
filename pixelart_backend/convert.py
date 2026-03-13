import csv
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pixelart_modules as pm
from numpy.typing import NDArray


PALETTE_SUFFIX = ".csv"


class Convert:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _candidate_paths(path: str) -> list[str]:
        normalized = path.strip()
        if not normalized:
            raise FileNotFoundError(path)

        if normalized.startswith("./"):
            normalized = normalized[2:]

        candidates = [normalized]
        if not normalized.startswith("color/"):
            candidates.append("color/" + normalized)

        return list(dict.fromkeys(candidates))

    @staticmethod
    def _csv_stems(base: Any) -> list[str]:
        names: list[str] = []
        for item in base:
            if item.is_file() and item.name.endswith(PALETTE_SUFFIX):
                names.append(item.name.removesuffix(PALETTE_SUFFIX))
        return names

    @staticmethod
    def list_color_palettes() -> tuple[str, ...]:
        names = set[str]()
        root = Path(__file__).resolve().parent.parent

        project_color = root / "color"
        if project_color.is_dir():
            names.update(Convert._csv_stems(project_color.iterdir()))

        package_root = files("pixelart_backend")
        package_color = package_root.joinpath("color")
        if package_color.is_dir():
            names.update(Convert._csv_stems(package_color.iterdir()))

        return tuple(sorted(names))

    @staticmethod
    def _resolve_palette_path(
        path: str,
        _project_root: Path | None = None,
        _package_root: Any | None = None,
    ) -> Any:
        candidates = Convert._candidate_paths(path)

        project_root = _project_root or Path(__file__).resolve().parent.parent
        package_root = _package_root or files("pixelart_backend")

        for candidate in candidates:
            direct_path = Path(candidate)
            if direct_path.is_file():
                return direct_path

            resource_path = package_root.joinpath(candidate)
            if resource_path.is_file():
                return resource_path

            fallback_path = project_root / candidate
            if fallback_path.is_file():
                return fallback_path

        raise FileNotFoundError(path)

    def read_csv(self, path: str) -> list[list[int]]:
        resolved = self._resolve_palette_path(path)
        with cast(Any, resolved).open(encoding="utf-8") as f:
            reader = csv.reader(f)
            color = [[int(v) for v in row] for row in reader]
            return color

    def convert(
        self,
        img: NDArray[np.uint8],
        option: str,
        custom: list[list[int]] | None = None,
    ) -> NDArray[np.uint64]:
        if option != "Custom":
            palette_file = option if option.endswith(PALETTE_SUFFIX) else option + PALETTE_SUFFIX
            color_palette = self.read_csv(palette_file)
        else:
            if not custom:
                raise ValueError("Custom Palette is empty.")
            color_palette = custom

        module_pm = cast(Any, pm)
        changed_raw: Any = module_pm.convert(
            img,
            np.array(color_palette, dtype=np.uint64),
        )
        changed = cast(NDArray[np.uint64], changed_raw)
        return changed

    def resize_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        img_size = image.shape[0] * image.shape[1]
        ratio = (img_size / 2073600) ** 0.5
        new_height = int(image.shape[0] / ratio)
        new_width = int(image.shape[1] / ratio)
        result = cv2.resize(image, (new_width, new_height))
        return cast(NDArray[np.uint8], result)

    def delete_alpha(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.shape[2] == 4:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
            conv_a = np.where(alpha != 0, 255, alpha).astype(np.uint8)
            merged = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], conv_a])
            return cast(NDArray[np.uint8], merged)
        return image

    def delete_transparent_color(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.shape[2] == 4:
            bgr = image[:, :, :3].copy()
            alpha = image[:, :, 3]
            mask = alpha == 0
            bgr[mask] = [255, 255, 255]
            merged = cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])
            return cast(NDArray[np.uint8], merged)
        return image


def list_palette_names() -> tuple[str, ...]:
    return Convert.list_color_palettes()


def load_palette_rows(palette_name: str) -> list[list[int]]:
    palette_path = palette_name
    if not palette_name.endswith(PALETTE_SUFFIX):
        palette_path = f"{palette_name}{PALETTE_SUFFIX}"
    return Convert().read_csv(palette_path)
