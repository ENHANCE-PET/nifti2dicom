"""Small shared contracts. Spatial affines always map XYZ indices to LPS mm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydicom.dataset import Dataset

ImageKind = Literal["image", "seg", "rgb"]


@dataclass(frozen=True)
class Geometry:
    affine: np.ndarray
    size: tuple[int, int, int]

    @property
    def spacing(self) -> tuple[float, float, float]:
        x, y, z = np.linalg.norm(self.affine[:3, :3], axis=0)
        return float(x), float(y), float(z)

    @property
    def iop(self) -> np.ndarray:
        directions = self.affine[:3, :3] / np.asarray(self.spacing)
        return np.concatenate((directions[:, 0], directions[:, 1]))

    @property
    def pixel_spacing(self) -> tuple[float, float]:
        x, y, _ = self.spacing
        return y, x

    def position(self, slice_index: int) -> np.ndarray:
        return np.asarray(self.affine[:3, 3] + slice_index * self.affine[:3, 2])


@dataclass(frozen=True)
class ImageVolume:
    data: np.ndarray  # TZYX for scalar/labels; TZYXC for RGB
    geometry: Geometry
    kind: ImageKind = "image"
    time_spacing: float | None = None  # seconds, only when explicitly known
    warnings: tuple[str, ...] = ()
    source: Path | None = None

    @property
    def timepoints(self) -> int:
        return int(self.data.shape[0])


@dataclass(frozen=True)
class ReferenceSeries:
    slices: tuple[Dataset, ...]  # time-major, then physical slice position
    paths: tuple[Path, ...]
    geometry: Geometry
    timepoints: int = 1
    warnings: tuple[str, ...] = ()

    @property
    def first(self) -> Dataset:
        return self.slices[0]


@dataclass(frozen=True)
class Progress:
    stage: str
    completed: int = 0
    total: int = 0
    message: str = ""


ProgressCallback = Callable[[Progress], None]


@dataclass(frozen=True)
class InspectionResult:
    kind: ImageKind
    shape: tuple[int, ...]
    spacing: tuple[float, float, float]
    timepoints: int
    modality: str
    reference_series_uid: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class ConversionResult:
    output: Path
    files: tuple[Path, ...]
    kind: ImageKind
    warnings: tuple[str, ...] = ()
    label_mapping: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "files": [str(p) for p in self.files],
            "kind": self.kind,
            "warnings": list(self.warnings),
            "label_mapping": self.label_mapping,
        }
