"""Label identity and provenance, independent of DICOM serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydicom.sr.coding import Code

from nifti2dicom.errors import LabelError


@dataclass(frozen=True)
class Segment:
    value: int
    number: int
    name: str
    category: Code
    property_type: Code


@dataclass(frozen=True)
class SegmentationInfo:
    segments: tuple[Segment, ...]
    algorithm_type: str
    algorithm_name: str | None
    algorithm_version: str | None
    warnings: tuple[str, ...]

    @property
    def mapping(self) -> dict[int, int]:
        return {segment.value: segment.number for segment in self.segments}


def read_labels(raw: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (str, Path)):
        try:
            with open(raw, encoding="utf-8") as stream:
                raw = json.load(stream)
        except (OSError, ValueError) as exc:
            raise LabelError(
                "Could not read the label descriptions.",
                hint="Supply a readable JSON file containing label numbers and names.",
            ) from exc
    if not isinstance(raw, dict):
        raise LabelError("Label descriptions must be a JSON object, not a list or a single value.")
    return raw


def _code(raw: Any, fallback: Code) -> Code:
    if raw is None:
        return fallback
    if not isinstance(raw, dict):
        raise LabelError("A coded label description must contain value, scheme and meaning.")
    if not all(
        isinstance(raw.get(key), str) and raw[key].strip() for key in ("value", "scheme", "meaning")
    ):
        raise LabelError("Coded values, schemes and meanings must be nonempty strings.")
    try:
        code = Code(str(raw["value"]), str(raw["scheme"]), str(raw["meaning"]))
    except KeyError as exc:
        raise LabelError("A coded label description needs value, scheme and meaning.") from exc
    if not all((code.value, code.scheme_designator, code.meaning)) or len(code.meaning) > 64:
        raise LabelError(
            "Code values and schemes must be nonempty; meanings must be 1–64 characters."
        )
    return code


def prepare_segments(
    data: np.ndarray,
    labels: dict[str, Any],
    *,
    algorithm_type: str | None,
    algorithm_name: str | None,
    algorithm_version: str | None,
) -> SegmentationInfo:
    if not np.isfinite(data).all() or np.any(data < 0) or np.any(data != np.floor(data)):
        raise LabelError(
            "A label mask must contain finite, nonnegative whole numbers.",
            hint=(
                "Use 0 for background and a positive integer for each segment. "
                "Probability maps need explicit thresholding before this conversion."
            ),
        )
    present = [int(value) for value in np.unique(data) if value != 0]
    if not present:
        raise LabelError(
            "No foreground found: this segmentation contains only background.",
            hint="Check the mask or threshold. No DICOM SEG was created.",
        )
    if len(present) > 65535:
        raise LabelError("This mask contains more than 65,535 segments.")

    algorithm = labels.get("algorithm", {})
    if not isinstance(algorithm, dict):
        raise LabelError("The algorithm description must contain type, name and version.")
    algorithm_type = algorithm_type or algorithm.get("type")
    algorithm_name = algorithm_name or algorithm.get("name")
    algorithm_version = algorithm_version or algorithm.get("version")
    if not isinstance(algorithm_type, str):
        raise LabelError(
            "Please describe how this segmentation was created.",
            hint=(
                "Set --algorithm-type manual, automatic or semiautomatic. "
                "For model output, also supply --algorithm-name and --algorithm-version."
            ),
        )
    algorithm_type = algorithm_type.upper().replace("-", "").replace("_", "")
    if algorithm_type not in {"MANUAL", "AUTOMATIC", "SEMIAUTOMATIC"}:
        raise LabelError("Algorithm type must be manual, automatic or semiautomatic.")
    if algorithm_type != "MANUAL" and not all(
        isinstance(v, str) and v.strip() for v in (algorithm_name, algorithm_version)
    ):
        raise LabelError(
            "Automatic and semiautomatic segmentations need an algorithm name and version.",
            hint="Supply the model or tool that produced the mask, not the conversion tool.",
        )

    raw_mapping = labels.get("organ_indices", labels.get("labels", labels))
    if not isinstance(raw_mapping, dict):
        raise LabelError("The label mapping must be a JSON object.")
    mapping: dict[int, Any] = {}
    for key, value in raw_mapping.items():
        if key == "algorithm":
            continue
        try:
            number = int(key)
        except (TypeError, ValueError) as exc:
            raise LabelError(
                f"Invalid label number '{key}'.", hint="Label keys must be nonnegative integers."
            ) from exc
        if number < 0 or (isinstance(key, float) and key != number) or number in mapping:
            raise LabelError(f"Invalid or duplicate label number '{key}'.")
        if number != 0:
            mapping[number] = value
    missing = set(present) - set(mapping)
    if mapping and missing:
        raise LabelError(
            f"Missing descriptions for label values: {sorted(missing)}.",
            hint="Add these values to the labels JSON file.",
        )

    warnings = []
    if set(mapping) - set(present):
        warnings.append("Descriptions for labels absent from the mask were omitted.")
    unspecified = Code("REGION", "99NIFTI2DICOM", "Unspecified region")
    segments = []
    for number, value in enumerate(present, start=1):
        raw = mapping.get(value, {"name": f"Segment {value}"})
        if isinstance(raw, str):
            raw = {"name": raw}
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("name"), str)
            or not raw["name"].strip()
        ):
            raise LabelError(f"Label {value} needs a nonempty name.")
        if len(raw["name"]) > 64:
            raise LabelError(f"The name for label {value} exceeds 64 characters.")
        property_raw = raw.get("type")
        if "SNOMED" in raw:
            snomed = raw["SNOMED"]
            if not isinstance(snomed, dict) or not snomed.get("ID") or not snomed.get("name"):
                raise LabelError(f"Label {value} has incomplete SNOMED metadata.")
            property_raw = {"value": str(snomed["ID"]), "scheme": "SCT", "meaning": snomed["name"]}
        category = _code(raw.get("category"), unspecified)
        property_type = _code(property_raw, unspecified)
        if category == unspecified or property_type == unspecified:
            warnings.append(
                f"Label {value}: unspecified coded meaning; "
                "supply category/type codes for semantic interoperability."
            )
        segments.append(Segment(value, number, raw["name"], category, property_type))
    return SegmentationInfo(
        tuple(segments), algorithm_type, algorithm_name, algorithm_version, tuple(warnings)
    )
