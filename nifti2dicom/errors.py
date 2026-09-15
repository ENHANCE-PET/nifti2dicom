"""Errors shared by the library, CLI and machine-readable reports."""

from __future__ import annotations

from typing import Any


class ConversionError(Exception):
    """An actionable conversion problem with a stable machine-readable code."""

    code = "conversion_error"

    def __init__(self, message: str, *, hint: str = "", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


class InputError(ConversionError):
    code = "invalid_input"


class GeometryError(ConversionError):
    code = "invalid_geometry"


class ReferenceError(ConversionError):
    code = "invalid_reference"


class AmbiguousReferenceError(ReferenceError):
    code = "ambiguous_reference"


class PixelEncodingError(ConversionError):
    code = "pixel_encoding"


class OutputError(ConversionError):
    code = "output_error"


class UnsupportedInputError(InputError):
    code = "unsupported_input"


class LabelError(InputError):
    code = "invalid_labels"
