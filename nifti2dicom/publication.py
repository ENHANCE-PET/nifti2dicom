"""Stage validated output and replace an existing directory with rollback."""

from __future__ import annotations

import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

from nifti2dicom.errors import OutputError


def validate_output(target: Path, sources: Sequence[Path], *, overwrite: bool) -> None:
    if target.is_symlink():
        raise OutputError(
            "The output path is a symbolic link.", hint="Choose a new output directory."
        )
    resolved = target.resolve()
    if resolved in {Path(resolved.anchor), Path.home(), Path.cwd().resolve()}:
        raise OutputError(
            "The output must be a dedicated conversion directory.",
            hint="Choose a new subdirectory for the DICOM files.",
        )
    for source in sources:
        source = source.resolve()
        if (
            source == resolved
            or resolved in source.parents
            or (source.is_dir() and source in resolved.parents)
        ):
            raise OutputError(
                "The output directory overlaps an input location.",
                hint="Choose an output directory outside the reference DICOM folder and inputs.",
            )
    if target.exists():
        if not target.is_dir():
            raise OutputError(
                f"Output path is an existing file: {target}", hint="Choose a directory name."
            )
        if not overwrite:
            raise OutputError(
                f"Output directory already exists: {target}",
                hint="Choose a new output directory or use --overwrite to replace it.",
            )


@contextmanager
def staged_output(target: Path, *, overwrite: bool) -> Iterator[Path]:
    """Yield private staging; publish only if the caller completes successfully."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = target.parent / f".{target.name}.nifti2dicom-lock"
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise OutputError(
                f"Another conversion may be writing to {target}.",
                hint=(
                    f"Wait for it to finish. If it stopped, remove the empty lock directory {lock}."
                ),
            ) from exc
        try:
            with TemporaryDirectory(
                prefix=f".{target.name}.staging-", dir=target.parent
            ) as stage_name:
                stage = Path(stage_name)
                yield stage
                if target.exists() and not overwrite:
                    raise OutputError(
                        f"Output directory already exists: {target}",
                        hint="Choose a different output directory.",
                    )
                backup = None
                if target.exists():
                    backup = Path(mkdtemp(prefix=f".{target.name}.backup-", dir=target.parent))
                    previous = backup / "previous"
                    try:
                        target.rename(previous)
                    except OSError:
                        backup.rmdir()
                        raise
                try:
                    stage.rename(target)
                except OSError as exc:
                    if backup is not None:
                        try:
                            previous.rename(target)
                        except OSError as rollback:
                            # Never let automatic temporary cleanup destroy this backup.
                            raise OutputError(
                                "Output publication and automatic restoration both failed.",
                                hint=(
                                    f"Your previous output is preserved at {previous}. "
                                    f"Move it back to {target}."
                                ),
                                details={"backup": str(previous)},
                            ) from rollback
                        backup.rmdir()
                    raise exc
                if backup is not None:
                    # The user requested replacement and the new output is now published.
                    shutil.rmtree(backup)
        finally:
            lock.rmdir()
    except OSError as exc:
        raise OutputError(
            f"Could not write DICOM output to {target}: {exc.strerror or exc}",
            hint="Check directory permissions and available disk space.",
        ) from exc
