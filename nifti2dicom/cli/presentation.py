"""Small, per-invocation terminal renderer using LION's coral/greige palette."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

import click
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, TaskID, TextColumn
from rich.progress import Progress as RichProgress
from rich.text import Text

from nifti2dicom.models import ConversionResult, InspectionResult, Progress

CORAL = "#E87461"
GREIGE = "#B5A89A"


class Presentation:
    """Own every CLI message, without a shared or redirected global console."""

    def __init__(self, *, json_output: bool = False, quiet: bool = False, debug: bool = False):
        self.json_output = json_output
        self.quiet = quiet
        self.debug = debug
        self.console = Console(file=sys.stderr, highlight=False)
        self._stages: set[str] = set()
        self._counted: set[str] = set()
        self._active_stage: str | None = None
        self._live: RichProgress | None = None
        self._task_id: TaskID | None = None

    def progress(self, event: Progress) -> None:
        if self.json_output or self.quiet:
            return
        stage = "writing" if event.stage == "encode" else event.stage
        if stage != self._active_stage:
            self._stop_progress()
            self._active_stage = stage
        if stage not in self._stages:
            self._stages.add(stage)
            line = Text(f"{len(self._stages):02d}", style=f"bold {CORAL}")
            line.append(" · ", style=GREIGE)
            line.append(stage.replace("_", " ").upper(), style="bold")
            if event.message:
                line.append(f"  {event.message}", style=GREIGE)
            self.console.print(line)
        if event.total <= 0 or stage == "complete":
            return
        if self.console.is_terminal:
            if self._live is None:
                self._live = RichProgress(
                    TextColumn("  "),
                    BarColumn(complete_style=CORAL, finished_style=CORAL),
                    MofNCompleteColumn(),
                    TextColumn("{task.description}", style=GREIGE, markup=False),
                    console=self.console,
                    transient=True,
                    redirect_stdout=False,
                    redirect_stderr=False,
                )
                self._task_id = self._live.add_task(
                    event.message, total=event.total, completed=event.completed
                )
                self._live.start()
            elif self._task_id is not None:
                self._live.update(
                    self._task_id,
                    total=event.total,
                    completed=event.completed,
                    description=event.message,
                )
        elif event.completed >= event.total and stage not in self._counted:
            self._counted.add(stage)
            self.console.print(Text(f"  {event.completed}/{event.total} · {event.message}"))

    def _stop_progress(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
            self._task_id = None

    def result(self, result: ConversionResult | InspectionResult) -> None:
        self._stop_progress()
        if self.json_output:
            click.echo(json.dumps({"status": "ok", **result.to_dict()}))
            return
        if self.quiet:
            return
        if isinstance(result, ConversionResult):
            count = len(result.files)
            self.console.print(
                Text(
                    f"Wrote {count} DICOM {'file' if count == 1 else 'files'} "
                    f"({result.kind}) to {result.output}",
                    style=f"bold {CORAL}",
                )
            )
        else:
            self.console.print(Text(f"Input kind: {result.kind}", style=f"bold {CORAL}"))
            axes = "time, slice, row, column" + (", channel" if result.kind == "rgb" else "")
            self.console.print(Text(f"Shape ({axes}): {result.shape}"))
            spacing = " × ".join(f"{value:g}" for value in result.spacing)
            self.console.print(Text(f"Spacing: {spacing} mm · Timepoints: {result.timepoints}"))
            self.console.print(
                Text(f"Reference: {result.modality} · Series {result.reference_series_uid}")
            )
        self.warnings(result.warnings)

    def warnings(self, messages: Sequence[str]) -> None:
        if not self.json_output and not self.quiet:
            for message in messages:
                self.console.print(Text(f"Warning: {message}", style="yellow"))

    def error(self, error: dict[str, Any]) -> None:
        self._stop_progress()
        if self.json_output:
            click.echo(json.dumps({"status": "error", "error": error}))
            return
        self.console.print(Text(f"Error: {error['message']}", style="bold red"))
        if error.get("hint"):
            self.console.print(Text(f"Hint: {error['hint']}", style=GREIGE))
        series = error.get("details", {}).get("series_uids", [])
        if series:
            self.console.print(Text("Available reference series:", style=GREIGE))
            for uid in series:
                self.console.print(Text(f"  {uid}"))
