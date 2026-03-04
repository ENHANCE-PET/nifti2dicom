"""DigiTx-inspired terminal theme for nifti2dicom CLI.

Coral & greige palette drawn from thedigitxlab.com:
  - cfonts block banner with coral-to-greige gradient
  - Numbered section headers ("01 · SECTION NAME")
  - Status lines with symbols (› info, ✓ ok, ! warn, ✗ err)
  - Coral progress bars, greige borders
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager

from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ── Palette ───────────────────────────────────────────────────────

CORAL = "#E87461"
GREIGE = "#B5A89A"
MUTED = "dim"

# ── Brand ─────────────────────────────────────────────────────────

BRAND = "nifti2dicom"
TAGLINE = "NIfTI to DICOM converter with reference series"

# ── Console singleton ─────────────────────────────────────────────

console = Console()


# ── Banner ────────────────────────────────────────────────────────


def print_banner(version: str) -> None:
    """Print the nifti2dicom banner using cfonts with coral-to-greige gradient."""
    try:
        from cfonts import render

        output = render(
            "nifti2dicom",
            font="block",
            gradient=[CORAL, GREIGE],
            transition=True,
            space=False,
        )
        indented = "\n".join(" " + line for line in output.split("\n"))
        console.file.write("\n" + indented)
    except ImportError:
        console.print(f"\n  [bold {CORAL}]{BRAND}[/bold {CORAL}]\n")

    console.print()
    console.print(f"  [{GREIGE}]{TAGLINE}[/{GREIGE}]")
    console.print(f"  [{MUTED}]v{version}[/{MUTED}]")
    console.print()


def print_version(version: str) -> None:
    """Print a compact branded version line."""
    t = Text()
    t.append(BRAND, style=f"bold {CORAL}")
    t.append(f"  v{version}", style=MUTED)
    console.print(t)


# ── Section headers ──────────────────────────────────────────────


def section(title: str, number: str | None = None) -> None:
    """Print a numbered section header."""
    console.print()
    t = Text()
    if number:
        t.append(f"  {number}", style=f"bold {CORAL}")
        t.append(" \u00b7 ", style=MUTED)
    else:
        t.append("  ", style="")
    t.append(title.upper(), style="bold")
    console.print(t)
    rule = "\u2500" * len(TAGLINE)
    console.print(f"  {rule}", style=GREIGE)


# ── Tables ───────────────────────────────────────────────────────


def make_table(title: str | None = None, **kwargs: object) -> Table:  # type: ignore[override]
    """Create a table with DigiTx styling (rounded, greige border)."""
    return Table(
        title=title,
        box=box.ROUNDED,
        border_style=GREIGE,
        title_style=f"bold {CORAL}",
        header_style="bold",
        padding=(0, 1),
        **kwargs,  # type: ignore[arg-type]
    )


def make_kv_table() -> Table:
    """Create a headerless two-column key-value table."""
    t = Table(
        box=box.ROUNDED,
        border_style=GREIGE,
        show_header=False,
        show_edge=False,
        padding=(0, 1, 0, 2),
    )
    t.add_column("Key", style=f"bold {CORAL}", no_wrap=True)
    t.add_column("Value")
    return t


# ── Status lines ─────────────────────────────────────────────────


def info(msg: str) -> None:
    """Info-level status line (coral arrow, dim text)."""
    console.print(f"  [{CORAL}]\u203a[/{CORAL}] [{MUTED}]{msg}[/{MUTED}]")


def ok(msg: str) -> None:
    """Success status line (green check)."""
    console.print(f"  [bold green]\u2713[/bold green] {msg}")


def warn(msg: str) -> None:
    """Warning status line (yellow bang)."""
    console.print(f"  [bold yellow]![/bold yellow] [yellow]{msg}[/yellow]")


def err(msg: str) -> None:
    """Error status line (red cross)."""
    console.print(f"  [bold red]\u2717[/bold red] {msg}")


# ── Progress helpers ────────────────────────────────────────────


@contextmanager
def spinner(label: str) -> Generator[None, None, None]:
    """Coral dots spinner for indeterminate operations."""
    p = Progress(
        TextColumn(" "),
        SpinnerColumn("dots", style=Style(color=CORAL)),
        TextColumn(f"[{MUTED}]{label}[/{MUTED}]"),
        console=console,
        transient=True,
    )
    with p:
        p.add_task(label, total=None)
        yield


@contextmanager
def progress(total: int, label: str) -> Generator[Callable[[], None], None, None]:
    """Coral progress bar for countable operations."""
    p = Progress(
        TextColumn(f"  [{CORAL}]\u25b8[/{CORAL}]"),
        BarColumn(complete_style=Style(color=CORAL), finished_style=Style(color=CORAL)),
        MofNCompleteColumn(),
        TextColumn(f"[{MUTED}]{label}[/{MUTED}]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with p:
        task = p.add_task(label, total=total)
        yield lambda: p.advance(task)
