"""Click commands and compatibility aliases, delegating to one public API."""

from __future__ import annotations

import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click

from nifti2dicom import __version__
from nifti2dicom.cli.presentation import Presentation
from nifti2dicom.errors import ConversionError

_MODES = {"--json", "--quiet", "--debug"}
_PATH = click.Path(path_type=Path)


def _display_options() -> list[click.Option]:
    return [
        click.Option(
            ["--json"],
            is_flag=True,
            expose_value=False,
            help="Write one JSON result or error to stdout.",
        ),
        click.Option(
            ["--quiet"],
            is_flag=True,
            expose_value=False,
            help="Suppress progress and summaries; errors remain visible.",
        ),
        click.Option(
            ["--debug"],
            is_flag=True,
            expose_value=False,
            help="Show a traceback for unexpected failures.",
        ),
    ]


def _input_options() -> list[click.Parameter]:
    return [
        click.Argument(["input_path"], type=_PATH, required=False, metavar="[NIFTI]"),
        click.Option(
            ["-n", "--nifti"],
            type=_PATH,
            help="Input NIfTI file; alternative to the positional NIFTI.",
        ),
        click.Option(
            ["-d", "--dicom-dir", "--reference", "reference"],
            type=_PATH,
            required=True,
            help="Reference DICOM directory or file.",
        ),
        click.Option(["--series-uid"], help="Select a series when the reference has several."),
        click.Option(
            ["-j", "--labels-json", "--labels", "labels"],
            type=_PATH,
            help="JSON label names and segmentation metadata.",
        ),
    ]


def _select_input(input_path: Path | None, nifti: Path | None) -> Path:
    if input_path is not None and nifti is not None:
        raise click.UsageError("Provide NIFTI once, as a positional path or with --nifti.")
    selected = input_path if input_path is not None else nifti
    if selected is None:
        raise click.UsageError(
            "Provide a NIFTI path, for example input.nii.gz --reference ./dicom."
        )
    return selected


def _convert(input_path: Path | None, nifti: Path | None, reference: Path, **options: Any) -> None:
    selected = _select_input(input_path, nifti)
    from nifti2dicom import api

    presentation: Presentation = click.get_current_context().obj
    spatial_dir = options.pop("spatial_dir", None)
    if spatial_dir is not None:
        options["header_source"] = reference
        reference = spatial_dir
        options["geometry"] = "reference"
    output = options.pop("output")
    for key in ("manufacturer", "manufacturer_model_name"):
        if options.get(key) is None:
            options.pop(key)
    progress = None if presentation.json_output or presentation.quiet else presentation.progress
    result = api.convert(
        selected,
        reference,
        output,
        on_progress=progress,
        **options,
    )
    presentation.result(result)


def _inspect(input_path: Path | None, nifti: Path | None, reference: Path, **options: Any) -> None:
    selected = _select_input(input_path, nifti)
    from nifti2dicom import api

    result = api.inspect(selected, reference, **options)
    presentation: Presentation = click.get_current_context().obj
    presentation.result(result)


def _conversion_command(name: str, *, kind: str = "auto", resample: bool = False) -> click.Command:
    params = _input_options()
    params.extend(
        [
            click.Option(
                ["-o", "--output"],
                type=_PATH,
                help="Output directory; defaults to <input>_dicom beside the input.",
            ),
            click.Option(
                ["--kind"],
                type=click.Choice(["auto", "image", "seg", "rgb"]),
                default=kind,
                show_default=True,
                help="Interpretation of the NIfTI data.",
            ),
            click.Option(
                ["--geometry"],
                type=click.Choice(["native", "reference"]),
                default="reference" if resample else "native",
                show_default=True,
                help="Output grid; segmentation always uses the reference grid.",
            ),
            click.Option(["-desc", "--description"], help="DICOM series description."),
            click.Option(
                ["--header-dir", "header_source"],
                type=_PATH,
                help="Optional separate DICOM header source.",
            ),
            click.Option(
                ["--overwrite", "--force", "overwrite"],
                is_flag=True,
                help="Replace an existing output after successful validation.",
            ),
            click.Option(
                ["--algorithm-type"],
                type=click.Choice(["manual", "automatic", "semiautomatic"]),
                help="SEG provenance; required unless specified in the labels JSON.",
            ),
            click.Option(
                ["--algorithm-name"], help="Name of the algorithm that created the labels."
            ),
            click.Option(["--algorithm-version"], help="Version of the segmentation algorithm."),
            click.Option(["--manufacturer"], help="SEG manufacturer metadata."),
            click.Option(["--model-name", "manufacturer_model_name"], help="SEG model metadata."),
        ]
    )
    if resample:
        params.append(
            click.Option(
                ["--spatial-dir"],
                type=_PATH,
                required=True,
                help="Target DICOM geometry; -d supplies source headers.",
            )
        )
    params.extend(_display_options())
    descriptions = {
        "nifti2dicom": "Convert a NIfTI using a DICOM reference. No interactive prompts.",
        "convert": "Convert a scalar NIfTI image using a DICOM reference.",
        "segment": "Convert labels to DICOM SEG on the reference grid.",
        "rgb": "Convert color data to RGB Secondary Capture DICOM.",
        "resample": "Resample onto --spatial-dir geometry with -d source headers.",
    }
    return click.Command(name, params=params, callback=_convert, help=descriptions[name])


_direct = click.version_option(__version__, prog_name="nifti2dicom")(
    _conversion_command("nifti2dicom")
)


def _modes(args: Sequence[str]) -> set[str]:
    """Find presentation flags without treating option values as flags."""
    flags: set[str] = set()
    needs_value = {
        option
        for parameter in _direct.params
        if isinstance(parameter, click.Option) and not parameter.is_flag
        for option in parameter.opts
    } | {"--spatial-dir"}
    iterator = iter(args)
    for arg in iterator:
        if arg == "--":
            break
        if arg in needs_value:
            next(iterator, None)
        elif arg in _MODES:
            flags.add(arg)
    return flags


class ConversionCLI(click.Group):
    """Dispatch direct input syntax and normalize all errors, including parsing."""

    def format_usage(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        formatter.write_usage(ctx.command_path, "[OPTIONS] [NIFTI]")

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _direct.format_options(ctx, formatter)
        self.format_commands(ctx, formatter)

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        **extra: Any,
    ) -> Any:
        arguments = list(sys.argv[1:] if args is None else args)
        modes = _modes(arguments)
        presentation = Presentation(
            json_output="--json" in modes, quiet="--quiet" in modes, debug="--debug" in modes
        )
        extra["obj"] = presentation
        first = next((arg for arg in arguments if arg not in _MODES), None)
        use_group = first is None or first in self.commands or first in {"--help", "--version"}
        command = self if use_group else _direct
        try:
            # Call the base implementation to avoid recursively entering this router.
            result = click.Command.main(
                command,
                args=arguments,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                **extra,
            )
        except ConversionError as exc:
            presentation.error(exc.to_dict())
            if standalone_mode:
                raise SystemExit(1) from None
            raise
        except click.ClickException as exc:
            presentation.error(
                {
                    "code": "usage_error",
                    "message": exc.format_message(),
                    "hint": "Run nifti2dicom --help for usage and examples.",
                    "details": {},
                }
            )
            if standalone_mode:
                raise SystemExit(exc.exit_code) from None
            raise
        except click.Abort:
            presentation.error(
                {
                    "code": "interrupted",
                    "message": "Conversion interrupted.",
                    "hint": "Run the command again when ready.",
                    "details": {},
                }
            )
            if standalone_mode:
                raise SystemExit(130) from None
            raise
        except Exception as exc:
            presentation.error(
                {
                    "code": "internal_error",
                    "message": f"Unexpected failure: {type(exc).__name__}: {exc}",
                    "hint": "Run again with --debug and report the traceback.",
                    "details": {},
                }
            )
            if presentation.debug:
                traceback.print_exc(file=sys.stderr)
            if standalone_mode:
                raise SystemExit(1) from None
            raise
        if standalone_mode:
            raise SystemExit(result if isinstance(result, int) else 0)
        return result


@click.group(cls=ConversionCLI, invoke_without_command=True)
@click.version_option(__version__, prog_name="nifti2dicom")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Convert a NIfTI using a DICOM reference. No interactive prompts.

    \b
    nifti2dicom input.nii.gz --reference ./dicom
    nifti2dicom input.nii.gz --reference ./dicom -o ./output --json
    nifti2dicom inspect input.nii.gz --reference ./dicom

    Integer data defaults to image; segmentation requires labels metadata or --kind seg.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.params.extend(_display_options())
cli.add_command(_conversion_command("convert", kind="image"))
cli.add_command(_conversion_command("segment", kind="seg"))
cli.add_command(_conversion_command("rgb", kind="rgb"))
cli.add_command(_conversion_command("resample", kind="image", resample=True))
cli.add_command(
    click.Command(
        "inspect",
        callback=_inspect,
        help="Inspect interpretation and geometry without writing files.",
        params=[
            *_input_options(),
            click.Option(
                ["--kind"],
                type=click.Choice(["auto", "image", "seg", "rgb"]),
                default="auto",
                show_default=True,
            ),
            *_display_options(),
        ],
    )
)
