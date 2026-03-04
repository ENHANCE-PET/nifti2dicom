"""Click CLI with styled help — subcommands: convert, segment, rgb, resample."""

from __future__ import annotations

import json
import sys

import click

from nifti2dicom import __version__
from nifti2dicom import cli_theme as theme


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="nifti2dicom")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """NIfTI to DICOM converter with reference series."""
    if ctx.invoked_subcommand is None:
        theme.print_banner(__version__)
        click.echo(ctx.get_help())


# ── convert ───────────────────────────────────────────────────────


@cli.command()
@click.option(
    "-d", "--dicom-dir", required=True,
    type=click.Path(exists=True), help="Reference DICOM series directory.",
)
@click.option(
    "-n", "--nifti", required=True,
    type=click.Path(exists=True), help="NIfTI file to convert.",
)
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory.")
@click.option(
    "-desc", "--description",
    default="converted by nifti2dicom", help="Series description.",
)
@click.option(
    "--header-dir", type=click.Path(exists=True),
    default=None, help="Header source DICOM directory.",
)
@click.option("--force", is_flag=True, help="Overwrite output if it exists.")
def convert(
    dicom_dir: str,
    nifti: str,
    output: str,
    description: str,
    header_dir: str | None,
    force: bool,
) -> None:
    """Convert a NIfTI image to DICOM using a reference series."""
    theme.print_banner(__version__)
    from nifti2dicom.convert_image import convert_nifti_to_dicom

    try:
        convert_nifti_to_dicom(
            ref_dir=dicom_dir,
            nifti_path=nifti,
            output_dir=output,
            series_description=description,
            header_dir=header_dir,
            force_overwrite=force,
        )
    except Exception as exc:
        theme.err(str(exc))
        sys.exit(1)


# ── segment ───────────────────────────────────────────────────────


@cli.command()
@click.option(
    "-d", "--dicom-dir", required=True,
    type=click.Path(exists=True), help="Reference DICOM series directory.",
)
@click.option(
    "-n", "--nifti", required=True,
    type=click.Path(exists=True), help="NIfTI segmentation file.",
)
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory.")
@click.option(
    "-j", "--labels-json", required=True,
    type=click.Path(exists=True), help="JSON label-to-organ mapping.",
)
@click.option(
    "--manufacturer",
    default="Quantitative Imaging and Medical Physics",
    help="DICOM Manufacturer tag.",
)
@click.option("--model-name", default="nifti2dicom", help="DICOM ManufacturerModelName.")
def segment(
    dicom_dir: str,
    nifti: str,
    output: str,
    labels_json: str,
    manufacturer: str,
    model_name: str,
) -> None:
    """Convert a NIfTI segmentation to DICOM SEG."""
    theme.print_banner(__version__)
    from nifti2dicom.convert_seg import convert_nifti_seg_to_dicom

    with open(labels_json) as f:
        organ_index = json.load(f)

    try:
        convert_nifti_seg_to_dicom(
            ref_dir=dicom_dir,
            nifti_path=nifti,
            output_path=output,
            organ_index=organ_index,
            manufacturer=manufacturer,
            manufacturer_model_name=model_name,
        )
    except Exception as exc:
        theme.err(str(exc))
        sys.exit(1)


# ── rgb ───────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "-d", "--dicom-dir", required=True,
    type=click.Path(exists=True), help="Reference DICOM series directory.",
)
@click.option(
    "-n", "--nifti", required=True,
    type=click.Path(exists=True), help="RGB NIfTI file.",
)
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory.")
def rgb(dicom_dir: str, nifti: str, output: str) -> None:
    """Convert an RGB NIfTI image to DICOM."""
    theme.print_banner(__version__)
    from nifti2dicom.convert_rgb import convert_rgb_nifti_to_dicom

    try:
        convert_rgb_nifti_to_dicom(
            ref_dir=dicom_dir,
            nifti_path=nifti,
            output_dir=output,
        )
    except Exception as exc:
        theme.err(str(exc))
        sys.exit(1)


# ── resample ──────────────────────────────────────────────────────


@cli.command()
@click.option(
    "-d", "--dicom-dir", required=True,
    type=click.Path(exists=True), help="Original DICOM series directory.",
)
@click.option(
    "-n", "--nifti", required=True,
    type=click.Path(exists=True), help="NIfTI file to convert.",
)
@click.option("-o", "--output", required=True, type=click.Path(), help="Output directory.")
@click.option(
    "--spatial-dir", required=True,
    type=click.Path(exists=True), help="DICOM dir for spatial reference.",
)
@click.option(
    "-desc", "--description",
    default="converted by nifti2dicom", help="Series description.",
)
def resample(
    dicom_dir: str,
    nifti: str,
    output: str,
    spatial_dir: str,
    description: str,
) -> None:
    """Convert a NIfTI to DICOM with resampling to match original geometry."""
    theme.print_banner(__version__)
    from nifti2dicom.converter import nifti_to_dicom_with_resampling

    try:
        nifti_to_dicom_with_resampling(
            nifti_image_path=nifti,
            original_dicom_directory=dicom_dir,
            dicom_output_directory=output,
            spatial_info_dicom_directory=spatial_dir,
            series_description=description,
        )
    except Exception as exc:
        theme.err(str(exc))
        sys.exit(1)
