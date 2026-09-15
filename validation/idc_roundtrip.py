"""Reproducible public-data oracle, independent of converter geometry helpers.

Run with the package's development environment. The input cache is populated
from the exact series in docs/validation/2026-09-15-idc-candidates.json; neither
source DICOM files nor their metadata are modified. This is opt-in validation,
not an automatic network dependency of the unit suite.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

from nifti2dicom import convert

RAS_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def fingerprints(paths):
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def package_fingerprints():
    root = Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "nifti2dicom").rglob("*.py"))
    }


def source_volume(paths, selection):
    """Establish time from source times/identifiers and space from plane tags."""
    groups = defaultdict(list)
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        assert str(ds.SeriesInstanceUID) == selection["SeriesInstanceUID"]
        if ds.Modality == "PT" and ds.SeriesType[0] == "DYNAMIC":
            time = float(ds.FrameReferenceTime)
        elif ds.Modality == "MR":
            time = int(ds.TemporalPositionIdentifier)
        else:
            time = 0
        groups[time].append((path, ds))
    times = sorted(groups)
    expected_t = (
        selection.get("NumberOfTimeSlices") or selection.get("NumberOfTemporalPositions") or 1
    )
    assert len(times) == expected_t
    first = groups[times[0]][0][1]
    iop = np.asarray(first.ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    frames = [
        sorted(groups[t], key=lambda pair: np.dot(pair[1].ImagePositionPatient, normal))
        for t in times
    ]
    nz = len(frames[0])
    assert nz > 1 and all(len(group) == nz for group in frames)
    positions = np.asarray([ds.ImagePositionPatient for _, ds in frames[0]], dtype=float)
    affine = np.eye(4)
    affine[:3, 0] = iop[:3] * float(first.PixelSpacing[1])
    affine[:3, 1] = iop[3:] * float(first.PixelSpacing[0])
    affine[:3, 2] = (positions[-1] - positions[0]) / (nz - 1)
    affine[:3, 3] = positions[0]
    assert np.all(np.diff(positions @ normal) > 0)
    assert len({str(ds.FrameOfReferenceUID) for group in frames for _, ds in group}) == 1
    values = np.empty((len(times), nz, int(first.Rows), int(first.Columns)), dtype=float)
    for t, group in enumerate(frames):
        for z, (path, header) in enumerate(group):
            np.testing.assert_allclose(header.ImageOrientationPatient, iop, rtol=0, atol=1e-8)
            np.testing.assert_allclose(header.ImagePositionPatient, positions[z], rtol=0, atol=1e-8)
            np.testing.assert_allclose(header.PixelSpacing, first.PixelSpacing, rtol=0, atol=1e-8)
            ds = pydicom.dcmread(path)
            values[t, z] = ds.pixel_array.astype(float) * float(getattr(ds, "RescaleSlope", 1))
            values[t, z] += float(getattr(ds, "RescaleIntercept", 0))
    assert np.isfinite(values).all()
    return values, affine, frames, times


def audit(paths, expected, affine, frames, times):
    """Map every output voxel into the source grid; do not trust file order."""
    inverse = np.linalg.inv(affine)
    recovered = np.full(expected.shape, np.nan)
    occupied = np.zeros(expected.shape, dtype=bool)
    maximum_error = maximum_world_error = tolerance = 0.0
    source_modality = frames[0][0][1].Modality
    metadata = (
        "FrameReferenceTime",
        "ActualFrameDuration",
        "DecayCorrection",
        "Units",
        "CorrectedImage",
        "AcquisitionTime",
        "AcquisitionDate",
        "DecayFactor",
    )
    seen_uids = set()
    series_uids = set()
    temporal_values = []
    for path in paths:
        ds = pydicom.dcmread(path)
        assert str(ds.SOPInstanceUID) not in seen_uids
        seen_uids.add(str(ds.SOPInstanceUID))
        series_uids.add(str(ds.SeriesInstanceUID))
        assert str(ds.FrameOfReferenceUID) == str(frames[0][0][1].FrameOfReferenceUID)
        if source_modality == "PT" and len(times) > 1:
            t = times.index(float(ds.FrameReferenceTime))
            assert (int(ds.ImageIndex) - 1) // expected.shape[1] == t
        elif len(times) > 1:
            t = int(ds.TemporalPositionIdentifier) - 1
            assert 0 <= t < len(times)
        else:
            t = 0
        yy, xx = np.indices((int(ds.Rows), int(ds.Columns)))
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        world = (
            np.asarray(ds.ImagePositionPatient, dtype=float)[:, None]
            + iop[:3, None] * float(ds.PixelSpacing[1]) * xx.ravel()
            + iop[3:, None] * float(ds.PixelSpacing[0]) * yy.ravel()
        )
        continuous = inverse[:3, :3] @ world + inverse[:3, 3, None]
        indices = np.rint(continuous).astype(int)
        x, y, z = indices
        for axis, size in zip(indices, expected.shape[:0:-1], strict=True):
            assert np.all((axis >= 0) & (axis < size))
        assert not occupied[t, z, y, x].any(), "Duplicate physical voxels"
        occupied[t, z, y, x] = True
        source_positions = np.asarray([h.ImagePositionPatient for _, h in frames[t]], dtype=float)
        source_iop = np.asarray(frames[t][0][1].ImageOrientationPatient, dtype=float)
        source_spacing = np.asarray(frames[t][0][1].PixelSpacing, dtype=float)
        original_world = (
            source_positions[z].T
            + source_iop[:3, None] * source_spacing[1] * x
            + source_iop[3:, None] * source_spacing[0] * y
        )
        world_error = float(np.max(np.linalg.norm(world - original_world, axis=0)))
        assert world_error < 0.001, world_error
        maximum_world_error = max(maximum_world_error, world_error)
        if source_modality == "PT":
            assert np.unique(z).size == 1
            original = frames[t][int(z[0])][1]
            for keyword in metadata:
                assert str(getattr(ds, keyword, "")) == str(getattr(original, keyword, "")), keyword
        elif source_modality == "CT":
            assert np.unique(z).size == 1
            original = frames[t][int(z[0])][1]
            for keyword in ("AcquisitionTime", "AcquisitionDate"):
                assert str(getattr(ds, keyword, "")) == str(getattr(original, keyword, "")), keyword
        elif source_modality == "MR":
            # This dataset has volume-uniform acquisition dates/times. It is
            # safe to preserve those even when canonicalization changes planes.
            for keyword in ("AcquisitionDate", "AcquisitionTime"):
                facts = {str(getattr(header, keyword, "")) for _, header in frames[t]}
                assert len(facts) == 1
                assert str(getattr(ds, keyword, "")) == facts.pop(), keyword
            assert "TemporalResolution" not in ds, "Invented uniform timing for irregular MR"
        decoded = ds.pixel_array.astype(float) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        recovered[t, z, y, x] = decoded.ravel()
        error = float(np.max(np.abs(decoded.ravel() - expected[t, z, y, x])))
        bound = abs(float(ds.RescaleSlope)) / 2 + 1e-7
        assert error <= bound, (path.name, error, bound)
        maximum_error = max(maximum_error, error)
        tolerance = max(tolerance, bound)
        temporal_values.append(t)
    assert len(series_uids) == 1
    assert occupied.all(), "Missing physical voxels"
    mutations = {}
    for name, axis in (("spatial_x", 3), ("spatial_y", 2), ("spatial_z", 1), ("time", 0)):
        if expected.shape[axis] == 1:
            continue
        differences = int(np.count_nonzero(np.abs(np.flip(recovered, axis) - expected) > tolerance))
        assert differences > 0, f"Oracle cannot detect {name} reversal"
        mutations[name] = differences
    return {
        "files": len(paths),
        "source_grid_voxels": int(expected.size),
        "timepoints": len(set(temporal_values)),
        "coverage_complete": True,
        "maximum_pixel_error": maximum_error,
        "largest_half_step_plus_numeric_tolerance": tolerance,
        "maximum_world_error_mm": maximum_world_error,
        "acquisition_metadata_preserved": True,
        "reconstructed_array_flip_sensitivity": mutations,
    }


def run_case(item, cache, destination, *, overwrite=False):
    alias, selection = item["alias"], item["series"]
    root = destination / alias
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted((cache / alias).glob("*.dcm"))
    assert len(paths) == selection["instanceCount"]
    before = fingerprints(paths)
    package_before = package_fingerprints()
    values, affine, frames, times = source_volume(paths, selection)
    source = root / "input.nii.gz"
    data = values.transpose(3, 2, 1, 0)
    image = nib.Nifti1Image(data[..., 0] if len(times) == 1 else data, RAS_LPS @ affine)
    image.header.set_xyzt_units("mm", "unknown")
    nib.save(image, source)
    np.testing.assert_array_equal(
        nib.load(source).get_fdata(), data[..., 0] if len(times) == 1 else data
    )
    modality = str(frames[0][0][1].Modality)
    extra = {}
    if modality == "PT":
        extra["frame_reference_times_ms"] = np.array(
            [[float(ds.FrameReferenceTime) for _, ds in group] for group in frames]
        )
        extra["frame_durations_ms"] = np.array(
            [[float(ds.ActualFrameDuration) for _, ds in group] for group in frames]
        )
        extra["times_ms"] = extra["frame_reference_times_ms"][:, 0]
        extra["durations_ms"] = extra["frame_durations_ms"][:, 0]
    elif modality == "MR":
        extra["acquisition_times"] = np.array(
            [str(group[0][1].AcquisitionTime) for group in frames]
        )
    truth = root / "truth.npz"
    np.savez_compressed(truth, values=values, affine_lps=affine, **extra)
    case = {
        "alias": alias,
        "modality": modality,
        "dicom": str(root / "dicom"),
        "truth": str(truth),
        "truth_sha256": sha256(truth.read_bytes()).hexdigest(),
        "expected_timepoints": len(times),
        "expected_slices": values.shape[1],
        "seg_kinds": [],
    }
    (root / "cases.json").write_text(json.dumps([case], indent=2) + "\n")
    print(
        json.dumps({"case": alias, "stage": "prepared", "shape_TZYX": list(values.shape)}),
        flush=True,
    )
    result = convert(source, cache / alias, root / "dicom", kind="image", overwrite=overwrite)
    assert len(result.files) % len(times) == 0
    case["expected_slices"] = len(result.files) // len(times)
    (root / "cases.json").write_text(json.dumps([case], indent=2) + "\n")
    report = audit(result.files, values, affine, frames, times)
    report.update(
        alias=alias,
        status="passed",
        warnings=list(result.warnings),
        source_sha256=before,
        output_sha256=fingerprints(result.files),
        nifti_sha256=sha256(source.read_bytes()).hexdigest(),
    )
    assert before == fingerprints(paths), "Original DICOMs changed"
    report["original_files_unchanged"] = True
    assert package_before == package_fingerprints(), "Package changed during the audit"
    report["provenance"] = {
        "package_source_sha256": package_before,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "nibabel", "pydicom")},
    }
    (root / "roundtrip.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if not k.endswith("sha256") and k != "provenance"}
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/validation/2026-09-15-idc-candidates.json",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true", help="Replace existing QA conversions")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = json.loads(args.manifest.read_text())["series"]
    selected = (
        [r for r in items if r["alias"] in args.case]
        if args.case
        else [r for r in items if r["alias"] != "ct_dynamic_description_only"]
    )
    assert selected and set(args.case) <= {r["alias"] for r in selected}
    outcomes = []
    for item in selected:
        try:
            run_case(item, args.cache.resolve(), args.output.resolve(), overwrite=args.overwrite)
            outcomes.append({"case": item["alias"], "status": "passed"})
        except Exception as exc:
            import traceback

            failure = {
                "case": item["alias"],
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            (args.output / item["alias"] / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            print(json.dumps(failure), flush=True)
            outcomes.append(failure)
    (args.output / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    return 0 if all(r["status"] == "passed" for r in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
