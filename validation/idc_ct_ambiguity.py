"""Verify safe rejection of the ten ambiguous, separately identified CT phases."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nifti2dicom import convert
from nifti2dicom.errors import AmbiguousReferenceError
from validation.idc_roundtrip import fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--nifti", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    originals = sorted(args.cache.glob("ct_respiratory_*/*.dcm"))
    assert len(originals) == 500
    before = fingerprints(originals)
    attempts = []
    with TemporaryDirectory(prefix="nifti2dicom-ct-ambiguity-") as temp:
        root = Path(temp)
        reference = root / "reference"
        reference.mkdir()
        for index, path in enumerate(originals):
            (reference / f"{index:04d}.dcm").symlink_to(path.resolve())
        for existing in (False, True):
            output = root / ("existing" if existing else "new")
            if existing:
                output.mkdir()
                (output / "keep.txt").write_text("existing user output\n")
            try:
                convert(args.nifti, reference, output, kind="image", overwrite=existing)
            except AmbiguousReferenceError as exc:
                assert exc.hint and len(exc.details["series_uids"]) == 10
                attempts.append({"existing_output": existing, **exc.to_dict()})
            else:
                raise AssertionError("Ambiguous phases were silently accepted")
            if existing:
                assert [p.name for p in output.iterdir()] == ["keep.txt"]
                assert (output / "keep.txt").read_text() == "existing user output\n"
            else:
                assert not output.exists()
        assert sorted(p.name for p in root.iterdir()) == ["existing", "reference"]
    assert before == fingerprints(originals)
    report = {
        "status": "passed",
        "source_files": len(originals),
        "source_files_unchanged": True,
        "no_partial_output": True,
        "existing_output_preserved": True,
        "attempts": attempts,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
