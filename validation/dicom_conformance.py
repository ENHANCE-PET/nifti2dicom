"""Check every generated instance with two independent, unmodified validators.

Run in an environment with dicom-validator and pydicom. Supply the downloaded
DICOM standard JSON directory and the official dciodvfy executable explicitly.
Results include every input hash, tool versions and aggregated diagnostics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import pydicom
from dicom_validator.validator.dicom_info import DicomInfo
from dicom_validator.validator.error_handler import NullValidationResultHandler
from dicom_validator.validator.iod_validator import IODValidator


def check_case(directory, info, executable, provenance):
    records, iod_patterns, native_patterns = [], Counter(), Counter()
    paths = sorted((directory / "dicom").glob("*.dcm"))
    expected = json.loads((directory / "roundtrip.json").read_text())["output_sha256"]
    assert {path.name for path in paths} == set(expected)
    for path in paths:
        digest = sha256(path.read_bytes()).hexdigest()
        assert digest == expected[path.name]
        result = IODValidator(
            pydicom.dcmread(path, defer_size=1024),
            info,
            error_handler=NullValidationResultHandler(),
            file_path=str(path),
        ).validate()
        for module, details in result.module_errors.items():
            for tag, error in details.items():
                iod_patterns[f"{module}: {tag}: {error.code.name}: {error.scope.name}"] += 1
        native = subprocess.run(
            [str(executable), str(path)], capture_output=True, text=True, timeout=60, check=False
        )
        lines = (native.stdout + native.stderr).splitlines()
        diagnostics = [line for line in lines if line.startswith(("Error", "Warning"))]
        native_patterns.update(diagnostics)
        native_errors = sum(line.startswith("Error") for line in lines)
        assert digest == sha256(path.read_bytes()).hexdigest(), "Validator altered input"
        records.append(
            {
                "file": path.name,
                "sha256": digest,
                "iod_errors": result.errors,
                "dciodvfy_exit": native.returncode,
                "dciodvfy_errors": native_errors,
            }
        )
    summary = {
        "case": directory.name,
        "files": len(records),
        "iod_errors": sum(row["iod_errors"] for row in records),
        "dciodvfy_errors": sum(row["dciodvfy_errors"] for row in records),
        "dciodvfy_failed_files": sum(row["dciodvfy_exit"] != 0 for row in records),
        "iod_diagnostics": dict(iod_patterns),
        "dciodvfy_diagnostics": dict(native_patterns),
        "files_unchanged": True,
        "provenance": provenance,
        "instances": records,
    }
    (directory / "conformance.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"instances", "provenance"}}),
        flush=True,
    )
    # Some dciodvfy diagnostics say Error despite a zero process exit status.
    return not (
        summary["iod_errors"] or summary["dciodvfy_errors"] or summary["dciodvfy_failed_files"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--standard", required=True, type=Path)
    parser.add_argument("--dciodvfy", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    names = ("dict_info.json", "iod_info.json", "module_info.json")
    info = DicomInfo(*(json.loads((args.standard / name).read_text()) for name in names))
    native_version = subprocess.run(
        [str(args.dciodvfy), "-version"], capture_output=True, text=True, check=False
    )
    provenance = {
        "dicom_validator": version("dicom-validator"),
        "pydicom": version("pydicom"),
        "standard_directory": str(args.standard),
        "validators_or_standard_modified": False,
        "standard_sha256": {n: sha256((args.standard / n).read_bytes()).hexdigest() for n in names},
        "dciodvfy_version": native_version.stdout + native_version.stderr,
        "dciodvfy_sha256": sha256(args.dciodvfy.read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    cases = sorted(path.parent for path in args.root.glob("*/roundtrip.json"))
    if args.case:
        cases = [path for path in cases if path.name in args.case]
        assert {p.name for p in cases} == set(args.case)
    assert cases
    outcomes = [check_case(path, info, args.dciodvfy, provenance) for path in cases]
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
