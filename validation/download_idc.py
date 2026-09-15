"""Download the audited public IDC cases into an explicit local cache.

Requires idc-index; not a runtime dependency of nifti2dicom. The pinned manifest
records source identifiers, attribution and licenses. Index-version drift is
an error rather than an implicit change to the validation dataset.
"""

import argparse
import json
from pathlib import Path

from idc_index import IDCClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/validation/2026-09-15-idc-candidates.json",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = [row for row in manifest["series"] if row["alias"] != "ct_dynamic_description_only"]
    assert sum(row["series"]["series_size_MB"] for row in selected) < 610
    client = IDCClient()
    assert client.get_idc_version() == manifest["idc_version"], "IDC index version changed"
    client.get_index_schema("index")
    client.fetch_index("index")
    for item in selected:
        row = item["series"]
        assert row["license_short_name"] in ("CC BY 3.0", "CC BY 4.0")
        destination = args.cache / item["alias"]
        destination.mkdir(parents=True, exist_ok=True)
        client.download_from_selection(
            downloadDir=str(destination),
            seriesInstanceUID=[row["SeriesInstanceUID"]],
            dirTemplate="",
        )
        assert len(list(destination.glob("*.dcm"))) == row["instanceCount"], item["alias"]
        print(
            json.dumps(
                {
                    "case": item["alias"],
                    "files": row["instanceCount"],
                    "estimated_MB": row["series_size_MB"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
