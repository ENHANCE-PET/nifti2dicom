"""Installed-Slicer acceptance, independent of converter geometry/pixel code."""

import json
import logging
import os
import sys
import traceback
from hashlib import sha256
from pathlib import Path

import numpy as np
import pydicom
import qt
import slicer
import vtk
from DICOMLib import DICOMUtils
from DICOMLib.DICOMBrowser import DICOMLoadableTable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_oracle import (
    check_frame_pixels,
    check_multivolume_timing,
    check_physical_coverage,
    check_plane_timing,
    check_representation,
    check_sequence_timing,
)

ROOT = Path(os.environ["SLICER_QA_ROOT"]).resolve()
ALIAS = os.environ.get("SLICER_QA_CASE", "pet_static_rider")
RUN = os.environ.get("SLICER_QA_RUN", ALIAS)
FORMAT = os.environ.get("SLICER_QA_FORMAT", "sequence")
assert FORMAT in {"sequence", "multivolume"}
ORIGINAL_FORMAT = qt.QSettings().value("DICOM/PreferredMultiVolumeImportFormat")
qt.QSettings().setValue("DICOM/PreferredMultiVolumeImportFormat", FORMAT)
CASE = next(c for c in json.loads((ROOT / "cases.json").read_text()) if c["alias"] == ALIAS)
MODALITY = CASE.get("modality", "PT")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
REPORT = {
    "case": ALIAS,
    "modality": MODALITY,
    "slicer_version": slicer.app.applicationVersion,
    "mode": "default",
    "preferred_representation": FORMAT,
    "revision": slicer.app.repositoryRevision,
    "cases_sha256": sha256((ROOT / "cases.json").read_bytes()).hexdigest(),
    "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    "oracle_sha256": sha256(Path(__file__).with_name("public_oracle.py").read_bytes()).hexdigest(),
    "selection": "DICOMUtils.selectHighestConfidenceLoadables + DICOMLoadableTable.setLoadables",
    "frames": [],
    "segmentations": [],
}


def attrs(node):
    return {str(n): str(node.GetAttribute(n)) for n in node.GetAttributeNames()}


def affine_lps(node):
    assert node.GetParentTransformNode() is None, "Unexpected parent transform"
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.array(
        [[matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
    )


def source_indices(node, truth):
    shape = slicer.util.arrayFromVolume(node).shape
    k, j, i = np.indices(shape)
    ijk = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
    world = affine_lps(node) @ ijk
    source = np.linalg.inv(truth["affine_lps"]) @ world
    nearest = np.rint(source[:3]).astype(int)
    residual = truth["affine_lps"] @ np.vstack([nearest, np.ones(i.size)]) - world
    world_error = float(np.linalg.norm(residual[:3], axis=0).max())
    assert world_error < 0.001, world_error
    truth_shape = truth["values"].shape[1:]
    assert np.all(nearest >= 0)
    assert np.all(nearest < np.array(truth_shape[::-1])[:, None])
    flat = np.ravel_multi_index(tuple(nearest[::-1]), truth_shape)
    check_physical_coverage(flat, np.prod(truth_shape))
    return nearest, world, world_error


def load_folder(directory):
    files = sorted(str(p) for p in directory.glob("*.dcm"))
    assert files

    def fingerprint():
        records = [
            [p.name, sha256(p.read_bytes()).hexdigest()] for p in sorted(directory.glob("*.dcm"))
        ]
        return sha256(json.dumps(records, separators=(",", ":")).encode("utf-8")).hexdigest()

    before = fingerprint()
    assert DICOMUtils.importDicom(str(directory), copyFiles=False)
    messages = []
    candidates, enabled = DICOMUtils.getLoadablesFromFileLists([files], messages=messages)
    assert enabled
    DICOMUtils.selectHighestConfidenceLoadables(candidates)
    table = DICOMLoadableTable(None)
    table.setLoadables(candidates)
    table.updateSelectedFromCheckstate()
    choices = [
        {
            "plugin": type(plugin).__name__,
            "name": loadable.name,
            "confidence": loadable.confidence,
            "selected": bool(loadable.selected),
            "warning": loadable.warning,
            "files": len(loadable.files),
        }
        for plugin, loadables in candidates.items()
        for loadable in loadables
    ]
    DICOMUtils.loadLoadables(candidates, messages=messages)
    selected = [item for loadables in candidates.values() for item in loadables if item.selected]
    result = {
        "directory": str(directory),
        "dicom_file_count": len(files),
        "dicom_files_sha256": before,
        "dicom_files_sha256_after_import": fingerprint(),
        "fingerprint_algorithm": (
            "SHA256 of UTF-8 JSON [[filename,SHA256(file bytes)],...] in filename order; "
            "ensure_ascii=True, separators=(',',':')"
        ),
        "choices": choices,
        "messages": messages,
        "load_success": [bool(item.loadSuccess) for item in selected],
    }
    roundtrip = directory.parent / "roundtrip.json"
    if roundtrip.is_file():
        result["roundtrip_report_sha256"] = sha256(roundtrip.read_bytes()).hexdigest()
    result["files_unchanged_during_import"] = before == result["dicom_files_sha256_after_import"]
    REPORT.setdefault("imports", []).append(result)
    assert result["files_unchanged_during_import"], "DICOM input changed during import"
    assert selected and all(item.loadSuccess for item in selected), result
    assert not messages, messages
    return candidates


def check_frame(node, truth, datasets, expected_t, source_uids=None, strict=True):
    values = slicer.util.arrayFromVolume(node)
    uids = (
        source_uids
        if source_uids is not None
        else (node.GetAttribute("DICOM.instanceUIDs") or "").split()
    )
    assert len(uids) == values.shape[0], (len(uids), values.shape)
    assert all(uid in datasets for uid in uids), "Unknown source instance"
    nearest, world, world_error = source_indices(node, truth)
    expected = truth["values"][expected_t][tuple(nearest[::-1])].reshape(values.shape)
    max_serialized_error, precision_violations = 0.0, 0
    plane_size = values.shape[1] * values.shape[2]
    for k, uid in enumerate(uids):
        ds = pydicom.dcmread(datasets[uid].filename)
        source_z = np.unique(nearest[2, k * plane_size : (k + 1) * plane_size])
        check_plane_timing(ds, truth, expected_t, source_z, MODALITY)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        spacing = np.array(ds.PixelSpacing, dtype=float)
        offset = (
            world[:3, k * plane_size : (k + 1) * plane_size]
            - np.array(ds.ImagePositionPatient, dtype=float)[:, None]
        )
        direction = np.column_stack(
            [iop[:3] * spacing[1], iop[3:] * spacing[0], np.cross(iop[:3], iop[3:])]
        )
        coords = np.linalg.solve(direction, offset)
        rounded = np.rint(coords).astype(int)
        assert np.max(np.abs(coords - rounded)) < 1e-5
        assert np.all(rounded[2] == 0), "SOP instance does not describe this loaded plane"
        assert np.all(rounded[:2] >= 0)
        assert np.all(rounded[0] < ds.Columns) and np.all(rounded[1] < ds.Rows)
        serialized = (
            ds.pixel_array[rounded[1], rounded[0]].astype(float) * float(ds.RescaleSlope)
            + float(ds.RescaleIntercept)
        ).reshape(values.shape[1:])
        error, violations = check_frame_pixels(values[k], serialized, expected[k], ds.RescaleSlope)
        max_serialized_error = max(max_serialized_error, error)
        precision_violations += violations
    result = {
        "timepoint": expected_t,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "voxel_count": int(values.size),
        "source_instance_count": len(uids),
        "maximum_world_error_mm": world_error,
        "maximum_serialized_value_error": max_serialized_error,
        "maximum_original_value_error": float(np.max(np.abs(values.astype(float) - expected))),
        "precision_bound_violations": precision_violations,
        "finite_pixels_verified": True,
        "complete_unique_physical_coverage_verified": True,
        "loaded_voxel_sha256": sha256(values.tobytes()).hexdigest(),
    }
    if MODALITY == "PT":
        times = sorted({float(datasets[uid].FrameReferenceTime) for uid in uids})
        durations = sorted({float(datasets[uid].ActualFrameDuration) for uid in uids})
        result["frame_reference_time_ms"] = times[0] if len(times) == 1 else times
        result["actual_frame_duration_ms"] = durations[0] if len(durations) == 1 else durations
        units = {str(datasets[uid].Units) for uid in uids}
        assert len(units) == 1
        result["value_units"] = units.pop()
        if result["value_units"] == "BQML":
            result["maximum_serialized_value_error_BQML"] = max_serialized_error
            result["maximum_original_value_error_BQML"] = result["maximum_original_value_error"]
    elif MODALITY == "MR":
        result["acquisition_time"] = str(truth["acquisition_times"][expected_t])
    REPORT["frames"].append(result)
    if strict:
        assert max_serialized_error == 0.0, result
        assert precision_violations == 0, result
    return result


def capture(name, volume, position_lps=None):
    import ScreenCapture

    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.mainWindow().resize(1400, 900)
    slicer.util.setSliceViewerLayers(background=volume)
    slicer.util.resetSliceViews()
    if position_lps is not None:
        ras = np.diag([-1.0, -1.0, 1.0]) @ np.asarray(position_lps)
        slicer.modules.markups.logic().JumpSlicesToLocation(*ras, True)
    slicer.app.processEvents()
    ScreenCapture.ScreenCaptureLogic().captureImageFromView(None, str(ROOT / name))
    display = volume.GetDisplayNode()
    if volume.IsA("vtkMRMLScalarVolumeNode"):
        before = {
            "window": display.GetWindow(),
            "level": display.GetLevel(),
            "automatic": bool(display.GetAutoWindowLevel()),
        }
        # Save a separately labeled presentation control. No pixels are changed.
        values = slicer.util.arrayFromVolume(volume)
        positive = values[values > 0]
        high = max(1.0, float(np.percentile(positive, 99))) if positive.size else 1.0
        display.SetAutoWindowLevel(False)
        display.SetWindowLevel(high, high / 2.0)
        slicer.app.processEvents()
        adjusted_name = name.replace(".png", "-windowed.png")
        ScreenCapture.ScreenCaptureLogic().captureImageFromView(None, str(ROOT / adjusted_name))
        REPORT.setdefault("presentation_controls", []).append(
            {
                "original_screenshot": name,
                "original_display": before,
                "adjusted_screenshot": adjusted_name,
                "adjusted_window": high,
                "adjusted_level": high / 2.0,
                "purpose": "Geometric inspection only: positive-voxel 99th percentile",
            }
        )
        display.SetWindowLevel(before["window"], before["level"])
        display.SetAutoWindowLevel(before["automatic"])


def check_seg(kind, volume, truth):
    existing = {n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")}
    load_folder(ROOT / f"{kind}-seg")
    nodes = [
        n
        for n in slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if n.GetID() not in existing
    ]
    assert len(nodes) == 1
    node = nodes[0]
    for other in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        other.GetDisplayNode().SetVisibility(other == node)
    nearest, _, _ = source_indices(volume, truth)
    label_truth = np.load(ROOT / "seg-truth.npz")["labels"]
    labels_in_loaded_grid = label_truth[tuple(nearest[::-1])].reshape(
        slicer.util.arrayFromVolume(volume).shape
    )
    expected_labels = {"QA sphere A": 5}
    if kind == "multilabel":
        expected_labels.update({"QA box B": 300, "QA asymmetric C": 1024})
    seg = node.GetSegmentation()
    assert seg.GetNumberOfSegments() == len(expected_labels)
    record = {
        "kind": kind,
        "synthetic_not_clinical": True,
        "segments": [],
        "attributes": attrs(node),
    }
    for index in range(seg.GetNumberOfSegments()):
        sid = seg.GetNthSegmentID(index)
        name = seg.GetSegment(sid).GetName()
        assert name in expected_labels, name
        actual = slicer.util.arrayFromSegmentBinaryLabelmap(node, sid, volume).astype(bool)
        expected = labels_in_loaded_grid == expected_labels[name]
        differences = int(np.count_nonzero(actual != expected))
        record["segments"].append(
            {
                "name": name,
                "nifti_label": 1 if kind == "binary" else expected_labels[name],
                "foreground_voxels": int(actual.sum()),
                "differing_voxels": differences,
            }
        )
        assert differences == 0, record
    sh = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    item = sh.GetItemByDataNode(node)
    referenced = set((sh.GetItemAttribute(item, "DICOM.ReferencedInstanceUIDs") or "").split())
    expected_refs = set((volume.GetAttribute("DICOM.instanceUIDs") or "").split())
    assert referenced == expected_refs, (len(referenced), len(expected_refs))
    record["referenced_source_instances"] = len(referenced)
    REPORT["segmentations"].append(record)
    capture(f"{RUN}-{kind}-seg.png", volume, (truth["affine_lps"] @ [55.0, 58.0, 25.0, 1.0])[:3])


def check_multivolume(truth, datasets):
    from qSlicerMultiVolumeExplorerModuleHelper import (
        qSlicerMultiVolumeExplorerModuleHelper as Helper,
    )

    nodes = slicer.util.getNodesByClass("vtkMRMLMultiVolumeNode")
    assert len(nodes) == 1
    mv = nodes[0]
    assert mv.GetNumberOfFrames() == CASE["expected_timepoints"]
    REPORT["multivolume"] = {
        "name": mv.GetName(),
        "attributes": attrs(mv),
        "dtype": str(slicer.util.arrayFromVolume(mv).dtype),
    }
    native_uids = (mv.GetAttribute("DICOM.instanceUIDs") or "").split()
    assert len(native_uids) == len(datasets)
    for t in range(mv.GetNumberOfFrames()):
        frame = Helper.extractFrame(None, mv, t)
        # Match each loaded plane to a referenced instance by its physical plane,
        # never assume the MultiVolume UID string follows Slicer's slice reversal.
        affine = affine_lps(frame)
        count = slicer.util.arrayFromVolume(frame).shape[0]
        assert count == CASE["expected_slices"]
        group = native_uids[t * count : (t + 1) * count]
        uids = []
        for k in range(count):
            origin = (affine @ [0.0, 0.0, float(k), 1.0])[:3]
            uids.append(
                min(
                    group,
                    key=lambda uid: np.linalg.norm(
                        np.array(datasets[uid].ImagePositionPatient, dtype=float) - origin
                    ),
                )
            )
        assert len(set(uids)) == count
        result = check_frame(frame, truth, datasets, t, source_uids=uids, strict=False)
        mv.GetDisplayNode().SetFrameComponent(t)
        slicer.app.processEvents()
        assert mv.GetDisplayNode().GetFrameComponent() == t
        result["display_component_checked"] = True
        print("MULTIVOLUME_FRAME", ALIAS, t, result["maximum_serialized_value_error"], flush=True)
        if t in {0, mv.GetNumberOfFrames() // 2, mv.GetNumberOfFrames() - 1}:
            capture(f"{RUN}-frame-{t:02d}.png", mv)
        slicer.mrmlScene.RemoveNode(frame)
    REPORT["multivolume"]["display_indices_checked"] = mv.GetNumberOfFrames()
    check_multivolume_timing(attrs(mv), truth, MODALITY)
    assert all(f["maximum_serialized_value_error"] == 0.0 for f in REPORT["frames"]), (
        "MultiVolume altered decoded values"
    )
    assert all(f["precision_bound_violations"] == 0 for f in REPORT["frames"]), (
        "MultiVolume exceeded source precision bounds"
    )


def run():
    assert hasattr(slicer.modules, "DICOMInstance"), "Script must run before testing-mode shutdown"
    with np.load(CASE["truth"]) as archive:
        assert sha256(Path(CASE["truth"]).read_bytes()).hexdigest() == CASE["truth_sha256"]
        # NpzFile decompresses an array on every access; materialize once so
        # checking each timepoint does not repeatedly decode the full 4D truth.
        truth = {key: archive[key] for key in archive.files}
        datasets = {
            str(ds.SOPInstanceUID): ds
            for p in Path(CASE["dicom"]).glob("*.dcm")
            for ds in [pydicom.dcmread(p, stop_before_pixels=True)]
        }
        with DICOMUtils.TemporaryDICOMDatabase(str(ROOT / f"{RUN}-database")):
            load_folder(Path(CASE["dicom"]))
            sequences = slicer.util.getNodesByClass("vtkMRMLSequenceNode")
            if CASE["expected_timepoints"] == 1:
                REPORT["actual_representation"] = "scalar"
                assert not sequences
                volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                assert len(volumes) == 1
                volume = volumes[0]
                check_frame(volume, truth, datasets, 0)
                REPORT["volume_attributes"] = attrs(volume)
                capture(f"{RUN}-{'pet' if MODALITY == 'PT' else MODALITY.lower()}.png", volume)
                for kind in CASE.get("seg_kinds", ["binary", "multilabel"]):
                    check_seg(kind, volume, truth)
            else:
                check_representation(
                    FORMAT,
                    len(sequences),
                    len(slicer.util.getNodesByClass("vtkMRMLMultiVolumeNode")),
                )
                REPORT["actual_representation"] = "sequence" if sequences else "multivolume"
                if not sequences:
                    check_multivolume(truth, datasets)
                    return
                assert len(sequences) == 1, len(sequences)
                sequence = sequences[0]
                assert sequence.GetNumberOfDataNodes() == CASE["expected_timepoints"]
                REPORT["sequence"] = {
                    "name": sequence.GetName(),
                    "index_name": sequence.GetIndexName(),
                    "index_unit": sequence.GetIndexUnit(),
                    "attributes": attrs(sequence),
                    "index_values": [
                        sequence.GetNthIndexValue(t) for t in range(sequence.GetNumberOfDataNodes())
                    ],
                }
                browsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
                assert len(browsers) == 1
                browser = browsers[0]
                for t in range(sequence.GetNumberOfDataNodes()):
                    node = sequence.GetNthDataNode(t)
                    result = check_frame(node, truth, datasets, t, strict=False)
                    expected_affine = affine_lps(node).copy()
                    expected_instance_uids = node.GetAttribute("DICOM.instanceUIDs")
                    browser.SetSelectedItemNumber(t)
                    slicer.app.processEvents()
                    proxy = browser.GetProxyNode(sequence)
                    # SaveChanges/shallow copies can alias data. Never compare
                    # two live buffers after navigation as an independent oracle.
                    proxy_hash = sha256(slicer.util.arrayFromVolume(proxy).tobytes()).hexdigest()
                    assert proxy_hash == result["loaded_voxel_sha256"]
                    assert np.array_equal(affine_lps(proxy), expected_affine)
                    assert proxy.GetAttribute("DICOM.instanceUIDs") == expected_instance_uids
                    result["browser_auto_window_level"] = bool(
                        proxy.GetDisplayNode().GetAutoWindowLevel()
                    )
                    if MODALITY == "PT":
                        assert result["browser_auto_window_level"]
                    if np.ptp(slicer.util.arrayFromVolume(proxy)) > 0:
                        assert proxy.GetDisplayNode().GetWindow() > 0
                    result["browser_proxy_exact"] = True
                    result["browser_proxy_sha256"] = proxy_hash
                    result["browser_oracle"] = (
                        "immutable pre-navigation checksum, affine copy and instance UID string"
                    )
                    print("FRAME_OK", ALIAS, t, flush=True)
                    if t in {
                        0,
                        sequence.GetNumberOfDataNodes() // 2,
                        sequence.GetNumberOfDataNodes() - 1,
                    }:
                        capture(f"{RUN}-frame-{t:02d}.png", proxy)
                REPORT["sequence"]["browser_indices_checked"] = sequence.GetNumberOfDataNodes()
                check_sequence_timing(
                    REPORT["sequence"]["index_values"],
                    sequence.GetIndexUnit(),
                    attrs(sequence),
                    truth,
                    MODALITY,
                )
                assert all(f["maximum_serialized_value_error"] == 0.0 for f in REPORT["frames"])
                assert all(f["precision_bound_violations"] == 0 for f in REPORT["frames"])
    REPORT["status"] = "passed"


try:
    run()
    REPORT["status"] = "passed"
    code = 0
except Exception:
    REPORT["status"] = "failed"
    REPORT["traceback"] = traceback.format_exc()
    print(REPORT["traceback"], flush=True)
    code = 1
(ROOT / f"{RUN}-acceptance.json").write_text(json.dumps(REPORT, indent=2))
print("SLICER_ACCEPTANCE", RUN, REPORT["status"], flush=True)
if ORIGINAL_FORMAT is None:
    qt.QSettings().remove("DICOM/PreferredMultiVolumeImportFormat")
else:
    qt.QSettings().setValue("DICOM/PreferredMultiVolumeImportFormat", ORIGINAL_FORMAT)
slicer.util.exit(code)
