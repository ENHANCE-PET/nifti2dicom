"""Run inside Slicer with --testing --python-script; uses the real importers."""

import logging
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pydicom
import qt
import slicer
import vtk
from DICOMLib import DICOMUtils
from DICOMLib.DICOMBrowser import DICOMLoadableTable
from pydicom.dataelem import RawDataElement
from pydicom.tag import Tag

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import change_header, pet_series

logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class PETImportTests(unittest.TestCase):
    def setUp(self):
        slicer.mrmlScene.Clear(0)
        self.temp = tempfile.TemporaryDirectory(prefix="nifti2dicom-slicer-test-")
        self.root = Path(self.temp.name)
        self.database = DICOMUtils.TemporaryDICOMDatabase(str(self.root / "database"))
        self.database.__enter__()
        settings = qt.QSettings()
        self.original_format = settings.value("DICOM/PreferredMultiVolumeImportFormat")
        qt.QSettings().setValue("DICOM/PreferredMultiVolumeImportFormat", "sequence")

    def tearDown(self):
        slicer.mrmlScene.Clear(0)
        self.database.__exit__(None, None, None)
        settings = qt.QSettings()
        if self.original_format is None:
            settings.remove("DICOM/PreferredMultiVolumeImportFormat")
        else:
            settings.setValue("DICOM/PreferredMultiVolumeImportFormat", self.original_format)
        self.temp.cleanup()

    def examine(self, paths, *, legacy=False):
        DICOMUtils.importDicom(str(paths[0].parent), copyFiles=False)
        files = sorted(str(p) for p in paths)
        if legacy:
            plugin = slicer.modules.dicomPlugins["MultiVolumeImporterPlugin"]()
            candidates = {plugin: plugin.examine([files])}
        else:
            candidates, _ = DICOMUtils.getLoadablesFromFileLists([files])
        DICOMUtils.selectHighestConfidenceLoadables(candidates)
        table = DICOMLoadableTable(None)
        table.setLoadables(candidates)
        table.updateSelectedFromCheckstate()
        selected = [
            (plugin, loadable)
            for plugin, ls in candidates.items()
            for loadable in ls
            if loadable.selected
        ]
        self.assertEqual(
            len(selected), 1, [(type(plugin).__name__, item.name) for plugin, item in selected]
        )
        return selected[0]

    def load_sequence(self, paths, *, legacy=False):
        plugin, loadable = self.examine(paths, legacy=legacy)
        self.assertTrue(
            getattr(loadable, "loadAsVolumeSequence", False),
            f"Default import did not select a sequence: {loadable.name}",
        )
        self.assertFalse(loadable.warning)
        self.assertTrue(plugin.load(loadable))
        sequences = slicer.util.getNodesByClass("vtkMRMLSequenceNode")
        self.assertEqual(len(sequences), 1)
        return sequences[0]

    def assert_sequence_values(self, sequence, expected, expected_matrix=None):
        if expected_matrix is None:
            expected_matrix = [[-2.5, 0, 0, 12], [0, -1.5, 0, -23], [0, 0, 4, -9], [0, 0, 0, 1]]
        self.assertEqual(sequence.GetNumberOfDataNodes(), len(expected))
        browsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        self.assertEqual(len(browsers), 1)
        browser = browsers[0]
        for t, truth in enumerate(expected):
            node = sequence.GetNthDataNode(t)
            actual = slicer.util.arrayFromVolume(node)
            np.testing.assert_array_equal(actual, truth)
            expected_hash = sha256(actual.tobytes()).hexdigest()
            expected_uids = node.GetAttribute("DICOM.instanceUIDs")
            matrix = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(matrix)
            np.testing.assert_allclose(
                [[matrix.GetElement(i, j) for j in range(4)] for i in range(4)],
                expected_matrix,
            )
            browser.SetSelectedItemNumber(t)
            slicer.app.processEvents()
            proxy = browser.GetProxyNode(sequence)
            self.assertEqual(
                sha256(slicer.util.arrayFromVolume(proxy).tobytes()).hexdigest(), expected_hash
            )
            self.assertEqual(proxy.GetAttribute("DICOM.instanceUIDs"), expected_uids)

    def test_default_dispatch_offers_sequence_when_acquisition_time_varies(self):
        paths, expected = pet_series(self.root / "dicom", varying_acquisition=True)
        sequence = self.load_sequence(paths)
        self.assert_sequence_values(sequence, expected)

    def test_native_pet_index_with_constant_acquisition_time(self):
        paths, expected = pet_series(self.root / "dicom")
        sequence = self.load_sequence(paths)
        self.assertEqual(sequence.GetIndexUnit(), "ms")
        self.assertEqual(
            [float(sequence.GetNthIndexValue(t)) for t in range(3)], [0.0, 2000.0, 17000.0]
        )
        self.assert_sequence_values(sequence, expected)

    def test_signed_pet_preserves_negative_physical_values(self):
        paths, expected = pet_series(self.root / "signed")

        def make_signed(ds):
            pixels = -ds.pixel_array.astype(np.int16)
            ds.PixelRepresentation = 1
            ds.PixelData = pixels.tobytes()

        for path in paths:
            change_header(path, make_signed)
        sequence = self.load_sequence(paths)
        self.assert_sequence_values(sequence, [-frame for frame in expected])

    def test_oblique_pet_preserves_both_axes_and_time(self):
        paths, expected = pet_series(self.root / "oblique")
        for path in paths:
            change_header(
                path,
                lambda ds: setattr(ds, "ImageOrientationPatient", [0.6, 0.8, 0.0, -0.8, 0.6, 0.0]),
            )
        sequence = self.load_sequence(paths)
        self.assert_sequence_values(
            sequence,
            expected,
            [[-1.5, 1.2, 0, 12], [-2.0, -0.9, 0, -23], [0, 0, 4, -9], [0, 0, 0, 1]],
        )

    def test_slice_specific_reference_times_keep_native_time_blocks(self):
        paths, expected = pet_series(self.root / "dicom", slice_times=True)
        sequence = self.load_sequence(paths)
        self.assertEqual(sequence.GetIndexUnit(), "count")
        self.assertEqual([float(sequence.GetNthIndexValue(t)) for t in range(3)], [1.0, 2.0, 3.0])
        self.assert_sequence_values(sequence, expected)

    def test_multivolume_keeps_fractions_and_values_above_uint16(self):
        paths, expected = pet_series(self.root / "dicom", varying_acquisition=True)
        qt.QSettings().setValue("DICOM/PreferredMultiVolumeImportFormat", "multivolume")
        plugin, loadable = self.examine(paths)
        self.assertFalse(getattr(loadable, "loadAsVolumeSequence", False))
        self.assertTrue(plugin.load(loadable))
        nodes = slicer.util.getNodesByClass("vtkMRMLMultiVolumeNode")
        self.assertEqual(len(nodes), 1)
        np.testing.assert_array_equal(
            slicer.util.arrayFromVolume(nodes[0]), np.stack(expected, axis=-1)
        )

    def test_sequence_contrast_recovers_after_an_empty_first_frame(self):
        paths, expected = pet_series(self.root / "dicom", varying_acquisition=True)
        sequence = self.load_sequence(paths, legacy=True)
        self.assert_sequence_values(sequence, expected)
        browser = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")[0]
        proxy = browser.GetProxyNode(sequence)
        slicer.util.setSliceViewerLayers(background=proxy)
        slicer.util.resetSliceViews()
        slicer.app.processEvents()
        self.assertGreater(proxy.GetDisplayNode().GetWindow(), 0.0)
        self.assertTrue(proxy.GetDisplayNode().GetAutoWindowLevel())

    def test_invalid_dynamic_pet_does_not_fall_back_to_a_spatial_stack(self):
        mutations = {
            "empty SeriesType": lambda paths: change_header(
                paths[0], lambda ds: setattr(ds, "SeriesType", None)
            ),
            "missing SeriesType throughout": lambda paths: [
                change_header(path, lambda ds: delattr(ds, "SeriesType")) for path in paths
            ],
            "missing image": lambda paths: paths.pop(),
            "duplicate ImageIndex": lambda paths: change_header(
                paths[-1], lambda ds: setattr(ds, "ImageIndex", 1)
            ),
            "missing FrameReferenceTime": lambda paths: change_header(
                paths[-1], lambda ds: delattr(ds, "FrameReferenceTime")
            ),
            "reversed time": lambda paths: change_header(
                paths[-1], lambda ds: setattr(ds, "FrameReferenceTime", "-1")
            ),
            "shifted geometry": lambda paths: change_header(
                paths[-1], lambda ds: setattr(ds, "ImagePositionPatient", [-11.0, 23.0, -5.0])
            ),
        }
        for index, (name, mutate) in enumerate(mutations.items()):
            with self.subTest(name=name):
                paths, _ = pet_series(self.root / f"invalid-{index}")
                mutate(paths)
                plugin, loadable = self.examine(paths)
                self.assertTrue(loadable.warning, "Malformed PET needs an understandable warning")
                self.assertIn("PET", loadable.warning)
                self.assertFalse(plugin.load(loadable))
                for cls in [
                    "vtkMRMLSequenceNode",
                    "vtkMRMLScalarVolumeNode",
                    "vtkMRMLMultiVolumeNode",
                ]:
                    self.assertFalse(
                        slicer.util.getNodesByClass(cls), f"Partial {cls} left after failure"
                    )

    def test_unreadable_header_returns_pet_warning(self):
        # Direct plugin entry also accepts externally supplied file lists. A
        # completely unreadable file is not a normally indexed DICOM instance.
        paths, _ = pet_series(self.root / "unreadable")
        paths[0].write_bytes(b"Not a DICOM image")
        plugin = slicer.modules.dicomPlugins["MultiVolumeImporterPlugin"]()
        loadables = plugin.examineForImport([[str(path) for path in paths]])
        self.assertEqual(len(loadables), 1)
        self.assertIn("PET image headers could not all be read", loadables[0].warning)
        self.assertFalse(plugin.load(loadables[0]))

    def test_lazy_header_decode_errors_return_pet_warnings(self):
        paths, _ = pet_series(self.root / "lazy-header")
        read_header = pydicom.dcmread
        for keyword in ("NumberOfTimeSlices", "Rows"):
            with self.subTest(keyword=keyword):
                damaged = read_header(paths[0], stop_before_pixels=True)
                tag = Tag(pydicom.datadict.tag_for_keyword(keyword))
                damaged[tag] = RawDataElement(tag, "US", 3, b"\x02\x00\xff", 0, False, True)

                def read_with_bad_element(path, damaged_header=damaged, **kwargs):
                    if str(path) == str(paths[0]):
                        return damaged_header
                    return read_header(path, **kwargs)

                plugin = slicer.modules.dicomPlugins["MultiVolumeImporterPlugin"]()
                with patch.object(pydicom, "dcmread", side_effect=read_with_bad_element):
                    loadables = plugin.examineForImport([[str(path) for path in paths]])
                self.assertEqual(len(loadables), 1)
                self.assertIn("PET", loadables[0].warning)
                self.assertFalse(plugin.load(loadables[0]))

    def test_static_pet_still_uses_scalar_reader(self):
        paths, expected = pet_series(self.root / "dicom", static=True)
        plugin, loadable = self.examine(paths)
        self.assertEqual(plugin.loadType, "Scalar Volume")
        self.assertTrue(plugin.load(loadable))
        nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        self.assertEqual(len(nodes), 1)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(nodes[0]), expected[0])

    def test_non_pet_dynamic_mr_keeps_native_sequence_import(self):
        paths, expected = pet_series(self.root / "mr", varying_acquisition=True)

        def make_mr(ds):
            ds.Modality = "MR"
            ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            for name in (
                "SeriesType",
                "NumberOfSlices",
                "NumberOfTimeSlices",
                "FrameReferenceTime",
            ):
                delattr(ds, name)

        for path in paths:
            change_header(path, make_mr)
        sequence = self.load_sequence(paths)
        self.assert_sequence_values(sequence, expected)

    def test_late_decode_failure_preserves_existing_volume(self):
        existing = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Keep me")
        truth = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        slicer.util.updateVolumeFromArray(existing, truth)
        existing.CreateDefaultDisplayNodes()
        slicer.util.setSliceViewerLayers(background=existing)
        selection = slicer.app.applicationLogic().GetSelectionNode()
        selection.SetReferenceActiveVolumeID(existing.GetID())
        yellow = slicer.app.layoutManager().sliceWidget("Yellow").mrmlSliceCompositeNode()
        yellow.SetBackgroundVolumeID(None)
        paths, _ = pet_series(self.root / "damaged")
        plugin, loadable = self.examine(paths)
        hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        original_items = vtk.vtkIdList()
        hierarchy.GetItemChildren(hierarchy.GetSceneItemID(), original_items, True)
        for path in paths[-2:]:
            change_header(path, lambda ds: delattr(ds, "PixelData"))
        self.assertFalse(plugin.load(loadable))
        self.assertIn("PET", loadable.warning)
        self.assertEqual(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"), [existing])
        self.assertFalse(slicer.util.getNodesByClass("vtkMRMLSequenceNode"))
        self.assertEqual(selection.GetActiveVolumeID(), existing.GetID())
        self.assertIsNone(yellow.GetBackgroundVolumeID())
        remaining_items = vtk.vtkIdList()
        hierarchy.GetItemChildren(hierarchy.GetSceneItemID(), remaining_items, True)
        self.assertEqual(remaining_items.GetNumberOfIds(), original_items.GetNumberOfIds())
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(existing), truth)

    def test_cancel_after_one_frame_does_not_publish_partial_series(self):
        paths, _ = pet_series(self.root / "cancel")
        plugin, loadable = self.examine(paths)

        class CancelAfterFirstFrame:
            value = 0

            @property
            def wasCanceled(self):
                return self.value == 1

            def close(self):
                pass

        with patch.object(
            slicer.util, "createProgressDialog", return_value=CancelAfterFirstFrame()
        ):
            self.assertFalse(plugin.load(loadable))
        self.assertIn("cancelled", loadable.warning)
        self.assertFalse(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
        self.assertFalse(slicer.util.getNodesByClass("vtkMRMLSequenceNode"))

    def test_decoder_geometry_checks_each_in_plane_axis(self):
        from Nifti2DicomPETLib.loader import _validate_decoding

        paths, _ = pet_series(self.root / "axis-check", static=True)
        plugin, loadable = self.examine(paths)
        node = plugin.load(loadable)
        matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(matrix)
        # Keep both origin and opposite corner unchanged while corrupting each
        # in-plane axis: 4 columns * 3 - 3 rows * 4 = 0 at the far corner.
        matrix.SetElement(0, 0, matrix.GetElement(0, 0) + 3)
        matrix.SetElement(0, 1, matrix.GetElement(0, 1) - 4)
        node.SetIJKToRASMatrix(matrix)
        with self.assertRaisesRegex(ValueError, "physical image grid"):
            _validate_decoding(node, [str(path) for path in paths])


result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(PETImportTests)
)
slicer.util.exit(0 if result.wasSuccessful() else 1)
