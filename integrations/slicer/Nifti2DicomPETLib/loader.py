"""Native decoding and atomic assembly of validated PET time blocks in Slicer."""

import numpy as np
import pydicom
import qt
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .scene import SceneTransaction


def _matrix(node):
    matrix = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(matrix)
    return matrix


def _validate_decoding(node, files):
    """Check the actual reader grid, not only agreement between source headers."""
    pixels = slicer.util.arrayFromVolume(node)
    if not np.isfinite(pixels).all():
        raise ValueError("PET decoding produced non-finite pixel values")
    by_uid = {
        str(ds.SOPInstanceUID): ds
        for ds in (pydicom.dcmread(path, stop_before_pixels=True) for path in files)
    }
    uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
    if len(uids) != len(files) or set(uids) != set(by_uid):
        raise ValueError("PET decoder did not preserve the source image references")
    matrix = _matrix(node)
    for z, uid in enumerate(uids):
        ds = by_uid[uid]
        if pixels.shape != (len(files), int(ds.Rows), int(ds.Columns)):
            raise ValueError("PET decoder changed the image dimensions")
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float).reshape(2, 3)
        origin = np.asarray(ds.ImagePositionPatient, dtype=float)
        corners = (
            (0, 0),
            (int(ds.Columns) - 1, 0),
            (0, int(ds.Rows) - 1),
            (int(ds.Columns) - 1, int(ds.Rows) - 1),
        )
        for x, y in corners:
            source = origin + x * float(ds.PixelSpacing[1]) * orientation[0]
            source += y * float(ds.PixelSpacing[0]) * orientation[1]
            loaded = np.asarray(matrix.MultiplyPoint((x, y, z, 1)))[:3] * (-1, -1, 1)
            if not np.allclose(source, loaded, atol=0.001, rtol=0):
                raise ValueError(
                    "PET decoder changed the physical image grid; resampling is not allowed"
                )


def _decode_frame(reader, files, name):
    candidates = reader.examineForImport([list(files)])
    candidates = [candidate for candidate in candidates if set(candidate.files) == set(files)]
    if not candidates:
        raise ValueError("PET time slice could not be read as one complete spatial volume")
    candidate = max(candidates, key=lambda item: item.confidence)
    # Explicit GDCM is the independently checked decoding path. Do not silently
    # switch pixel semantics with a user preference or harden an acquisition warp.
    node = reader.loadFilesWithSeriesReader("GDCM", candidate.files, name)
    if node is None or node.GetImageData() is None:
        raise ValueError("PET pixel data could not be decoded; check for damaged or missing images")
    reader.setVolumeNodeProperties(node, candidate)
    _validate_decoding(node, files)
    return node


def _set_frame_metadata(node, frames):
    node.SetAttribute("MultiVolume.FrameLabels", ",".join(frames.index_values))
    node.SetAttribute("MultiVolume.NumberOfFrames", str(len(frames.frame_files)))
    node.SetAttribute("MultiVolume.FrameIdentifyingDICOMTagName", frames.tag_name)
    node.SetAttribute("MultiVolume.FrameIdentifyingDICOMTagUnits", frames.index_unit)
    node.SetAttribute("Nifti2DicomPET.IntegrationVersion", "1")


def load_pet(plugin, loadable, frames):
    """Publish a complete sequence/MultiVolume, or remove only this load's nodes."""
    scene = slicer.mrmlScene
    sequence_mode = loadable.loadAsVolumeSequence
    result = scene.CreateNodeByClass(
        "vtkMRMLSequenceNode" if sequence_mode else "vtkMRMLMultiVolumeNode"
    )
    result.UnRegister(None)
    result.SetName(scene.GenerateUniqueName(loadable.name))
    _set_frame_metadata(result, frames)
    if sequence_mode:
        result.SetIndexName(frames.index_name)
        result.SetIndexUnit(frames.index_unit)
    image, image_array, color_id = None, None, None
    source_uids = []
    reader = slicer.modules.dicomPlugins["DICOMScalarVolumePlugin"]()
    # Initialize shared hierarchy/color nodes before tracking temporary objects.
    slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(scene)
    slicer.modules.colors.logic().GetPETColorNodeID(slicer.vtkMRMLPETProceduralColorNode.PETheat)
    selection = slicer.app.applicationLogic().GetSelectionNode()
    progress = slicer.util.createProgressDialog(
        labelText="Loading " + loadable.name,
        value=0,
        maximum=len(frames.frame_files),
        windowModality=qt.Qt.WindowModal,
    )
    try:
        for t, files in enumerate(frames.frame_files):
            progress.value = t
            slicer.app.processEvents()
            if progress.wasCanceled:
                raise ValueError("PET import was cancelled")
            with SceneTransaction():
                frame = _decode_frame(reader, files, f"PET time slice {t + 1}")
                source_uids.extend(frame.GetAttribute("DICOM.instanceUIDs").split())
                if t == 0:
                    color_id = frame.GetDisplayNode().GetColorNodeID()
                    if not sequence_mode:
                        result.SetIJKToRASMatrix(_matrix(frame))
                        result.SetVoxelValueQuantity(frame.GetVoxelValueQuantity())
                        result.SetVoxelValueUnits(frame.GetVoxelValueUnits())
                        image = vtk.vtkImageData()
                        image.SetExtent(frame.GetImageData().GetExtent())
                        image.AllocateScalars(vtk.VTK_DOUBLE, len(frames.frame_files))
                        image_array = vtk_to_numpy(image.GetPointData().GetScalars())
                if sequence_mode:
                    # SetDataNodeAtValue deep-copies pixels/attributes. Neither the
                    # temporary reader buffer nor later browser edits can alias it.
                    result.SetDataNodeAtValue(frame, frames.index_values[t])
                else:
                    image_array[:, t] = slicer.util.arrayFromVolume(frame).reshape(-1)
        with SceneTransaction() as publication:
            scene.AddNode(result)
            if sequence_mode:
                browser = scene.AddNewNodeByClass(
                    "vtkMRMLSequenceBrowserNode", result.GetName() + " browser"
                )
                browser.SetAndObserveMasterSequenceNodeID(result.GetID())
                browser.SetSaveChanges(result, False)
                browser.SetOverwriteProxyName(result, True)
                visible = browser.GetProxyNode(result)
                visible.CreateDefaultDisplayNodes()
                display = visible.GetDisplayNode()
                display.SetAndObserveColorNodeID(color_id)
                display.SetAutoWindowLevel(True)
                if slicer.modules.sequences.autoShowToolBar:
                    slicer.modules.sequences.setToolBarActiveBrowserNode(browser)
                    slicer.modules.sequences.setToolBarVisible(True)
            else:
                result.SetAndObserveImageData(image)
                result.SetNumberOfFrames(len(frames.frame_files))
                labels = vtk.vtkDoubleArray()
                for value in frames.index_values:
                    labels.InsertNextValue(float(value))
                result.SetLabelArray(labels)
                result.SetLabelName(frames.index_unit)
                result.SetAttribute("DICOM.instanceUIDs", " ".join(source_uids))
                display = scene.AddNewNodeByClass("vtkMRMLMultiVolumeDisplayNode")
                display.SetAndObserveColorNodeID(color_id)
                display.SetAutoWindowLevel(True)
                result.SetAndObserveDisplayNodeID(display.GetID())
                visible = result
            plugin.addSeriesInSubjectHierarchy(loadable, visible)
            slicer.util.setSliceViewerLayers(background=visible)
            selection.SetReferenceActiveVolumeID(visible.GetID())
            slicer.app.applicationLogic().PropagateVolumeSelection()
            publication.commit()
        return result
    finally:
        progress.close()
