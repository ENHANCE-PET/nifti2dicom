"""External Slicer adapter for native classic dynamic PET import.

Install this directory as an additional module path. The signed Slicer bundle
and nifti2dicom conversion engine are deliberately not modified.
"""

import logging

import pydicom
import qt
import slicer
from DICOMLib import DICOMLoadable
from MultiVolumeImporterPlugin import MultiVolumeImporterPluginClass
from Nifti2DicomPETLib.frames import group_frames
from Nifti2DicomPETLib.loader import load_pet
from pydicom.errors import BytesLengthException, InvalidDicomError
from slicer.ScriptedLoadableModule import ScriptedLoadableModule

PET_STORAGE = "1.2.840.10008.5.1.4.1.1.128"


def _dynamic_pet_hint(dataset):
    """Claim malformed dynamic profiles too, so validation can explain them."""
    if str(dataset.get("SOPClassUID", "")) != PET_STORAGE:
        return False
    try:
        if "DYNAMIC" in str(dataset.get("SeriesType", "")):
            return True
        return int(dataset.get("NumberOfTimeSlices", 0)) > 1
    except (TypeError, ValueError, OverflowError, InvalidDicomError, BytesLengthException):
        # Undecodable PET profile metadata must reach validation, not a scalar
        # fallback. This is only ownership detection, never an acceptance rule.
        return True


class Nifti2DicomPETPlugin(MultiVolumeImporterPluginClass):
    """Own native dynamic PET; delegate other profiles to Slicer's importer."""

    def examineForImport(self, fileLists):
        return self.examine(fileLists)

    def examine(self, fileLists):
        loadables, other_lists = [], []
        for files in fileLists:
            records = []
            read_error = None
            dynamic = False
            for path in files:
                try:
                    dataset = pydicom.dcmread(path, stop_before_pixels=True)
                    records.append((path, dataset))
                    dynamic = _dynamic_pet_hint(dataset) or dynamic
                except (OSError, ValueError, InvalidDicomError, BytesLengthException) as error:
                    read_error = error
            if not dynamic:
                other_lists.append(files)
                continue
            try:
                if read_error:
                    raise ValueError(
                        "PET image headers could not all be read. Re-import the complete series."
                    ) from read_error
                frames = group_frames(records)
                preferred_sequence = (
                    qt.QSettings().value("DICOM/PreferredMultiVolumeImportFormat", "default")
                    != "multivolume"
                )
                for sequence in (preferred_sequence, not preferred_sequence):
                    loadable = DICOMLoadable()
                    loadable.files = list(files)
                    kind = "Volume Sequence" if sequence else "MultiVolume"
                    description = str(records[0][1].get("SeriesDescription", "Dynamic PET"))
                    loadable.name = f"{description}: {len(frames.frame_files)} frames {kind}"
                    loadable.tooltip = (
                        "Native PET time slices; original spatial grid and decoded values."
                    )
                    loadable.confidence = 1.0 if sequence == preferred_sequence else 0.95
                    loadable.selected = sequence == preferred_sequence
                    loadable.loadAsVolumeSequence = sequence
                    loadable.petFrames = frames
                    loadables.append(loadable)
            except ValueError as error:
                loadable = DICOMLoadable()
                loadable.files = list(files)
                loadable.name = "Dynamic PET — cannot safely load"
                loadable.warning = str(error)
                loadable.tooltip = loadable.warning
                loadable.confidence, loadable.selected = 1.0, True
                loadable.petError = loadable.warning
                loadables.append(loadable)
        if other_lists:
            loadables.extend(super().examine(other_lists))
        return loadables

    def load(self, loadable):
        if hasattr(loadable, "petError"):
            logging.error(loadable.petError)
            return False
        if not hasattr(loadable, "petFrames"):
            return super().load(loadable)
        try:
            # Do not trust a stale examination if files changed before Load.
            frames = group_frames(
                [(path, pydicom.dcmread(path, stop_before_pixels=True)) for path in loadable.files]
            )
            return load_pet(self, loadable, frames)
        except Exception as error:
            loadable.warning = f"PET import stopped: {error}. No partial PET series was kept."
            logging.exception(loadable.warning)
            return False


class Nifti2DicomPET(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "NIfTI to DICOM PET integration"
        parent.categories = ["Developer Tools.DICOM Plugins"]
        parent.dependencies = ["DICOM", "MultiVolumeImporterPlugin", "Sequences"]
        parent.hidden = True
        parent.helpText = (
            "Precision-safe native classic dynamic PET import. See integrations/slicer/README.md."
        )
        parent.contributors = ["nifti2dicom contributors"]
        slicer.modules.dicomPlugins["MultiVolumeImporterPlugin"] = Nifti2DicomPETPlugin
