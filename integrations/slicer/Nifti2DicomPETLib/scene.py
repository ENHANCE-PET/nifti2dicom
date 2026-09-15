"""Rollback synchronous native-reader side effects without clearing user data."""

import slicer
import vtk


def _hierarchy_items(hierarchy):
    items = vtk.vtkIdList()
    hierarchy.GetItemChildren(hierarchy.GetSceneItemID(), items, True)
    return [items.GetId(index) for index in range(items.GetNumberOfIds())]


class SceneTransaction:
    """Own newly created nodes/items; preserve the existing per-view selections.

    Do not process GUI events inside this scope: new nodes must belong to the
    synchronous operation, not unrelated user work. Explicit commit publishes
    them. Without commit, even a successful temporary read is rolled back.
    """

    def __enter__(self):
        self.committed = False
        self.scene = slicer.mrmlScene
        self.node_ids = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLNode")}
        self.hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(self.scene)
        self.item_ids = set(_hierarchy_items(self.hierarchy))
        self.selection = slicer.app.applicationLogic().GetSelectionNode()
        self.active = self.selection.GetActiveVolumeID()
        self.views = [
            (
                node,
                node.GetBackgroundVolumeID(),
                node.GetForegroundVolumeID(),
                node.GetLabelVolumeID(),
            )
            for node in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode")
        ]
        return self

    def commit(self):
        self.committed = True

    def __exit__(self, exception_type, exception, traceback):
        if self.committed and exception_type is None:
            return
        for node in reversed(slicer.util.getNodesByClass("vtkMRMLNode")):
            # Removing a volume can automatically remove its display/storage.
            if node.GetScene() == self.scene and node.GetID() not in self.node_ids:
                self.scene.RemoveNode(node)
        for item in reversed(_hierarchy_items(self.hierarchy)):
            if item not in self.item_ids:
                self.hierarchy.RemoveItem(item, False, False)
        self.selection.SetReferenceActiveVolumeID(self.active)
        for node, background, foreground, label in self.views:
            if node.GetScene() == self.scene:
                node.SetBackgroundVolumeID(background)
                node.SetForegroundVolumeID(foreground)
                node.SetLabelVolumeID(label)
