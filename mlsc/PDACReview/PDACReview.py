"""PDACReview — dashboard for the PDAC photon-counting CT lymph-node runs.

Standalone Slicer scripted module (add <repo>/mlsc/PDACReview to
Edit → Application Settings → Modules → Additional module paths). It is a
test-bed for the PCCT data and deliberately independent of the LNQ modules.

What it shows (Apache ECharts inside a qSlicerWebWidget):

  * Overview — one bar group per study (patient + day offset + case id):
    median segmented lymph-node volume per model across that study's series,
    with min / max, coefficient of variation and node counts in the tooltip
    and in the table below. Click a study to drill in.
  * Study — every axial series of that study side by side: total mL and node
    count per model, the series' phase / spectral kind / thickness / fraction
    of interpolated slices, the per-model agreement across series (CV, range),
    and per-series node lists. "Load" opens the series in this Slicer: CT +
    one segmentation per model (registry colors) + one probability map.
  * Geometry issues — every series whose staging flagged a geometry problem
    (missing / interpolated slices, irregular spacing, …), with Load buttons
    and a Copy button; also written to manifest/geometry_issues.csv.

Probability review (module panel, LNQReview-style): a combo box picks which
model's probability map is shown (default: the model with the largest
segmentation in that series; only one map is in memory at a time) and a
log-scaled slider sets the threshold — voxels above it are colored Inferno in
the slice views and a thin iso-band at the threshold is volume-rendered in
3D, both updating live. Works even for series where nothing was segmented,
so residual probability is visible.

Data comes from <root>/manifest/pdac_stats.json, produced by
<repo>/mlsc/pdac_stats.py; the "Compute stats" button runs it in-process
(SimpleITK ships with Slicer), so no separate Python environment is needed.
"""
import json
import logging
import math
import os
import sys

import ctk
import qt
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleWidget)

MLSC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MLSC_DIR not in sys.path:
    sys.path.insert(0, MLSC_DIR)

# Registry colors (lnq-segmenter/_registry.json) so the dashboard and the
# segmentations in the scene agree.
MODEL_COLORS = {
    "mediastinal-v1":    (200, 100, 230),
    "abdominopelvic-v1": (120, 220, 120),
    "axillary-v1":       (255, 150, 100),
    "inguinal-v1":       (240, 220, 60),
}
MODEL_SHORT = {"mediastinal-v1": "mediastinal", "abdominopelvic-v1": "abd/pelvic",
               "axillary-v1": "axillary", "inguinal-v1": "inguinal"}

# Log-scaled threshold slider, same range as LNQReview's ThresholdController.
LOG_MIN, LOG_MAX, SLIDER_TICKS = -5.0, 0.0, 1000
THRESHOLD_PRESETS = (0.001, 0.01, 0.1, 0.5)


def slider_to_threshold(value):
    frac = max(0.0, min(1.0, value / SLIDER_TICKS))
    return 10 ** (LOG_MIN + frac * (LOG_MAX - LOG_MIN))


def threshold_to_slider(threshold):
    if threshold <= 0:
        return 0
    frac = (math.log10(threshold) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return int(round(max(0.0, min(1.0, frac)) * SLIDER_TICKS))


class PDACReview(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "PDAC Review"
        parent.categories = ["LNQ"]
        parent.dependencies = []
        parent.contributors = ["Steve Pieper (Isomics)"]
        parent.helpText = __doc__
        parent.acknowledgementText = "PDAC PCCT test cohort; LNQ models via lnq-segmenter."


# =============================================================================
# Logic
# =============================================================================

class PDACReviewLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.stats = None
        self.statsPath = None
        self._byVolume = {}
        # current scene state
        self.currentVolumeId = None
        self.ctNode = None
        self.segNodes = {}
        self.probNode = None
        self.probModel = None
        self.probVRDisplayNode = None
        self.threshold = 0.01

    # ---- stats -----------------------------------------------------------
    def statsFile(self, root):
        return os.path.join(root, "manifest", "pdac_stats.json")

    def loadStats(self, root):
        path = self.statsFile(root)
        if not os.path.isfile(path):
            self.stats = None
            self._byVolume = {}
            return None
        with open(path) as f:
            self.stats = json.load(f)
        self.statsPath = path
        self._byVolume = {v["volume_id"]: v for v in self.stats.get("volumes", [])}
        return self.stats

    def computeStats(self, root, min_node_ml, force=False, progress=None):
        import pdac_stats  # <repo>/mlsc/pdac_stats.py
        self.stats = pdac_stats.compute(root, min_node_ml=min_node_ml, force=force,
                                        progress=progress)
        self.statsPath = self.statsFile(root)
        self._byVolume = {v["volume_id"]: v for v in self.stats.get("volumes", [])}
        return self.stats

    def volume(self, volume_id):
        return self._byVolume.get(volume_id)

    def geometryIssues(self):
        if not self.stats:
            return []
        return [v for v in self.stats["volumes"] if v.get("flags")]

    # ---- scene -----------------------------------------------------------
    def clearScene(self):
        self.probVRDisplayNode = None
        self.probNode = None
        self.ctNode = None
        self.segNodes = {}
        for cls in ("vtkMRMLVolumeRenderingDisplayNode", "vtkMRMLSegmentationNode",
                    "vtkMRMLScalarVolumeNode"):
            for n in slicer.util.getNodesByClass(cls):
                slicer.mrmlScene.RemoveNode(n)

    def defaultProbModel(self, rec):
        """Model with the largest segmentation in this series (ties → first
        in the stats' model order); falls back to the first with a map."""
        best, best_ml = None, -1.0
        for m in self.stats["models"]:
            e = rec["models"].get(m)
            if not e or not e.get("prob_path"):
                continue
            if e.get("total_ml", 0) > best_ml:
                best, best_ml = m, e.get("total_ml", 0)
        return best

    def loadSeries(self, volume_id, prob_model=None):
        """Load one series: CT + SEG per model + the chosen probability map."""
        rec = self.volume(volume_id)
        if rec is None:
            raise KeyError(f"{volume_id} not in {self.statsPath}")
        ct_path = rec["ct_path"]
        if not os.path.isfile(ct_path):
            raise FileNotFoundError(ct_path)
        self.clearScene()
        self.currentVolumeId = volume_id
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        ct = slicer.util.loadVolume(ct_path)
        ct.SetName(f"PDAC:{volume_id}")
        disp = ct.GetDisplayNode()
        if disp is not None:
            disp.SetAutoWindowLevel(False)
            disp.SetWindowLevel(400, 40)
        self.ctNode = ct
        for m in self.stats["models"]:
            entry = rec["models"].get(m)
            if not entry or not os.path.isfile(entry.get("seg_path", "")):
                continue
            if entry.get("n_voxels", 0) == 0:
                continue      # empty labelmap → nothing to show
            node = slicer.util.loadSegmentation(entry["seg_path"])
            if node is None:
                continue
            node.SetName(f"PDAC:{m}")
            segmentation = node.GetSegmentation()
            color = tuple(c / 255.0 for c in MODEL_COLORS.get(m, (150, 150, 150)))
            if segmentation.GetNumberOfSegments():
                s = segmentation.GetSegment(segmentation.GetNthSegmentID(0))
                s.SetName(f"{MODEL_SHORT.get(m, m)} ({entry['n_nodes']} nodes, {entry['total_ml']} mL)")
                s.SetColor(*color)
            d = node.GetDisplayNode()
            if d is not None:
                d.SetVisibility2DFill(True)
                d.SetOpacity2DFill(0.35)
                d.SetVisibility2DOutline(True)
                d.SetVisibility3D(True)
                d.SetOpacity3D(0.5)
            try:
                node.CreateClosedSurfaceRepresentation()
            except Exception as exc:  # noqa: BLE001
                logging.warning("closed surface for %s: %s", m, exc)
            self.segNodes[m] = node
        layoutManager = slicer.app.layoutManager()
        for color in ("Red", "Yellow", "Green"):
            sw = layoutManager.sliceWidget(color)
            if sw is None:
                continue
            cn = sw.sliceLogic().GetSliceCompositeNode()
            cn.SetBackgroundVolumeID(ct.GetID())
            cn.SetForegroundVolumeID(None)
            cn.SetLinkedControl(True)
            sw.sliceLogic().FitSliceToAll()
        self.setProbabilityModel(prob_model or self.defaultProbModel(rec))
        # Jump slices to the largest node of the displayed model (or the
        # first model that has one).
        order = [self.probModel] + [m for m in self.stats["models"] if m != self.probModel]
        for m in order:
            nodes = (rec["models"].get(m) or {}).get("nodes") or []
            if nodes:
                lps = nodes[0]["centroid_lps"]
                slicer.modules.markups.logic().JumpSlicesToLocation(-lps[0], -lps[1], lps[2], True)
                break
        threeD = layoutManager.threeDWidget(0)
        if threeD is not None:
            threeD.threeDView().resetFocalPoint()
        return rec

    # ---- probability display ---------------------------------------------
    def setProbabilityModel(self, model):
        """Swap the displayed probability map (only one is kept in memory)."""
        rec = self.volume(self.currentVolumeId) if self.currentVolumeId else None
        if self.probVRDisplayNode is not None:
            slicer.mrmlScene.RemoveNode(self.probVRDisplayNode)
            self.probVRDisplayNode = None
        if self.probNode is not None:
            slicer.mrmlScene.RemoveNode(self.probNode)
            self.probNode = None
        self.probModel = model
        if rec is None or not model:
            return None
        entry = rec["models"].get(model) or {}
        path = entry.get("prob_path")
        if not path or not os.path.isfile(path):
            return None
        # show=False: otherwise loadVolume makes the map the slice background,
        # displacing the CT.
        node = slicer.util.loadVolume(path, properties={"show": False})
        if node is None:
            return None
        node.SetName(f"PDAC:{model}-prob")
        self.probNode = node
        heat = slicer.util.getFirstNodeByName("Inferno")
        d = node.GetDisplayNode()
        if d is not None and heat is not None:
            d.SetAndObserveColorNodeID(heat.GetID())
        layoutManager = slicer.app.layoutManager()
        for color in ("Red", "Yellow", "Green"):
            sw = layoutManager.sliceWidget(color)
            if sw is None:
                continue
            cn = sw.sliceLogic().GetSliceCompositeNode()
            if self.ctNode is not None:
                cn.SetBackgroundVolumeID(self.ctNode.GetID())
            cn.SetForegroundVolumeID(node.GetID())
            cn.SetForegroundOpacity(0.55)
        self.probVRDisplayNode = self._setupVolumeRendering(node)
        self.applyThreshold(self.threshold)
        return node

    def applyThreshold(self, threshold):
        """Slice-view threshold + 3D iso-band, updated together (hot)."""
        self.threshold = max(10 ** LOG_MIN, min(1.0, float(threshold)))
        if self.probNode is None:
            return
        d = self.probNode.GetDisplayNode()
        if d is not None:
            d.SetAutoWindowLevel(False)
            d.SetWindowLevelMinMax(self.threshold, 1.0)
            d.SetThreshold(self.threshold, 1.0)
            d.SetApplyThreshold(True)
        if self.probVRDisplayNode is not None:
            self._updateVRTransferFunction(self.probVRDisplayNode, self.threshold)

    @staticmethod
    def _setupVolumeRendering(prob_node):
        vrLogic = slicer.modules.volumerendering.logic()
        disp = vrLogic.GetFirstVolumeRenderingDisplayNode(prob_node)
        if disp is None:
            disp = vrLogic.CreateDefaultVolumeRenderingNodes(prob_node)
        if disp is None:
            return None
        disp.SetVisibility(True)
        return disp

    @staticmethod
    def _updateVRTransferFunction(disp, threshold):
        """Spike opacity half a decade wide (in log space) around the
        threshold, Inferno-ish colors — same iso-band idea as LNQReview so
        the 3D view shows the surface the slice threshold implies."""
        propNode = disp.GetVolumePropertyNode()
        prop = propNode.GetVolumeProperty() if propNode is not None else None
        if prop is None:
            return
        t = max(1e-5, min(0.999, float(threshold)))
        lo = max(1e-6, t / (10 ** 0.5))
        hi = min(1.0, t * (10 ** 0.5))
        opacity = prop.GetScalarOpacity()
        opacity.RemoveAllPoints()
        opacity.AddPoint(0.0, 0.0)
        opacity.AddPoint(lo, 0.0)
        opacity.AddPoint(t, 1.0)
        opacity.AddPoint(hi, 0.0)
        opacity.AddPoint(1.0, 0.0)
        rgb = prop.GetRGBTransferFunction()
        rgb.RemoveAllPoints()
        rgb.AddRGBPoint(0.0, 0.05, 0.03, 0.18)
        rgb.AddRGBPoint(lo, 0.40, 0.10, 0.40)
        rgb.AddRGBPoint(t, 1.00, 0.75, 0.10)
        rgb.AddRGBPoint(hi, 0.95, 0.55, 0.10)
        rgb.AddRGBPoint(1.0, 1.00, 0.95, 0.85)
        grad = prop.GetGradientOpacity()
        grad.RemoveAllPoints()
        grad.AddPoint(0.0, 1.0)
        grad.AddPoint(255.0, 1.0)
        prop.SetShade(True)
        prop.SetAmbient(0.35)
        prop.SetDiffuse(0.65)
        prop.SetSpecular(0.10)
        prop.SetInterpolationTypeToLinear()


# =============================================================================
# Widget
# =============================================================================

class PDACReviewWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PDACReviewLogic()
        self._dashboardWindow = None
        self._webWidget = None
        self._updatingCombo = False
        settings = qt.QSettings()

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        self.rootEdit = ctk.ctkPathLineEdit()
        self.rootEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.rootEdit.currentPath = settings.value("PDACReview/root", "/Volumes/12T/PHI/PDAC")
        self.rootEdit.setToolTip("Pipeline tree: contains manifest/, cohort/, E########/ …")
        form.addRow("Data root:", self.rootEdit)

        self.minNodeSpin = qt.QDoubleSpinBox()
        self.minNodeSpin.setRange(0.0, 5.0)
        self.minNodeSpin.setSingleStep(0.01)
        self.minNodeSpin.setDecimals(3)
        self.minNodeSpin.setValue(float(settings.value("PDACReview/minNodeMl", 0.05)))
        self.minNodeSpin.setToolTip("Connected components smaller than this are counted as specks, not nodes.")
        form.addRow("Min node (mL):", self.minNodeSpin)

        buttons = qt.QHBoxLayout()
        self.computeButton = qt.QPushButton("Compute stats")
        self.computeButton.setToolTip("Run pdac_stats.py over every volume (cached per volume; only new SEGs are recomputed).")
        self.forceCheck = qt.QCheckBox("force")
        self.dashboardButton = qt.QPushButton("Open dashboard")
        buttons.addWidget(self.computeButton)
        buttons.addWidget(self.forceCheck)
        buttons.addWidget(self.dashboardButton)
        self.layout.addLayout(buttons)

        self.progress = qt.QProgressBar()
        self.progress.visible = False
        self.layout.addWidget(self.progress)
        self.statusLabel = qt.QLabel("")
        self.statusLabel.wordWrap = True
        self.layout.addWidget(self.statusLabel)

        # ---- probability review (LNQReview-style) ----
        probBox = ctk.ctkCollapsibleButton()
        probBox.text = "Probability map"
        self.layout.addWidget(probBox)
        pl = qt.QVBoxLayout(probBox)
        self.loadedLabel = qt.QLabel("(no series loaded)")
        self.loadedLabel.wordWrap = True
        pl.addWidget(self.loadedLabel)
        row = qt.QHBoxLayout()
        row.addWidget(qt.QLabel("Show:"))
        self.probCombo = qt.QComboBox()
        self.probCombo.setToolTip("Which model's probability map to display (one at a time; "
                                  "default = largest segmentation in this series).")
        row.addWidget(self.probCombo, 1)
        pl.addLayout(row)
        row = qt.QHBoxLayout()
        self.thresholdSlider = qt.QSlider(qt.Qt.Horizontal)
        self.thresholdSlider.setMinimum(0)
        self.thresholdSlider.setMaximum(SLIDER_TICKS)
        self.thresholdSlider.setToolTip("Log-scaled probability threshold: 1e-5 … 1. "
                                        "Colors voxels ≥ p in the slices and volume-renders the p iso-band in 3D.")
        self.logic.threshold = float(settings.value("PDACReview/threshold", 0.01))
        self.thresholdSlider.setValue(threshold_to_slider(self.logic.threshold))
        self.thresholdLabel = qt.QLabel(f"p ≥ {self.logic.threshold:.4g}")
        self.thresholdLabel.setMinimumWidth(90)
        row.addWidget(self.thresholdSlider, 1)
        row.addWidget(self.thresholdLabel)
        pl.addLayout(row)
        presets = qt.QHBoxLayout()
        for p in THRESHOLD_PRESETS:
            b = qt.QPushButton(f"{p:g}")
            b.setToolTip(f"Set threshold to p ≥ {p:g}")
            b.connect("clicked()", lambda p=p: self.thresholdSlider.setValue(threshold_to_slider(p)))
            presets.addWidget(b)
        self.probVisibleCheck = qt.QCheckBox("show")
        self.probVisibleCheck.checked = True
        presets.addWidget(self.probVisibleCheck)
        pl.addLayout(presets)
        self.layout.addStretch(1)

        self.computeButton.connect("clicked()", self.onCompute)
        self.dashboardButton.connect("clicked()", self.onOpenDashboard)
        self.rootEdit.connect("currentPathChanged(QString)", self._onRootChanged)
        self.thresholdSlider.connect("valueChanged(int)", self._onThresholdChanged)
        self.probCombo.connect("currentIndexChanged(int)", self._onProbComboChanged)
        self.probVisibleCheck.connect("toggled(bool)", self._onProbVisible)
        self._refreshStatus()

    # ---- helpers ---------------------------------------------------------
    def root(self):
        return self.rootEdit.currentPath

    def _onRootChanged(self, path):
        qt.QSettings().setValue("PDACReview/root", path)
        self._refreshStatus()

    def _refreshStatus(self):
        stats = self.logic.loadStats(self.root())
        if stats is None:
            self.statusLabel.text = ("No manifest/pdac_stats.json under this root yet — "
                                     "click Compute stats.")
            return
        n_issues = len(self.logic.geometryIssues())
        self.statusLabel.text = (f"{stats['n_patients']} patients, {stats['n_studies']} studies, "
                                 f"{stats['n_volumes']} volumes, models: {', '.join(stats['models'])} "
                                 f"(computed {stats['generated_at']}, min node {stats['min_node_ml']} mL; "
                                 f"{n_issues} series with geometry flags)")

    def onCompute(self):
        root = self.root()
        if not os.path.isfile(os.path.join(root, "manifest", "volumes.csv")):
            slicer.util.errorDisplay(f"No manifest/volumes.csv under {root}")
            return
        qt.QSettings().setValue("PDACReview/minNodeMl", self.minNodeSpin.value)
        self.progress.visible = True
        self.progress.setValue(0)
        self.computeButton.enabled = False

        def progress(i, n, vid):
            self.progress.setMaximum(n)
            self.progress.setValue(i)
            self.statusLabel.text = f"[{i}/{n}] {vid}"
            slicer.app.processEvents()

        try:
            with slicer.util.tryWithErrorDisplay("Stats computation failed", waitCursor=True):
                self.logic.computeStats(root, self.minNodeSpin.value, force=self.forceCheck.checked,
                                        progress=progress)
        finally:
            self.progress.visible = False
            self.computeButton.enabled = True
        self._refreshStatus()
        if self._dashboardWindow is not None and self._dashboardWindow.visible:
            self.renderDashboard()

    def onOpenDashboard(self):
        if self.logic.stats is None and self.logic.loadStats(self.root()) is None:
            slicer.util.errorDisplay("No stats yet — click Compute stats first.")
            return
        if self._dashboardWindow is None:
            win = qt.QWidget()
            win.setWindowTitle("PDAC Review dashboard")
            win.setWindowFlags(qt.Qt.Window)
            lay = qt.QVBoxLayout(win)
            lay.setContentsMargins(0, 0, 0, 0)
            self._webWidget = slicer.qSlicerWebWidget()
            lay.addWidget(self._webWidget)
            win.resize(1280, 900)
            self._dashboardWindow = win
        self.renderDashboard()
        self._dashboardWindow.show()
        self._dashboardWindow.raise_()

    def renderDashboard(self):
        html = build_dashboard_html(self.logic.stats)
        self._webWidget.setHtml(html)

    # ---- probability controls -------------------------------------------
    def _onThresholdChanged(self, value):
        t = slider_to_threshold(value)
        self.thresholdLabel.text = f"p ≥ {t:.4g}"
        qt.QSettings().setValue("PDACReview/threshold", t)
        self.logic.applyThreshold(t)

    def _onProbComboChanged(self, index):
        if self._updatingCombo or index < 0:
            return
        model = self.probCombo.itemData(index)
        with slicer.util.tryWithErrorDisplay("Could not load probability map", waitCursor=True):
            self.logic.setProbabilityModel(model)
        self._onProbVisible(self.probVisibleCheck.checked)

    def _onProbVisible(self, on):
        lm = slicer.app.layoutManager()
        for color in ("Red", "Yellow", "Green"):
            sw = lm.sliceWidget(color)
            if sw is not None:
                sw.sliceLogic().GetSliceCompositeNode().SetForegroundOpacity(0.55 if on else 0.0)
        if self.logic.probVRDisplayNode is not None:
            self.logic.probVRDisplayNode.SetVisibility(bool(on))

    def _fillProbCombo(self, rec):
        self._updatingCombo = True
        try:
            self.probCombo.clear()
            for m in self.logic.stats["models"]:
                e = rec["models"].get(m)
                if e and e.get("prob_path"):
                    self.probCombo.addItem(f"{MODEL_SHORT.get(m, m)}  ({e['total_ml']} mL, {e['n_nodes']} nodes)", m)
            for i in range(self.probCombo.count):
                if self.probCombo.itemData(i) == self.logic.probModel:
                    self.probCombo.setCurrentIndex(i)
                    break
        finally:
            self._updatingCombo = False

    # ---- called from JS via window.slicerPython.evalPython ---------------
    def loadSeries(self, volume_id, prob_model=None):
        try:
            with slicer.util.tryWithErrorDisplay(f"Could not load {volume_id}", waitCursor=True):
                rec = self.logic.loadSeries(volume_id, prob_model=prob_model)
            self._fillProbCombo(rec)
            self._onProbVisible(self.probVisibleCheck.checked)
            self.loadedLabel.text = (f"{rec['patient']} · {rec['study_id']} · #{rec['series_number']} "
                                     f"{rec['series_description']}"
                                     + (f"\n⚠ {rec['flags']}" if rec.get("flags") else ""))
            self.statusLabel.text = (f"Loaded {volume_id}: {len(self.logic.segNodes)} segmentations, "
                                     f"probability: {self.logic.probModel or 'none'}")
            slicer.util.mainWindow().raise_()
        except Exception as exc:  # noqa: BLE001
            logging.exception("loadSeries %s", volume_id)
            self.statusLabel.text = f"Load failed: {exc}"

    def copyGeometryIssues(self):
        """Put the geometry-issue list on the clipboard as tab-separated text."""
        lines = ["patient\tcase\tstudy\tseries\tdescription\tthickness_mm\tslices\tinterpolated\tflags"]
        for v in self.logic.geometryIssues():
            lines.append("\t".join(str(x) for x in (
                v["patient"], v["case_id"], v["study_id"], v["series_number"], v["series_description"],
                v["slice_thickness_mm"], v["n_slices"], f"{100 * v['missing_frac']:.0f}%", v["flags"])))
        qt.QApplication.clipboard().setText("\n".join(lines))
        self.statusLabel.text = f"Copied {len(lines) - 1} geometry-issue rows to the clipboard."


# =============================================================================
# Dashboard HTML (ECharts)
# =============================================================================

def _slim_volume(v):
    """Only what the page needs (keeps the embedded JSON small)."""
    out = {k: v.get(k) for k in ("volume_id", "case_id", "study_id", "patient", "day", "series_number",
                                 "series_description", "phase", "phase_seconds", "spectral",
                                 "kernel", "slice_thickness_mm", "n_slices", "missing_frac", "flags")}
    out["models"] = {}
    for m, e in v["models"].items():
        entry = {k: e.get(k) for k in ("total_ml", "n_nodes", "n_specks", "largest_ml", "nodes_ml", "qc")}
        entry["nodes"] = [{k: n[k] for k in ("ml", "short_mm", "long_mm")} for n in e.get("nodes", [])[:25]]
        out["models"][m] = entry
    return out


def build_dashboard_html(stats):
    data = {
        "generated_at": stats.get("generated_at"),
        "min_node_ml": stats.get("min_node_ml"),
        "root": stats.get("root"),
        "models": stats["models"],
        "colors": {m: "rgb(%d,%d,%d)" % MODEL_COLORS.get(m, (150, 150, 150)) for m in stats["models"]},
        "short": {m: MODEL_SHORT.get(m, m) for m in stats["models"]},
        "studies": stats["studies"],
        "volumes": [_slim_volume(v) for v in stats["volumes"]],
    }
    return DASHBOARD_TEMPLATE.replace("%%DATA%%", json.dumps(data))


DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PDAC Review</title>
<script src="https://fastly.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  :root { --bg:#fff; --fg:#222; --muted:#666; --line:#ddd; --hover:#f3f6fa; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1e1e1e; --fg:#e6e6e6; --muted:#aaa; --line:#444; --hover:#2a2f36; } }
  body { margin:0; font:13px/1.4 -apple-system, Helvetica, Arial, sans-serif; color:var(--fg); background:var(--bg); }
  header { display:flex; flex-wrap:wrap; gap:14px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  header .meta { color:var(--muted); }
  header .spacer { flex:1; }
  .filters { display:flex; flex-wrap:wrap; gap:10px 16px; padding:8px 16px; border-bottom:1px solid var(--line); align-items:center; }
  .filters label { display:inline-flex; align-items:center; gap:4px; }
  .swatch { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:2px; }
  .chart { height:320px; }
  .row { display:flex; gap:12px; padding:0 8px; }
  .row .chart { flex:1; }
  main { padding: 8px 8px 24px; }
  table { border-collapse:collapse; width:100%; margin:8px 0; }
  th, td { padding:4px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
  th:first-child, td:first-child, td.left, th.left { text-align:left; }
  tr.clickable { cursor:pointer; }
  tr.clickable:hover { background:var(--hover); }
  th { position:sticky; top:0; background:var(--bg); }
  button { font:inherit; padding:2px 8px; cursor:pointer; }
  .back { color:#3a7bd5; cursor:pointer; }
  .warn { color:#c0392b; }
  .muted { color:var(--muted); }
  details { margin:2px 0 6px 0; }
  .nodes { font-size:12px; color:var(--muted); }
  .agree td { text-align:right; }
  code { font-size:12px; }
</style></head>
<body>
<header>
  <h1>PDAC Review</h1>
  <span id="crumb"></span>
  <span class="meta" id="meta"></span>
  <span class="spacer"></span>
  <button id="geomBtn" title="Series whose staging flagged geometry problems"></button>
</header>
<div class="filters" id="filters"></div>
<main id="main"></main>
<script>
const D = %%DATA%%;
const M = D.models;
const fmt = (x, d=2) => (x === null || x === undefined || x === "") ? "" : Number(x).toFixed(d);
const pct = x => (100 * x).toFixed(0) + "%";
const sid = v => v.study_id || v.case_id;
let state = { view: "overview", study: null, models: new Set(M), phases: new Set(), spectral: new Set(),
              thick: new Set(), maxMissing: 1.0 };
const charts = [];
function disposeCharts() { while (charts.length) charts.pop().dispose(); }
function mkChart(el, option) { const c = echarts.init(el, null, {renderer: "canvas"}); c.setOption(option); charts.push(c); return c; }
window.addEventListener("resize", () => charts.forEach(c => c.resize()));

const allPhases = [...new Set(D.volumes.map(v => v.phase || "?"))].sort((a,b) => (phaseSec(a)-phaseSec(b)));
const allSpectral = [...new Set(D.volumes.map(v => v.spectral))].sort();
const allThick = [...new Set(D.volumes.map(v => v.slice_thickness_mm))].sort((a,b)=>a-b);
state.phases = new Set(allPhases); state.spectral = new Set(allSpectral); state.thick = new Set(allThick);
function phaseSec(p) { const v = D.volumes.find(v => (v.phase || "?") === p); return v ? v.phase_seconds : 1e9; }
const issues = D.volumes.filter(v => v.flags);

function seriesLabel(v) {
  return `#${v.series_number} ${v.phase || ""} ${v.spectral === "monoenergetic" ? (v.series_description.match(/\d+\s*keV/) || ["keV"])[0] : v.spectral} ${v.slice_thickness_mm}mm`;
}
function passesFilters(v) {
  return state.phases.has(v.phase || "?") && state.spectral.has(v.spectral) && state.thick.has(v.slice_thickness_mm) && v.missing_frac <= state.maxMissing;
}
function volumesOf(studyId) {
  return D.volumes.filter(v => sid(v) === studyId && passesFilters(v)).sort((a,b) => a.series_number - b.series_number);
}
function activeModels() { return M.filter(m => state.models.has(m)); }
function median(a) { if (!a.length) return null; const s=[...a].sort((x,y)=>x-y); const h=s.length>>1; return s.length%2 ? s[h] : (s[h-1]+s[h])/2; }
function cv(a) { if (a.length < 2) return 0; const mean = a.reduce((x,y)=>x+y,0)/a.length; if (!mean) return 0; const sd = Math.sqrt(a.reduce((x,y)=>x+(y-mean)**2,0)/a.length); return sd/mean; }

function renderFilters() {
  const f = document.getElementById("filters");
  const chk = (group, val, label, on) => `<label><input type="checkbox" data-group="${group}" data-val="${val}" ${on ? "checked" : ""}>${label}</label>`;
  f.innerHTML =
    `<span class="muted">Models:</span> ` + M.map(m => chk("models", m, `<span class="swatch" style="background:${D.colors[m]}"></span>${D.short[m]}`, state.models.has(m))).join(" ") +
    ` <span class="muted">| Phase:</span> ` + allPhases.map(p => chk("phases", p, p, state.phases.has(p))).join(" ") +
    ` <span class="muted">| Kind:</span> ` + allSpectral.map(s => chk("spectral", s, s, state.spectral.has(s))).join(" ") +
    ` <span class="muted">| Thickness:</span> ` + allThick.map(t => chk("thick", t, t + " mm", state.thick.has(t))).join(" ") +
    ` <span class="muted">| Max interpolated:</span> <select id="maxMissing">` +
      [0, 0.1, 0.25, 0.5, 1.0].map(x => `<option value="${x}" ${x === state.maxMissing ? "selected" : ""}>${x === 1 ? "any" : "≤ " + pct(x)}</option>`).join("") + `</select>`;
  f.querySelectorAll("input[type=checkbox][data-group]").forEach(el => el.addEventListener("change", e => {
    const g = e.target.dataset.group; let val = e.target.dataset.val; if (g === "thick") val = Number(val);
    if (e.target.checked) state[g].add(val); else state[g].delete(val); render();
  }));
  f.querySelector("#maxMissing").addEventListener("change", e => { state.maxMissing = Number(e.target.value); render(); });
}

function studyRows() {
  const studies = [...new Set(D.volumes.map(sid))];
  const rows = studies.map(c => {
    const vols = volumesOf(c); if (!vols.length) return null;
    const v0 = D.volumes.find(v => sid(v) === c);
    const r = { case_id: c, patient: v0.patient, day: v0.day, n: vols.length, models: {} };
    for (const m of activeModels()) {
      const ml = vols.filter(v => v.models[m]).map(v => v.models[m].total_ml);
      const nn = vols.filter(v => v.models[m]).map(v => v.models[m].n_nodes);
      r.models[m] = { median_ml: median(ml), min_ml: Math.min(...ml), max_ml: Math.max(...ml), cv_ml: cv(ml),
                      median_nodes: median(nn), min_nodes: Math.min(...nn), max_nodes: Math.max(...nn), n: ml.length };
    }
    return r;
  }).filter(Boolean);
  rows.sort((a,b) => a.patient.localeCompare(b.patient) || a.day - b.day || a.case_id.localeCompare(b.case_id));
  return rows;
}

function renderOverview() {
  document.getElementById("crumb").innerHTML = `<span class="muted">overview</span>`;
  const rows = studyRows();
  const main = document.getElementById("main");
  main.innerHTML = `<div class="row"><div id="c1" class="chart"></div><div id="c2" class="chart"></div></div><div id="tbl"></div>`;
  const labels = rows.map(r => `${r.patient} d${r.day}\n${r.case_id}`);
  const mk = (key, title, unit) => ({
    title: { text: title, left: "center", textStyle: { fontSize: 13 } },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, formatter: ps => {
      const r = rows[ps[0].dataIndex]; let s = `<b>${r.patient}</b> day ${r.day} · ${r.case_id} · ${r.n} series<br>`;
      for (const p of ps) { const m = p.seriesId, x = r.models[m]; if (!x) continue;
        s += `${p.marker}${D.short[m]}: median ${fmt(x.median_ml)} mL [${fmt(x.min_ml)}–${fmt(x.max_ml)}], CV ${pct(x.cv_ml)}, nodes ${x.median_nodes} [${x.min_nodes}–${x.max_nodes}]<br>`; }
      return s; } },
    legend: { bottom: 0, data: activeModels().map(m => D.short[m]) },
    grid: { left: 50, right: 16, top: 30, bottom: 70 },
    xAxis: { type: "category", data: labels, axisLabel: { interval: 0, rotate: rows.length > 8 ? 45 : 0, fontSize: 10 } },
    yAxis: { type: "value", name: unit },
    series: activeModels().map(m => ({ id: m, name: D.short[m], type: "bar", itemStyle: { color: D.colors[m] },
      data: rows.map(r => r.models[m] ? r.models[m][key] : null) })),
  });
  const c1 = mkChart(document.getElementById("c1"), mk("median_ml", "Median segmented lymph-node volume per study", "mL"));
  const c2 = mkChart(document.getElementById("c2"), mk("median_nodes", "Median node count per study", "nodes"));
  const go = p => { if (p.componentType === "series") openStudy(rows[p.dataIndex].case_id); };
  c1.on("click", go); c2.on("click", go);
  let h = `<table><tr><th class="left">patient</th><th>day</th><th class="left">study</th><th>series</th>`;
  for (const m of activeModels()) h += `<th><span class="swatch" style="background:${D.colors[m]}"></span>${D.short[m]} mL (min–max)</th><th>nodes</th>`;
  h += `</tr>`;
  for (const r of rows) {
    h += `<tr class="clickable" data-case="${r.case_id}"><td class="left">${r.patient}</td><td>${r.day}</td><td class="left">${r.case_id}</td><td>${r.n}</td>`;
    for (const m of activeModels()) { const x = r.models[m];
      h += x ? `<td>${fmt(x.median_ml)} <span class="muted">(${fmt(x.min_ml)}–${fmt(x.max_ml)})</span></td><td>${x.median_nodes} <span class="muted">(${x.min_nodes}–${x.max_nodes})</span></td>` : `<td></td><td></td>`; }
    h += `</tr>`;
  }
  h += `</table>`;
  document.getElementById("tbl").innerHTML = h;
  main.querySelectorAll("tr.clickable").forEach(tr => tr.addEventListener("click", () => openStudy(tr.dataset.case)));
}

function openStudy(studyId) { state.view = "study"; state.study = studyId; render(); }

function loadButton(v) { return `<button data-vid="${v.volume_id}" title="Load CT + segmentations + probability map in Slicer">Load</button>`; }
function wireLoadButtons(root) { root.querySelectorAll("button[data-vid]").forEach(b => b.addEventListener("click", () => loadInSlicer(b.dataset.vid))); }

function renderStudy() {
  const vols = volumesOf(state.study);
  const v0 = D.volumes.find(v => sid(v) === state.study);
  document.getElementById("crumb").innerHTML = `<span class="back" id="back">← overview</span> &nbsp; <b>${v0.patient}</b> day ${v0.day} · ${state.study}`;
  document.getElementById("back").addEventListener("click", () => { state.view = "overview"; render(); });
  const main = document.getElementById("main");
  main.innerHTML = `<div class="row"><div id="c1" class="chart"></div><div id="c2" class="chart"></div></div><div id="agree"></div><div id="tbl"></div>`;
  const labels = vols.map(seriesLabel);
  const mk = (key, title, unit) => ({
    title: { text: title, left: "center", textStyle: { fontSize: 13 } },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { bottom: 0, data: activeModels().map(m => D.short[m]) },
    grid: { left: 50, right: 16, top: 30, bottom: 80 },
    xAxis: { type: "category", data: labels, axisLabel: { interval: 0, rotate: 30, fontSize: 10 } },
    yAxis: { type: "value", name: unit },
    series: activeModels().map(m => ({ id: m, name: D.short[m], type: "bar", itemStyle: { color: D.colors[m] },
      data: vols.map(v => v.models[m] ? v.models[m][key] : null) })),
  });
  const c1 = mkChart(document.getElementById("c1"), mk("total_ml", "Segmented volume per series", "mL"));
  const c2 = mkChart(document.getElementById("c2"), mk("n_nodes", "Node count per series", "nodes"));
  const go = p => { if (p.componentType === "series") loadInSlicer(vols[p.dataIndex].volume_id, p.seriesId); };
  c1.on("click", go); c2.on("click", go);
  let a = `<table class="agree"><tr><th class="left">agreement across ${vols.length} series</th><th>median mL</th><th>min</th><th>max</th><th>CV mL</th><th>median nodes</th><th>min</th><th>max</th><th>CV nodes</th></tr>`;
  for (const m of activeModels()) {
    const ml = vols.filter(v => v.models[m]).map(v => v.models[m].total_ml), nn = vols.filter(v => v.models[m]).map(v => v.models[m].n_nodes);
    if (!ml.length) continue;
    a += `<tr><td class="left"><span class="swatch" style="background:${D.colors[m]}"></span>${D.short[m]}</td><td>${fmt(median(ml))}</td><td>${fmt(Math.min(...ml))}</td><td>${fmt(Math.max(...ml))}</td><td class="${cv(ml) > 0.5 ? "warn" : ""}">${pct(cv(ml))}</td><td>${median(nn)}</td><td>${Math.min(...nn)}</td><td>${Math.max(...nn)}</td><td class="${cv(nn) > 0.5 ? "warn" : ""}">${pct(cv(nn))}</td></tr>`;
  }
  a += `</table>`;
  document.getElementById("agree").innerHTML = a;
  let h = `<table><tr><th></th><th>#</th><th class="left">series</th><th>phase</th><th>kind</th><th>thk</th><th>slices</th><th>interp.</th>`;
  for (const m of activeModels()) h += `<th><span class="swatch" style="background:${D.colors[m]}"></span>${D.short[m]} mL</th><th>nodes</th><th>largest</th>`;
  h += `</tr>`;
  for (const v of vols) {
    h += `<tr><td>${loadButton(v)}</td><td>${v.series_number}</td><td class="left" title="${v.flags}">${v.series_description}${v.flags ? ' <span class="warn" title="' + v.flags + '">⚠</span>' : ""}</td><td>${v.phase}</td><td>${v.spectral}</td><td>${v.slice_thickness_mm}</td><td>${v.n_slices}</td><td class="${v.missing_frac > 0.25 ? "warn" : ""}">${v.missing_frac ? pct(v.missing_frac) : ""}</td>`;
    for (const m of activeModels()) { const e = v.models[m];
      h += e ? `<td>${fmt(e.total_ml)}</td><td>${e.n_nodes}${e.n_specks ? `<span class="muted"> +${e.n_specks} specks</span>` : ""}</td><td>${fmt(e.largest_ml)}</td>` : `<td></td><td></td><td></td>`; }
    h += `</tr>`;
    const nodeLists = activeModels().filter(m => v.models[m] && v.models[m].nodes.length).map(m =>
      `<span class="swatch" style="background:${D.colors[m]}"></span>${D.short[m]}: ` + v.models[m].nodes.map(n => `${fmt(n.ml)} mL (${n.short_mm}×${n.long_mm} mm)`).join(", "));
    if (nodeLists.length) h += `<tr><td></td><td colspan="${7 + 3 * activeModels().length}" class="left nodes">${nodeLists.join("<br>")}</td></tr>`;
  }
  h += `</table>`;
  document.getElementById("tbl").innerHTML = h;
  wireLoadButtons(main);
}

function renderGeometry() {
  document.getElementById("crumb").innerHTML = `<span class="back" id="back">← overview</span> &nbsp; <b>geometry issues</b> · ${issues.length} of ${D.volumes.length} series`;
  document.getElementById("back").addEventListener("click", () => { state.view = "overview"; render(); });
  const main = document.getElementById("main");
  const counts = {};
  for (const v of issues) for (const f of v.flags.split(";")) { const k = f.split("=")[0]; counts[k] = (counts[k] || 0) + 1; }
  let h = `<p class="muted">Series whose DICOM → NRRD staging flagged a geometry problem (best-effort volumes were still written; missing slices were interpolated). ` +
          `Flag counts: ${Object.entries(counts).map(([k, n]) => `${k} ${n}`).join(", ")}. ` +
          `Also in <code>${D.root}/manifest/geometry_issues.csv</code>. <button id="copyGeom">Copy list</button></p>`;
  h += `<table><tr><th></th><th class="left">patient</th><th class="left">study</th><th>#</th><th class="left">series</th><th>thk</th><th>slices</th><th>interp.</th><th class="left">flags</th></tr>`;
  const rows = [...issues].sort((a, b) => b.missing_frac - a.missing_frac || a.patient.localeCompare(b.patient) || a.series_number - b.series_number);
  for (const v of rows) {
    h += `<tr><td>${loadButton(v)}</td><td class="left">${v.patient}</td><td class="left">${sid(v)}</td><td>${v.series_number}</td><td class="left">${v.series_description}</td><td>${v.slice_thickness_mm}</td><td>${v.n_slices}</td><td class="${v.missing_frac > 0.25 ? "warn" : ""}">${v.missing_frac ? pct(v.missing_frac) : ""}</td><td class="left">${v.flags.split(";").join(" · ")}</td></tr>`;
  }
  h += `</table>`;
  main.innerHTML = h;
  wireLoadButtons(main);
  main.querySelector("#copyGeom").addEventListener("click", () => {
    if (window.slicerPython) window.slicerPython.evalPython(`slicer.modules.PDACReviewWidget.copyGeometryIssues()`);
  });
}

function loadInSlicer(vid, model) {
  if (!window.slicerPython) { alert("Not running inside Slicer (window.slicerPython missing)."); return; }
  const arg = model ? `, '${model}'` : "";
  window.slicerPython.evalPython(`slicer.modules.PDACReviewWidget.loadSeries('${vid}'${arg})`);
}

function render() {
  disposeCharts();
  document.getElementById("meta").textContent = `${D.studies.length} studies · ${D.volumes.length} volumes · nodes ≥ ${D.min_node_ml} mL · ${D.generated_at}`;
  const gb = document.getElementById("geomBtn");
  gb.textContent = `⚠ Geometry issues (${issues.length})`;
  gb.onclick = () => { state.view = "geometry"; render(); };
  renderFilters();
  if (state.view === "study") renderStudy(); else if (state.view === "geometry") renderGeometry(); else renderOverview();
}
render();
</script>
</body></html>
"""
