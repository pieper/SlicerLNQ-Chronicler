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
    one segmentation per model (registry colors) + optionally the probability
    maps as an Inferno overlay.

Data comes from <root>/manifest/pdac_stats.json, produced by
<repo>/mlsc/pdac_stats.py; the "Compute stats" button runs it in-process
(SimpleITK ships with Slicer), so no separate Python environment is needed.
"""
import json
import logging
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

    # ---- scene -----------------------------------------------------------
    def clearScene(self):
        for cls in ("vtkMRMLVolumeRenderingDisplayNode", "vtkMRMLSegmentationNode",
                    "vtkMRMLScalarVolumeNode"):
            for n in slicer.util.getNodesByClass(cls):
                slicer.mrmlScene.RemoveNode(n)

    def loadSeries(self, volume_id, models=None, with_prob=False):
        """Load one series: CT + SEG per model (+ probability maps)."""
        rec = self.volume(volume_id)
        if rec is None:
            raise KeyError(f"{volume_id} not in {self.statsPath}")
        ct_path = rec["ct_path"]
        if not os.path.isfile(ct_path):
            raise FileNotFoundError(ct_path)
        self.clearScene()
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        ct = slicer.util.loadVolume(ct_path)
        ct.SetName(f"PDAC:{volume_id}")
        disp = ct.GetDisplayNode()
        if disp is not None:
            disp.SetAutoWindowLevel(False)
            disp.SetWindowLevel(400, 40)
        seg_nodes = []
        prob_node = None
        for m in (models or rec["models"].keys()):
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
            seg_nodes.append(node)
            if with_prob and prob_node is None and entry.get("prob_path") \
                    and os.path.isfile(entry["prob_path"]):
                prob_node = slicer.util.loadVolume(entry["prob_path"])
                if prob_node is not None:
                    prob_node.SetName(f"PDAC:{m}-prob")
        layoutManager = slicer.app.layoutManager()
        for color in ("Red", "Yellow", "Green"):
            sw = layoutManager.sliceWidget(color)
            if sw is None:
                continue
            cn = sw.sliceLogic().GetSliceCompositeNode()
            cn.SetBackgroundVolumeID(ct.GetID())
            cn.SetForegroundVolumeID(prob_node.GetID() if prob_node else None)
            if prob_node:
                cn.SetForegroundOpacity(0.5)
            cn.SetLinkedControl(True)
            sw.sliceLogic().FitSliceToAll()
        if prob_node is not None:
            pd = prob_node.GetDisplayNode()
            heat = slicer.util.getFirstNodeByName("Inferno")
            if pd is not None and heat is not None:
                pd.SetAndObserveColorNodeID(heat.GetID())
                pd.SetAutoWindowLevel(False)
                pd.SetWindowLevel(1.0, 0.5)
                pd.SetThreshold(0.05, 1.0)
                pd.SetApplyThreshold(True)
        # Jump slices to the largest node of the first model that has one.
        for m in (models or rec["models"].keys()):
            nodes = (rec["models"].get(m) or {}).get("nodes") or []
            if nodes:
                lps = nodes[0]["centroid_lps"]
                ras = (-lps[0], -lps[1], lps[2])
                slicer.modules.markups.logic().JumpSlicesToLocation(*ras, True)
                break
        threeD = layoutManager.threeDWidget(0)
        if threeD is not None:
            threeD.threeDView().resetFocalPoint()
        return {"ct": ct, "segmentations": seg_nodes, "prob": prob_node}


# =============================================================================
# Widget
# =============================================================================

class PDACReviewWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = PDACReviewLogic()
        self._dashboardWindow = None
        self._webWidget = None
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

        self.probCheck = qt.QCheckBox("Load probability map as overlay")
        self.probCheck.checked = settings.value("PDACReview/loadProb", "false") == "true"
        form.addRow("", self.probCheck)

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
        self.layout.addStretch(1)

        self.computeButton.connect("clicked()", self.onCompute)
        self.dashboardButton.connect("clicked()", self.onOpenDashboard)
        self.rootEdit.connect("currentPathChanged(QString)", self._onRootChanged)
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
        self.statusLabel.text = (f"{stats['n_patients']} patients, {stats['n_studies']} studies, "
                                 f"{stats['n_volumes']} volumes, models: {', '.join(stats['models'])} "
                                 f"(computed {stats['generated_at']}, min node {stats['min_node_ml']} mL)")

    def onCompute(self):
        root = self.root()
        if not os.path.isfile(os.path.join(root, "manifest", "volumes.csv")):
            slicer.util.errorDisplay(f"No manifest/volumes.csv under {root}")
            return
        settings = qt.QSettings()
        settings.setValue("PDACReview/minNodeMl", self.minNodeSpin.value)
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
        html = build_dashboard_html(self.logic.stats, self.probCheck.checked)
        self._webWidget.setHtml(html)

    # ---- called from JS via window.slicerPython.evalPython ---------------
    def loadSeries(self, volume_id, with_prob=None):
        if with_prob is None:
            with_prob = self.probCheck.checked
        qt.QSettings().setValue("PDACReview/loadProb", "true" if with_prob else "false")
        try:
            with slicer.util.tryWithErrorDisplay(f"Could not load {volume_id}", waitCursor=True):
                nodes = self.logic.loadSeries(volume_id, with_prob=with_prob)
            self.statusLabel.text = (f"Loaded {volume_id}: {len(nodes['segmentations'])} segmentations"
                                     + (" + probability" if nodes["prob"] else ""))
            slicer.util.mainWindow().raise_()
        except Exception as exc:  # noqa: BLE001
            logging.exception("loadSeries %s", volume_id)
            self.statusLabel.text = f"Load failed: {exc}"


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


def build_dashboard_html(stats, load_prob):
    data = {
        "generated_at": stats.get("generated_at"),
        "min_node_ml": stats.get("min_node_ml"),
        "models": stats["models"],
        "colors": {m: "rgb(%d,%d,%d)" % MODEL_COLORS.get(m, (150, 150, 150)) for m in stats["models"]},
        "short": {m: MODEL_SHORT.get(m, m) for m in stats["models"]},
        "studies": stats["studies"],
        "volumes": [_slim_volume(v) for v in stats["volumes"]],
        "load_prob": bool(load_prob),
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
</style></head>
<body>
<header>
  <h1>PDAC Review</h1>
  <span id="crumb"></span>
  <span class="meta" id="meta"></span>
</header>
<div class="filters" id="filters"></div>
<main id="main"></main>
<script>
const D = %%DATA%%;
const M = D.models;
const fmt = (x, d=2) => (x === null || x === undefined || x === "") ? "" : Number(x).toFixed(d);
const pct = x => (100 * x).toFixed(0) + "%";
let state = { view: "overview", study: null, models: new Set(M), phases: new Set(), spectral: new Set(),
              thick: new Set(), maxMissing: 1.0, loadProb: D.load_prob };
const charts = [];
function disposeCharts() { while (charts.length) charts.pop().dispose(); }
function mkChart(el, option) { const c = echarts.init(el, null, {renderer: "canvas"}); c.setOption(option); charts.push(c); return c; }
window.addEventListener("resize", () => charts.forEach(c => c.resize()));

const allPhases = [...new Set(D.volumes.map(v => v.phase || "?"))].sort((a,b) => (phaseSec(a)-phaseSec(b)));
const allSpectral = [...new Set(D.volumes.map(v => v.spectral))].sort();
const allThick = [...new Set(D.volumes.map(v => v.slice_thickness_mm))].sort((a,b)=>a-b);
state.phases = new Set(allPhases); state.spectral = new Set(allSpectral); state.thick = new Set(allThick);
function phaseSec(p) { const v = D.volumes.find(v => (v.phase || "?") === p); return v ? v.phase_seconds : 1e9; }

function seriesLabel(v) {
  return `#${v.series_number} ${v.phase || ""} ${v.spectral === "monoenergetic" ? (v.series_description.match(/\d+\s*keV/) || ["keV"])[0] : v.spectral} ${v.slice_thickness_mm}mm`;
}
function volumesOf(studyId) {
  return D.volumes.filter(v => (v.study_id || v.case_id) === studyId && state.phases.has(v.phase || "?") &&
    state.spectral.has(v.spectral) && state.thick.has(v.slice_thickness_mm) && v.missing_frac <= state.maxMissing)
    .sort((a,b) => a.series_number - b.series_number);
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
      [0, 0.1, 0.25, 0.5, 1.0].map(x => `<option value="${x}" ${x === state.maxMissing ? "selected" : ""}>${x === 1 ? "any" : "≤ " + pct(x)}</option>`).join("") + `</select>` +
    ` <label><input type="checkbox" id="loadProb" ${state.loadProb ? "checked" : ""}>load probability map</label>`;
  f.querySelectorAll("input[type=checkbox][data-group]").forEach(el => el.addEventListener("change", e => {
    const g = e.target.dataset.group; let val = e.target.dataset.val; if (g === "thick") val = Number(val);
    if (e.target.checked) state[g].add(val); else state[g].delete(val); render();
  }));
  f.querySelector("#maxMissing").addEventListener("change", e => { state.maxMissing = Number(e.target.value); render(); });
  f.querySelector("#loadProb").addEventListener("change", e => { state.loadProb = e.target.checked; });
}

function studyRows() {
  const studies = [...new Set(D.volumes.map(v => v.study_id || v.case_id))];
  const rows = studies.map(c => {
    const vols = volumesOf(c); if (!vols.length) return null;
    const v0 = D.volumes.find(v => (v.study_id || v.case_id) === c);
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

function openStudy(caseId) { state.view = "study"; state.study = caseId; render(); }

function renderStudy() {
  const vols = volumesOf(state.study);
  const v0 = D.volumes.find(v => (v.study_id || v.case_id) === state.study);
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
  mkChart(document.getElementById("c1"), mk("total_ml", "Segmented volume per series", "mL"));
  mkChart(document.getElementById("c2"), mk("n_nodes", "Node count per series", "nodes"));
  // agreement across series (volume-only)
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
    h += `<tr><td><button data-vid="${v.volume_id}">Load</button></td><td>${v.series_number}</td><td class="left" title="${v.flags}">${v.series_description}${v.flags ? ' <span class="warn" title="' + v.flags + '">⚠</span>' : ""}</td><td>${v.phase}</td><td>${v.spectral}</td><td>${v.slice_thickness_mm}</td><td>${v.n_slices}</td><td class="${v.missing_frac > 0.25 ? "warn" : ""}">${v.missing_frac ? pct(v.missing_frac) : ""}</td>`;
    for (const m of activeModels()) { const e = v.models[m];
      h += e ? `<td>${fmt(e.total_ml)}</td><td>${e.n_nodes}${e.n_specks ? `<span class="muted"> +${e.n_specks} specks</span>` : ""}</td><td>${fmt(e.largest_ml)}</td>` : `<td></td><td></td><td></td>`; }
    h += `</tr>`;
    const nodeLists = activeModels().filter(m => v.models[m] && v.models[m].nodes.length).map(m =>
      `<span class="swatch" style="background:${D.colors[m]}"></span>${D.short[m]}: ` + v.models[m].nodes.map(n => `${fmt(n.ml)} mL (${n.short_mm}×${n.long_mm} mm)`).join(", "));
    if (nodeLists.length) h += `<tr><td></td><td colspan="${7 + 3 * activeModels().length}" class="left nodes">${nodeLists.join("<br>")}</td></tr>`;
  }
  h += `</table>`;
  document.getElementById("tbl").innerHTML = h;
  main.querySelectorAll("button[data-vid]").forEach(b => b.addEventListener("click", () => loadInSlicer(b.dataset.vid)));
}

function loadInSlicer(vid) {
  if (!window.slicerPython) { alert("Not running inside Slicer (window.slicerPython missing)."); return; }
  window.slicerPython.evalPython(`slicer.modules.PDACReviewWidget.loadSeries('${vid}', ${state.loadProb ? "True" : "False"})`);
}

function render() {
  disposeCharts();
  document.getElementById("meta").textContent = `${D.studies.length} studies · ${D.volumes.length} volumes · nodes ≥ ${D.min_node_ml} mL · ${D.generated_at}`;
  renderFilters();
  if (state.view === "study") renderStudy(); else renderOverview();
}
render();
</script>
</body></html>
"""
