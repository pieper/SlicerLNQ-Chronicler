"""model_registry.py — which models the pipeline knows, where their files go.

Two kinds of model:

  * lnq-segmenter registry models (mediastinal-v1, …): resolved by the
    lnq_segmenter package at run time; binary lymph-node SEG + one probability
    map, files `<model>-seg.nrrd` / `<model>-prob.nrrd` (legacy names kept).
  * external nnU-Net v2 model folders listed in models.json (pants-v1, …):
    multi-label, files `<model>.seg.nrrd` (a Slicer segmentation NRRD with
    segment names/colors in the header) plus one probability map per class of
    interest, e.g. `<model>-prob.nrrd`, `<model>-pancreas-prob.nrrd`.

Everything that names an output file, a cohort link or a dashboard row goes
through here so build_cohort / predict_batch / pdac_stats / PDACReview agree.
Standard library + (for write_slicer_seg_nrrd) SimpleITK only.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(_HERE, "models.json")

# Display defaults for the lnq-segmenter models (colors mirror the registry).
LNQ_DISPLAY = {
    "mediastinal-v1":    {"short": "mediastinal", "color": [200, 100, 230]},
    "abdominopelvic-v1": {"short": "abd/pelvic",  "color": [120, 220, 120]},
    "axillary-v1":       {"short": "axillary",    "color": [255, 150, 100]},
    "inguinal-v1":       {"short": "inguinal",    "color": [240, 220, 60]},
}
LNQ_SEG_SUFFIX, LNQ_PROB_SUFFIX = "-seg.nrrd", "-prob.nrrd"


def load_registry(path=REGISTRY_PATH):
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


_REGISTRY = None


def registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def spec(model):
    """models.json entry for an external model, or None for lnq models."""
    return registry().get(model)


def is_external(model):
    return model in registry()


def model_folder(model, models_cache):
    s = spec(model)
    return os.path.join(models_cache, s["folder"]) if s else None


# --------------------------------------------------------------------------- files

def output_files(vol_dir, model):
    """Ordered dict role -> absolute path of every file a model writes for a
    volume. Roles: 'seg', 'prob' (the primary probability map), then one role
    per extra class ('<class>-prob')."""
    s = spec(model)
    out = {}
    if s is None:
        out["seg"] = os.path.join(vol_dir, model + LNQ_SEG_SUFFIX)
        out["prob"] = os.path.join(vol_dir, model + LNQ_PROB_SUFFIX)
        return out
    out["seg"] = os.path.join(vol_dir, model + s["outputs"]["seg"])
    for cls, suffix in s["outputs"]["prob"].items():
        role = "prob" if suffix == "-prob.nrrd" else suffix[1:-len(".nrrd")]
        out[role] = os.path.join(vol_dir, model + suffix)
    return out


def output_paths(volume_row, model):
    """(seg, prob) — kept for callers that only need the classic pair."""
    files = output_files(os.path.dirname(volume_row["ct_path"]), model)
    return files["seg"], files.get("prob")


def cohort_link_files(work, volume_id, model):
    """role -> link path under cohort/predictions/<model>/. lnq models keep the
    LNQReview convention (<vid>.nrrd, <vid>-prob.nrrd); external models mirror
    their suffixes (<vid>.seg.nrrd, <vid>-prob.nrrd, <vid>-pancreas-prob.nrrd)."""
    pred_dir = os.path.join(work, "cohort", "predictions", model)
    s = spec(model)
    links = {}
    if s is None:
        links["seg"] = os.path.join(pred_dir, f"{volume_id}.nrrd")
        links["prob"] = os.path.join(pred_dir, f"{volume_id}-prob.nrrd")
        return links
    links["seg"] = os.path.join(pred_dir, volume_id + s["outputs"]["seg"])
    for cls, suffix in s["outputs"]["prob"].items():
        role = "prob" if suffix == "-prob.nrrd" else suffix[1:-len(".nrrd")]
        links[role] = os.path.join(pred_dir, volume_id + suffix)
    return links


# --------------------------------------------------------------------------- stats keys

def stats_keys(model):
    """[(key, cfg)] — one dashboard/stats row per key. cfg has: model, labels
    (list of label *names*, or None for 'any foreground'), seg_suffix,
    prob_suffix, short, color, component, default_prob."""
    s = spec(model)
    if s is None:
        d = LNQ_DISPLAY.get(model, {"short": model, "color": [150, 150, 150]})
        return [(model, {"model": model, "labels": None, "seg_suffix": LNQ_SEG_SUFFIX,
                         "prob_suffix": LNQ_PROB_SUFFIX, "short": d["short"], "color": d["color"],
                         "component": "node", "default_prob": True})]
    out = []
    for cls, cfg in s["stats"].items():
        out.append((f"{model}/{cls}", {"model": model, "labels": list(cfg["labels"]),
                                       "seg_suffix": s["outputs"]["seg"], "prob_suffix": cfg["prob"],
                                       "short": cfg["short"], "color": cfg["color"],
                                       "component": cfg.get("component", "component"),
                                       "default_prob": bool(cfg.get("default_prob", False))}))
    return out


def all_stats_keys(models):
    out = []
    for m in models:
        out.extend(stats_keys(m))
    return out


# --------------------------------------------------------------------------- labels

def read_dataset_labels(model_dir):
    """dataset.json 'labels' -> {name: [values]} (regions become lists)."""
    with open(os.path.join(model_dir, "dataset.json")) as f:
        labels = json.load(f)["labels"]
    out = {}
    for name, v in labels.items():
        out[name] = [int(x) for x in v] if isinstance(v, (list, tuple)) else [int(v)]
    return out


def resolve_label_values(names, dataset_labels):
    """Union of label values for a list of label/region names."""
    values = set()
    for n in names:
        if n not in dataset_labels:
            raise KeyError(f"label {n!r} not in dataset.json")
        values.update(dataset_labels[n])
    return sorted(values)


def label_value_names(dataset_labels):
    """value -> name for plain (non-region) labels, background excluded."""
    out = {}
    for name, values in dataset_labels.items():
        if name == "background" or len(values) != 1:
            continue
        out[values[0]] = name
    return out


# --------------------------------------------------------------------------- Slicer .seg.nrrd

def write_slicer_seg_nrrd(labelmap, out_path, value_names, colors=None, extents=None):
    """Write a SimpleITK uint8 labelmap as a Slicer segmentation NRRD
    (`.seg.nrrd`): one shared-labelmap layer plus the Segment<N>_* header
    fields Slicer's segmentation reader uses for names, colors and label
    values. Drag-and-drop into Slicer then shows named, colored segments.

    value_names: {label value: segment name} (only labels present are written
    if `extents` is given for them; otherwise all).
    colors:      {segment name: (r, g, b) 0-255}; missing -> gray.
    extents:     {label value: (xmin, xmax, ymin, ymax, zmin, zmax)} in the
                 labelmap's IJK; if None they are computed here.
    """
    import SimpleITK as sitk
    img = sitk.Cast(labelmap, sitk.sitkUInt8)
    if extents is None:
        shape = sitk.LabelShapeStatisticsImageFilter()
        shape.Execute(img)
        extents = {}
        for label in shape.GetLabels():
            x, y, z, sx, sy, sz = shape.GetBoundingBox(label)
            extents[int(label)] = (x, x + sx - 1, y, y + sy - 1, z, z + sz - 1)
    colors = colors or {}
    n = 0
    for value in sorted(value_names):
        if value not in extents:
            continue                       # label absent from this volume
        name = value_names[value]
        r, g, b = colors.get(name, (150, 150, 150))
        e = extents[value]
        k = f"Segment{n}_"
        img.SetMetaData(k + "ID", f"Segment_{value}")
        img.SetMetaData(k + "Name", name)
        img.SetMetaData(k + "NameAutoGenerated", "0")
        img.SetMetaData(k + "Color", f"{r / 255:.4f} {g / 255:.4f} {b / 255:.4f}")
        img.SetMetaData(k + "ColorAutoGenerated", "0")
        img.SetMetaData(k + "LabelValue", str(value))
        img.SetMetaData(k + "Layer", "0")
        img.SetMetaData(k + "Extent", " ".join(str(int(v)) for v in e))
        img.SetMetaData(k + "Tags", "")
        n += 1
    img.SetMetaData("Segmentation_ContainedRepresentationNames", "Binary labelmap|")
    img.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")
    img.SetMetaData("Segmentation_ReferenceImageExtentOffset", "0 0 0")
    img.SetMetaData("Segmentation_ConversionParameters", "")
    tmp = out_path + ".partial.seg.nrrd"
    sitk.WriteImage(img, tmp, useCompression=True)
    os.replace(tmp, out_path)
    return n
