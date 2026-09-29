"""model_registry.py: file naming, stats keys, label resolution, Slicer .seg.nrrd writer."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import build_cohort  # noqa: E402
import model_registry  # noqa: E402

DATASET_JSON = {"channel_names": {"0": "CT"},
                "labels": {"background": 0, "liver": 1, "pancreas": [2, 3, 4], "pancreas_head": 3,
                           "pancreas_tail": 4, "pancreatic_lesion": 5},
                "regions_class_order": [1, 2, 3, 4, 5], "file_ending": ".nii.gz"}


def test_output_files_lnq_vs_external(tmp_path):
    vol = str(tmp_path)
    lnq = model_registry.output_files(vol, "mediastinal-v1")
    assert lnq == {"seg": os.path.join(vol, "mediastinal-v1-seg.nrrd"),
                   "prob": os.path.join(vol, "mediastinal-v1-prob.nrrd")}
    ext = model_registry.output_files(vol, "pants-v1")
    assert ext["seg"].endswith("pants-v1.seg.nrrd")
    assert ext["prob"].endswith("pants-v1-prob.nrrd")
    assert ext["pancreas-prob"].endswith("pants-v1-pancreas-prob.nrrd")
    links = model_registry.cohort_link_files("/w", "E1__s", "pants-v1")
    assert links["seg"].endswith("predictions/pants-v1/E1__s.seg.nrrd")
    assert links["pancreas-prob"].endswith("predictions/pants-v1/E1__s-pancreas-prob.nrrd")
    assert model_registry.cohort_link_files("/w", "E1__s", "inguinal-v1")["seg"].endswith("predictions/inguinal-v1/E1__s.nrrd")


def test_stats_keys():
    keys = model_registry.all_stats_keys(["axillary-v1", "pants-v1"])
    names = [k for k, _ in keys]
    assert names == ["axillary-v1", "pants-v1/lesion", "pants-v1/pancreas"]
    cfg = dict(keys)
    assert cfg["axillary-v1"]["labels"] is None and cfg["axillary-v1"]["default_prob"]
    assert cfg["pants-v1/lesion"]["labels"] == ["pancreatic_lesion"] and cfg["pants-v1/lesion"]["default_prob"]
    assert cfg["pants-v1/pancreas"]["prob_suffix"] == "-pancreas-prob.nrrd" and not cfg["pants-v1/pancreas"]["default_prob"]


def test_label_resolution(tmp_path):
    json.dump(DATASET_JSON, open(tmp_path / "dataset.json", "w"))
    labels = model_registry.read_dataset_labels(str(tmp_path))
    assert labels["pancreas"] == [2, 3, 4] and labels["pancreatic_lesion"] == [5]
    assert model_registry.resolve_label_values(["pancreas", "pancreatic_lesion"], labels) == [2, 3, 4, 5]
    assert model_registry.label_value_names(labels) == {1: "liver", 3: "pancreas_head", 4: "pancreas_tail", 5: "pancreatic_lesion"}


def test_write_slicer_seg_nrrd_roundtrip(tmp_path):
    arr = np.zeros((10, 12, 14), dtype=np.uint8)
    arr[2:5, 3:6, 4:8] = 1
    arr[6:8, 7:9, 9:12] = 5
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.8, 2.0))
    out = str(tmp_path / "m.seg.nrrd")
    n = model_registry.write_slicer_seg_nrrd(img, out, {1: "liver", 3: "pancreas_head", 5: "pancreatic_lesion"},
                                             {"liver": (150, 90, 60), "pancreatic_lesion": (255, 30, 30)})
    assert n == 2                                  # label 3 absent → not written
    back = sitk.ReadImage(out)
    assert back.GetSpacing() == (0.7, 0.8, 2.0)
    assert (sitk.GetArrayFromImage(back) == arr).all()
    keys = back.GetMetaDataKeys()
    assert back.GetMetaData("Segment0_Name") == "liver" and back.GetMetaData("Segment0_LabelValue") == "1"
    assert back.GetMetaData("Segment1_Name") == "pancreatic_lesion" and back.GetMetaData("Segment1_LabelValue") == "5"
    assert back.GetMetaData("Segment1_Extent") == "9 11 7 8 6 7"
    assert back.GetMetaData("Segment1_Color").startswith("1.0000 0.1176")
    assert back.GetMetaData("Segmentation_MasterRepresentation") == "Binary labelmap"
    assert "Segment2_Name" not in keys


def test_pending_requires_all_outputs(tmp_path):
    vol_dir = tmp_path / "E1" / "s"
    vol_dir.mkdir(parents=True)
    ct = str(vol_dir / "ct.nrrd")
    open(ct, "wb").write(b"NRRD0004\n")
    row = {"volume_id": "E1__s", "ct_path": ct}
    assert build_cohort.pending_volumes([row], "pants-v1") == [row]
    files = model_registry.output_files(str(vol_dir), "pants-v1")
    for role in ("seg", "prob"):
        open(files[role], "wb").write(b"x")
    assert build_cohort.pending_volumes([row], "pants-v1") == [row]     # pancreas-prob still missing
    open(files["pancreas-prob"], "wb").write(b"x")
    assert build_cohort.pending_volumes([row], "pants-v1") == []
    n = build_cohort.link_volume(str(tmp_path), row, ["pants-v1"])
    assert n == 4
    assert os.path.islink(tmp_path / "cohort" / "predictions" / "pants-v1" / "E1__s-pancreas-prob.nrrd")
