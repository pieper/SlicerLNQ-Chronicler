"""build_cohort.py / bench_report.py / qc_extras.py on a staged synthetic study."""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bench_report  # noqa: E402
import build_cohort  # noqa: E402
import qc_extras  # noqa: E402
import stage_dicom  # noqa: E402
from test_stage_dicom import write_series  # noqa: E402

MODELS = "mediastinal-v1 axillary-v1"


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    root = tmp_path_factory.mktemp("dicom")
    z = lambda n: [float(i) for i in range(n)]  # noqa: E731
    for case, nser in (("E00000001", 2), ("E00000002", 1)):
        cdir = os.path.join(root, case)
        os.makedirs(cdir)
        for s in range(1, nser + 1):
            write_series(cdir, s, f"Body {s}.0", z(20 + 5 * s))
        write_series(cdir, 9, "Topogram", [0.0, 0.0], image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"))
        stage_dicom.stage_case(case, cdir, str(tmp_path_factory.getbasetemp() / "work"), min_slices=20)
    return str(tmp_path_factory.getbasetemp() / "work")


def _read(path, delim=","):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


def test_build_cohort_links_and_tasks(work):
    assert build_cohort.main(["--work", work, "--models", MODELS, "--chunk-size", "2"]) == 0
    vols = _read(os.path.join(work, "manifest", "volumes.csv"))
    assert len(vols) == 3
    idx = _read(os.path.join(work, "manifest", "series_index.csv"))
    assert len(idx) == 5 and sum(1 for r in idx if r["decision"] == "rejected") == 2
    for v in vols:
        link = os.path.join(work, "cohort", "nrrd", f"{v['volume_id']}_0000.nrrd")
        assert os.path.islink(link) and os.path.isfile(link)
        assert not os.path.isabs(os.readlink(link))
    tasks = _read(os.path.join(work, "manifest", "predict_tasks.tsv"), "\t")
    # 3 volumes / chunk 2 → 2 chunks per model, 2 models
    assert len(tasks) == 4
    assert [t["model"] for t in tasks] == ["mediastinal-v1"] * 2 + ["axillary-v1"] * 2
    sizes = sorted(int(t["n_volumes"]) for t in tasks if t["model"] == "mediastinal-v1")
    assert sizes == [1, 2]
    chunk = open(tasks[0]["chunk_file"]).read().strip().splitlines()
    vid, ct = chunk[0].split("\t")
    assert vid.startswith("E0000000") and ct.endswith("ct.nrrd")


def test_pending_skips_done_volumes(work):
    vols = _read(os.path.join(work, "manifest", "volumes.csv"))
    seg, prob = build_cohort.output_paths(vols[0], "mediastinal-v1")
    for p in (seg, prob):
        open(p, "wb").write(b"NRRD0004\n")
    build_cohort.main(["--work", work, "--models", MODELS, "--chunk-size", "8"])
    tasks = _read(os.path.join(work, "manifest", "predict_tasks.tsv"), "\t")
    med = [t for t in tasks if t["model"] == "mediastinal-v1"]
    assert sum(int(t["n_volumes"]) for t in med) == 2
    seg_link, prob_link = build_cohort.cohort_link_paths(work, vols[0]["volume_id"], "mediastinal-v1")
    assert os.path.islink(seg_link) and os.path.islink(prob_link)
    # --volume-ids restricts planning
    build_cohort.main(["--work", work, "--models", MODELS, "--volume-ids", vols[1]["volume_id"]])
    tasks = _read(os.path.join(work, "manifest", "predict_tasks.tsv"), "\t")
    assert all(int(t["n_volumes"]) == 1 for t in tasks) and len(tasks) == 2


def test_bench_select(work, capsys):
    build_cohort.main(["--work", work, "--models", MODELS])
    assert bench_report.main(["--work", work, "--select", "--n", "3"]) == 0
    ids = capsys.readouterr().out.strip().split(",")
    assert len(ids) == 3 and len(set(ids)) == 3


def test_bench_report_recommends(work, capsys):
    bdir = os.path.join(work, "manifest", "bench")
    os.makedirs(bdir, exist_ok=True)
    def summary(part, gpu_mb, secs, fallback=False):
        return {"model": "mediastinal-v1", "partition": part, "gpu": "GPU", "load_seconds": 30,
                "n_failed": 0, "rss_self_mb": 20000, "rss_children_mb": 4000,
                "volumes": [{"status": "ok", "total_seconds": secs, "gpu_peak_alloc_mb": gpu_mb,
                             "cpu_fallback": fallback, "rss_self_mb": 20000, "rss_children_mb": 4000}]}
    json.dump(summary("rtx6000", 23000, 300, True), open(os.path.join(bdir, "rtx6000-mediastinal-v1.json"), "w"))
    json.dump(summary("rtx8000", 23000, 250), open(os.path.join(bdir, "rtx8000-mediastinal-v1.json"), "w"))
    json.dump(summary("dgx-a100", 23000, 120), open(os.path.join(bdir, "dgx-a100-mediastinal-v1.json"), "w"))
    assert bench_report.main(["--work", work]) == 0
    out = capsys.readouterr().out
    assert "recommend: GPU_PARTITION=rtx8000" in out
    assert "PREDICT_MEM=32G" in out   # 24000 MB * 1.3 = 30.5 GB → 32


def test_qc_extras(work, capsys):
    vols = _read(os.path.join(work, "manifest", "volumes.csv"))
    qdir = os.path.join(work, "cohort", "qc", "mediastinal-v1")
    os.makedirs(qdir, exist_ok=True)
    with open(os.path.join(qdir, "qc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "pred_volume_mL_p0.5", "has_gt"])
        w.writeheader()
        for v in vols:
            w.writerow({"case_id": v["volume_id"], "pred_volume_mL_p0.5": 1.5, "has_gt": False})
        w.writerow({"case_id": "unknown", "pred_volume_mL_p0.5": 0, "has_gt": False})
    assert qc_extras.main(["--work", work, "--models", MODELS]) == 0
    rows = _read(os.path.join(qdir, "qc.csv"))
    assert rows[0]["series_description"].startswith("Body") and rows[0]["case"] == "E00000001"
    assert rows[0]["pred_volume_mL_p0.5"] == "1.5"
    assert rows[-1]["series_description"] == ""
