"""predict_batch.py run loop with a fake predictor (no torch / nnU-Net needed)."""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import build_cohort  # noqa: E402
import predict_batch  # noqa: E402
import stage_dicom  # noqa: E402
from test_stage_dicom import write_series  # noqa: E402


@pytest.fixture()
def staged(tmp_path):
    cdir = tmp_path / "in" / "E00000007"
    cdir.mkdir(parents=True)
    write_series(str(cdir), 1, "Body 1.0", [float(i) for i in range(22)])
    write_series(str(cdir), 2, "Body 2.0", [float(i) for i in range(24)])
    work = tmp_path / "work"
    stage_dicom.stage_case("E00000007", str(cdir), str(work), min_slices=20)
    build_cohort.main(["--work", str(work), "--models", "mediastinal-v1", "--chunk-size", "8"])
    return str(work)


def _fake_predictor(monkeypatch, fail_on=None):
    import SimpleITK as sitk

    class FakePredictor:
        dataset_json = {"file_ending": ".nrrd"}

        def predict_from_files(self, inputs, outputs, save_probabilities, overwrite,
                               num_processes_preprocessing, num_processes_segmentation_export):
            ct = sitk.ReadImage(inputs[0][0])
            if fail_on and fail_on in inputs[0][0]:
                raise RuntimeError("boom")
            seg = sitk.Image(ct.GetSize(), sitk.sitkUInt8)
            seg.CopyInformation(ct)
            sitk.WriteImage(seg, outputs[0] + ".nrrd")
            arr = sitk.GetArrayFromImage(ct)
            probs = np.zeros((2,) + arr.shape, dtype=np.float32)
            probs[1, 0, 0, 0] = 0.9
            np.savez(outputs[0] + ".npz", probabilities=probs)
            print("fake nnunet done")

    def load(model, device="cuda", folds=None):
        return FakePredictor(), {"name": model, "version": "1.0.0", "folds": [0]}

    # lnq_segmenter.predict._write_probability_map is imported inside
    # predict_volume; provide a stand-in module with the same behaviour.
    def write_prob(nnunet_out, input_path, probability_output, progress_callback=None):
        data = np.load(nnunet_out + ".npz")["probabilities"][1]
        img = sitk.GetImageFromArray(data.astype("float32"))
        img.CopyInformation(sitk.ReadImage(input_path))
        sitk.WriteImage(img, probability_output, useCompression=True)
        os.remove(nnunet_out + ".npz")

    fake_pkg = types.ModuleType("lnq_segmenter")
    fake_pred = types.ModuleType("lnq_segmenter.predict")
    fake_pred._write_probability_map = write_prob
    fake_pkg.predict = fake_pred
    monkeypatch.setitem(sys.modules, "lnq_segmenter", fake_pkg)
    monkeypatch.setitem(sys.modules, "lnq_segmenter.predict", fake_pred)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(predict_batch, "load_predictor", load)


def test_task_run_writes_outputs_and_links(staged, monkeypatch, tmp_path):
    _fake_predictor(monkeypatch)
    rc = predict_batch.main(["--work", staged, "--task-index", "0", "--tmp-dir", str(tmp_path),
                             "--nnunet-base", str(tmp_path / "nn")])
    assert rc == 0
    vol_dir = os.path.join(staged, "E00000007", "001_Body_1-0")
    assert os.path.isfile(os.path.join(vol_dir, "mediastinal-v1-seg.nrrd"))
    assert os.path.isfile(os.path.join(vol_dir, "mediastinal-v1-prob.nrrd"))
    assert not any(p.endswith(".partial") for p in os.listdir(vol_dir))
    link = os.path.join(staged, "cohort", "predictions", "mediastinal-v1",
                        "E00000007__001_Body_1-0-prob.nrrd")
    assert os.path.islink(link) and os.path.isfile(link)
    import SimpleITK as sitk
    prob = sitk.GetArrayFromImage(sitk.ReadImage(link))
    assert prob.dtype == np.float32 and abs(prob[0, 0, 0] - 0.9) < 1e-6
    jsonl = os.path.join(staged, "logs", "predict-mediastinal-v1-0000.jsonl")
    recs = [json.loads(l) for l in open(jsonl)]
    assert [r["status"] for r in recs] == ["ok", "ok"]
    assert recs[0]["n_voxels"] > 0 and "total_seconds" in recs[0]
    # nothing left in the scratch dir
    assert not [d for d in os.listdir(tmp_path) if d.startswith("lnq-")]

    # second run: everything skipped
    rc = predict_batch.main(["--work", staged, "--task-index", "0", "--tmp-dir", str(tmp_path),
                             "--nnunet-base", str(tmp_path / "nn")])
    assert rc == 0
    recs = [json.loads(l) for l in open(jsonl)]
    assert [r["status"] for r in recs[-2:]] == ["skipped", "skipped"]


def test_failure_isolated_and_bench_summary(staged, monkeypatch, tmp_path):
    _fake_predictor(monkeypatch, fail_on="001_Body_1-0")
    out = tmp_path / "bench.json"
    rc = predict_batch.main(["--work", staged, "--model", "mediastinal-v1",
                             "--volumes", "E00000007__001_Body_1-0,E00000007__002_Body_2-0",
                             "--tmp-dir", str(tmp_path), "--bench-out", str(out),
                             "--nnunet-base", str(tmp_path / "nn")])
    assert rc == 1
    s = json.load(open(out))
    assert s["n_ok"] == 1 and s["n_failed"] == 1
    failed = [v for v in s["volumes"] if v["status"] == "failed"][0]
    assert "boom" in failed["error"]
    assert os.path.isfile(os.path.join(staged, "E00000007", "002_Body_2-0", "mediastinal-v1-seg.nrrd"))
    assert not os.path.exists(os.path.join(staged, "E00000007", "001_Body_1-0", "mediastinal-v1-seg.nrrd"))


def test_unknown_volume_id(staged, monkeypatch, tmp_path):
    _fake_predictor(monkeypatch)
    rc = predict_batch.main(["--work", staged, "--model", "mediastinal-v1", "--volumes", "nope",
                             "--tmp-dir", str(tmp_path), "--nnunet-base", str(tmp_path / "nn")])
    assert rc == 2
