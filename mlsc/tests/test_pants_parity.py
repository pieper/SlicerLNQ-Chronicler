"""Real-weights checks for the external (PanTS) model path.

Needs the PanTS model folder (dataset.json, plans.json, fold_all/checkpoint_best.pth)
and a venv with torch + nnunetv2 + nibabel:

    PANTS_MODEL_DIR=/path/to/nnUNet-PanTS-regions python -m pytest mlsc/tests/test_pants_parity.py -q -s

Skipped otherwise. CPU is fine (one patch, a few minutes on a laptop).

1. geometry round trip: the working-resolution array nnU-Net's preprocessor
   produces, mapped back with predict_batch's affine, must land on the CT grid
   (correlation with the CT ≥ 0.99) — proves orientation + half-voxel handling
   without running the network.
2. parity: our custom export (labels at 1 mm → nearest resample) vs nnU-Net's
   stock predict_from_files on the same phantom must agree on ≥ 97 % of
   voxels and on every label present; the phantom has a flipped, permuted
   direction matrix so a left/right or axis mistake would show up.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.environ.get("PANTS_MODEL_DIR")
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="PANTS_MODEL_DIR not set")


def make_phantom(path, shape=(48, 64, 64), spacing=(1.5, 1.4, 1.6)):
    """Soft-tissue ellipsoid with a brighter blob and a dark 'bowel' region,
    in HU, on a grid with a non-identity, partly flipped direction."""
    import SimpleITK as sitk
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = [s / 2 for s in shape]
    body = ((z - cz) / (0.45 * shape[0])) ** 2 + ((y - cy) / (0.42 * shape[1])) ** 2 + ((x - cx) / (0.45 * shape[2])) ** 2 <= 1
    arr = np.full(shape, -1000, dtype=np.int16)
    arr[body] = 40
    blob = ((z - cz) / 6) ** 2 + ((y - cy - 8) / 7) ** 2 + ((x - cx + 10) / 7) ** 2 <= 1
    arr[blob] = 120
    dark = ((z - cz + 8) / 5) ** 2 + ((y - cy + 10) / 6) ** 2 + ((x - cx - 12) / 6) ** 2 <= 1
    arr[dark] = -60
    rng = np.random.default_rng(0)
    arr = (arr + rng.normal(0, 8, shape)).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((spacing[2], spacing[1], spacing[0]))
    img.SetOrigin((-52.0, 31.0, 17.0))
    img.SetDirection((-1, 0, 0, 0, 0, 1, 0, 1, 0))       # flipped x, y/z swapped
    sitk.WriteImage(img, path, useCompression=True)
    return img


@pytest.fixture(scope="module")
def predictor():
    import predict_batch
    import model_registry
    spec = model_registry.spec("pants-v1")
    os.environ["MLSC_MODEL_DIR_PANTS_V1"] = MODEL_DIR
    p, entry = predict_batch.load_folder_predictor("pants-v1", spec, device="cpu")
    return p


def test_geometry_round_trip(tmp_path, predictor):
    import SimpleITK as sitk
    import nibabel as nib
    ct_path = str(tmp_path / "ct.nrrd")
    ct = make_phantom(ct_path)
    nii = str(tmp_path / "img_0000.nii.gz")
    sitk.WriteImage(ct, nii)
    pm, cm, dj = predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json
    data, _, props = cm.preprocessor_class(verbose=False).run_case([nii], None, pm, cm, dj)
    arr = data[0]                                            # normalized, (z,y,x) at 1 mm
    affine = np.asarray(props["nibabel_stuff"]["reoriented_affine"], dtype=float)
    bbox, shape_c = props["bbox_used_for_cropping"], props["shape_after_cropping_and_before_resampling"]
    M = np.eye(4)
    for a in range(3):
        ratio = float(shape_c[a]) / float(arr.shape[a])
        M[2 - a, 2 - a] = ratio
        M[2 - a, 3] = float(bbox[a][0]) - 0.5 + 0.5 * ratio
    out = str(tmp_path / "back.nii.gz")
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(np.float32), affine @ M), out)
    back = sitk.Resample(sitk.ReadImage(out), ct, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    a = sitk.GetArrayFromImage(back).ravel()
    b = sitk.GetArrayFromImage(ct).ravel().astype(np.float32)
    mask = np.abs(a) > 0                                     # inside the cropped region
    corr = np.corrcoef(a[mask], b[mask])[0, 1]
    print(f"round-trip correlation {corr:.4f} over {mask.sum()} voxels, working shape {arr.shape}")
    assert corr > 0.99


def test_parity_with_stock_export(tmp_path, predictor):
    import SimpleITK as sitk
    import model_registry
    import predict_batch
    spec = model_registry.spec("pants-v1")
    ct_path = str(tmp_path / "ct.nrrd")
    ct = make_phantom(ct_path)
    vol_dir = str(tmp_path / "vol")
    os.makedirs(vol_dir)
    out_files = model_registry.output_files(vol_dir, "pants-v1")
    m = predict_batch.predict_volume_multiclass(predictor, "pants-v1", spec, ct_path, out_files, str(tmp_path))
    print("ours:", {k: v for k, v in m.items() if k != "class_voxels"}, m["class_voxels"])
    ours = sitk.ReadImage(out_files["seg"])
    assert ours.GetSize() == ct.GetSize()
    assert np.allclose(ours.GetDirection(), ct.GetDirection()) and np.allclose(ours.GetOrigin(), ct.GetOrigin())
    assert ours.GetMetaData("Segmentation_MasterRepresentation") == "Binary labelmap"

    # stock nnU-Net export on the same volume
    nii_dir = tmp_path / "stock_in"
    nii_dir.mkdir()
    sitk.WriteImage(ct, str(nii_dir / "case_0000.nii.gz"))
    stock_dir = tmp_path / "stock_out"
    predictor.predict_from_files([[str(nii_dir / "case_0000.nii.gz")]], [str(stock_dir / "case")],
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1)
    stock = sitk.ReadImage(str(stock_dir / "case.nii.gz"))
    stock_on_ct = sitk.Resample(stock, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    a = sitk.GetArrayFromImage(ours)
    b = sitk.GetArrayFromImage(stock_on_ct)
    agree = float((a == b).mean())
    fg = (a > 0) | (b > 0)
    agree_fg = float((a[fg] == b[fg]).mean()) if fg.any() else 1.0
    print(f"label agreement: all voxels {agree:.4f}, foreground voxels {agree_fg:.4f}; "
          f"labels ours={sorted(np.unique(a).tolist())} stock={sorted(np.unique(b).tolist())}")
    assert agree > 0.97
    assert set(np.unique(a).tolist()) == set(np.unique(b).tolist())
    # probability maps sit on the CT grid and agree with the label map
    prob = sitk.ReadImage(out_files["pancreas-prob"])
    assert prob.GetSize() == ct.GetSize()
    p = sitk.GetArrayFromImage(prob)
    labels = model_registry.read_dataset_labels(MODEL_DIR)
    in_region = np.isin(a, labels["pancreas"])
    if in_region.any():
        assert (p[in_region] > 0.3).mean() > 0.9
