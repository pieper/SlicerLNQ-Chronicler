"""Local tests for stage_dicom.py using a synthetic DICOM study.

Run:  python -m pytest mlsc/tests -q   (needs pydicom, SimpleITK, numpy)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import stage_dicom  # noqa: E402

AXIAL = [1, 0, 0, 0, 1, 0]
CORONAL = [1, 0, 0, 0, 0, -1]
CT_IMAGE = stage_dicom.CT_IMAGE
ENHANCED_CT = stage_dicom.ENHANCED_CT
SC_IMAGE = "1.2.840.10008.5.1.4.1.1.7"

ROWS = COLS = 16
STUDY_UID = generate_uid()
FOR_UID = generate_uid()


def _base_ds(path, sop_class, series_uid, series_number, desc, instance_number,
             image_type=("ORIGINAL", "PRIMARY", "AXIAL"), modality="CT"):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = FOR_UID
    ds.Modality = modality
    ds.SeriesNumber = series_number
    ds.SeriesDescription = desc
    ds.InstanceNumber = instance_number
    ds.ImageType = list(image_type)
    ds.PatientName = "SHOULD^NOT^LEAK"
    ds.PatientID = "MRN-DO-NOT-LEAK"
    ds.StudyDate = "20260101"
    ds.StudyTime = "120000"
    ds.Rows = ROWS
    ds.Columns = COLS
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.ConvolutionKernel = "Br40"
    ds.KVP = 120
    return ds


def _plane(value):
    return np.full((ROWS, COLS), value, dtype=np.uint16)


def write_series(root, series_number, desc, positions, iop=AXIAL, sop_class=CT_IMAGE,
                 image_type=("ORIGINAL", "PRIMARY", "AXIAL"), modality="CT",
                 acquisition_numbers=None, spacing=(0.5, 0.5), thickness=1.0,
                 series_uid=None, first_index=0):
    """Write one legacy single-frame series; raw value = 1024 + 10*index."""
    series_uid = series_uid or generate_uid()
    sdir = os.path.join(root, f"series{series_number}")
    os.makedirs(sdir, exist_ok=True)
    for i, z in enumerate(positions, first_index):
        path = os.path.join(sdir, f"img{i:04d}.dcm")
        ds = _base_ds(path, sop_class, series_uid, series_number, desc, i + 1,
                      image_type=image_type, modality=modality)
        ds.ImageOrientationPatient = list(iop)
        ds.ImagePositionPatient = [-10.0, -20.0, float(z)] if iop == AXIAL else [-10.0, float(z), 30.0]
        ds.PixelSpacing = list(spacing)
        ds.SliceThickness = thickness
        ds.RescaleSlope = 1
        ds.RescaleIntercept = -1024
        if acquisition_numbers is not None:
            ds.AcquisitionNumber = acquisition_numbers[i]
        ds.PixelData = _plane(1024 + 10 * i).tobytes()
        ds.save_as(path, write_like_original=False)
    return series_uid


def write_enhanced_series(root, series_number, desc, positions):
    series_uid = generate_uid()
    sdir = os.path.join(root, f"series{series_number}")
    os.makedirs(sdir, exist_ok=True)
    path = os.path.join(sdir, "multiframe.dcm")
    ds = _base_ds(path, ENHANCED_CT, series_uid, series_number, desc, 1)
    n = len(positions)
    ds.NumberOfFrames = n
    shared = Dataset()
    po = Dataset()
    po.ImageOrientationPatient = list(AXIAL)
    shared.PlaneOrientationSequence = Sequence([po])
    pm = Dataset()
    pm.PixelSpacing = [0.5, 0.5]
    pm.SliceThickness = 1.0
    shared.PixelMeasuresSequence = Sequence([pm])
    pv = Dataset()
    pv.RescaleSlope = 1
    pv.RescaleIntercept = -1024
    shared.PixelValueTransformationSequence = Sequence([pv])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    frames = []
    # Deliberately store frames in reverse z order to exercise sorting.
    order = list(range(n))[::-1]
    for k in order:
        fg = Dataset()
        pp = Dataset()
        pp.ImagePositionPatient = [-10.0, -20.0, float(positions[k])]
        fg.PlanePositionSequence = Sequence([pp])
        frames.append(fg)
    ds.PerFrameFunctionalGroupsSequence = Sequence(frames)
    pixels = np.stack([_plane(1024 + 10 * k) for k in order], axis=0)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, write_like_original=False)
    return series_uid


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("dicom")
    case = os.path.join(root, "site", "E12345678")
    os.makedirs(case)
    z = lambda n, dz=1.0: [i * dz for i in range(n)]  # noqa: E731
    write_series(case, 1, "Topogram", [0.0, 0.0], image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"))
    write_series(case, 2, "Body 1.0 Br40", z(30))
    write_series(case, 3, "Coronal MPR", z(30), iop=CORONAL, image_type=("DERIVED", "SECONDARY", "MPR"))
    write_series(case, 4, "Dup slice", z(29) + [10.0])
    write_series(case, 5, "Missing slices", [float(i) for i in range(30) if i not in (10, 11, 12)])
    write_series(case, 6, "Irregular", [0, 1, 2, 3.4, 4, 5, 6, 7, 8, 9] + [float(i) for i in range(10, 30)])
    write_enhanced_series(case, 7, "Enhanced 0.4", z(25, 0.4))
    write_series(case, 8, "Dose Report", [0.0], sop_class=SC_IMAGE, image_type=("DERIVED", "SECONDARY"))
    write_series(case, 9, "Tiny", z(10))
    write_series(case, 10, "Body VNC 1.0", z(30))
    write_series(case, 11, "Interleaved", z(30) + z(30),
                 acquisition_numbers=[1] * 30 + [2] * 30)
    write_series(case, 12, "PET", z(30), modality="PT")
    write_series(case, 13, "Iodine map", z(30), image_type=("DERIVED", "SECONDARY", "VNC"))
    # 70 of 100 positions present (30 % missing): kept + interpolated by default
    write_series(case, 14, "Half copied", [float(i) for i in range(40)] + [float(i) for i in range(41, 100, 2)])
    # a stray non-DICOM file
    with open(os.path.join(case, "notes.txt"), "w") as f:
        f.write("not dicom\n")
    work = tmp_path_factory.mktemp("work")
    rows = stage_dicom.stage_case("E12345678", case, str(work), min_slices=20)
    return {"case": case, "work": str(work), "rows": rows}


def _row(study, series_number, idx=0):
    rows = [r for r in study["rows"] if r["series_number"] == series_number]
    assert rows, f"no row for series {series_number}"
    return rows[idx]


def test_case_discovery(study):
    root = os.path.dirname(os.path.dirname(study["case"]))
    assert [os.path.basename(d) for d in stage_dicom.find_case_dirs(root)] == ["E12345678"]


def test_rejections(study):
    assert _row(study, 1)["decision"] == "rejected" and "localizer" in _row(study, 1)["reasons"]
    assert "orientation=coronal" in _row(study, 3)["reasons"]
    assert "sop_class=SecondaryCapture" in _row(study, 8)["reasons"]
    assert "too_few_slices=10" in _row(study, 9)["reasons"]
    assert "modality=PT" in _row(study, 12)["reasons"]
    r14 = _row(study, 14)
    assert r14["decision"] == "staged"
    assert "missing_slices=30" in r14["flags"] and "interpolated_slices=30" in r14["flags"]


def test_clean_axial_volume(study):
    import SimpleITK as sitk
    r = _row(study, 2)
    assert r["decision"] == "staged" and r["flags"] == ""
    assert r["series_dir"] == "002_Body_1-0_Br40"
    assert r["volume_id"] == "E12345678__002_Body_1-0_Br40"
    assert "kernel=Br40" in r["tags"]
    img = sitk.ReadImage(r["ct_path"])
    assert img.GetSize() == (COLS, ROWS, 30)
    assert img.GetSpacing() == (0.5, 0.5, 1.0)
    assert img.GetOrigin() == (-10.0, -20.0, 0.0)
    assert img.GetPixelID() == sitk.sitkInt16
    arr = sitk.GetArrayFromImage(img)
    assert arr[0, 0, 0] == 0 and arr[29, 0, 0] == 290  # HU = raw - 1024
    geom = json.load(open(os.path.join(os.path.dirname(r["ct_path"]), "geometry.json")))
    assert geom["size_xyz"] == [COLS, ROWS, 30]
    assert "SHOULD" not in json.dumps(geom) and "MRN" not in json.dumps(geom)


def test_duplicate_slice_flagged(study):
    r = _row(study, 4)
    assert r["decision"] == "staged"
    assert "duplicate_positions=1" in r["flags"]
    assert r["n_slices"] == 29


def test_missing_slices_interpolated(study):
    import SimpleITK as sitk
    r = _row(study, 5)
    assert "missing_slices=3" in r["flags"] and "interpolated_slices=3" in r["flags"]
    assert "irregular_spacing" not in r["flags"]
    img = sitk.ReadImage(r["ct_path"])
    assert img.GetSize() == (COLS, ROWS, 30)          # full grid, not packed
    assert img.GetSpacing() == (0.5, 0.5, 1.0)
    arr = sitk.GetArrayFromImage(img)
    assert arr[9, 0, 0] == 90 and arr[13, 0, 0] == 100   # neighbours of the gap
    assert all(90 < arr[k, 0, 0] < 100 for k in (10, 11, 12))
    geom = json.load(open(os.path.join(os.path.dirname(r["ct_path"]), "geometry.json")))
    assert geom["n_missing_interpolated"] == 3 and geom["size_xyz"][2] == 30


def test_irregular_spacing_flagged(study):
    import SimpleITK as sitk
    r = _row(study, 6)
    assert "irregular_spacing=2" in r["flags"]
    img = sitk.ReadImage(r["ct_path"])
    assert abs(img.GetSpacing()[2] - 1.0) < 1e-6  # median spacing used


def test_enhanced_multiframe(study):
    import SimpleITK as sitk
    r = _row(study, 7)
    assert r["decision"] == "staged", r
    assert r["flags"] == ""
    img = sitk.ReadImage(r["ct_path"])
    assert img.GetSize() == (COLS, ROWS, 25)
    assert abs(img.GetSpacing()[2] - 0.4) < 1e-6
    arr = sitk.GetArrayFromImage(img)
    assert arr[0, 0, 0] == 0 and arr[24, 0, 0] == 240  # sorted ascending in z


def test_spectral_tag(study):
    assert "spectral=vnc" in _row(study, 10)["tags"]
    assert _row(study, 10)["spectral"] == "vnc"
    assert _row(study, 13)["spectral"] == "iodine"   # description beats ImageType


def test_interleaved_split(study):
    rows = [r for r in study["rows"] if r["series_number"] == 11]
    assert [r["decision"] for r in rows] == ["staged", "staged"]
    assert {r["series_dir"] for r in rows} == {"011_Interleaved_stk1", "011_Interleaved_stk2"}
    assert all("split_by_acquisition_number=2" in r["flags"] for r in rows)


def test_series_csv_written(study):
    path = os.path.join(study["work"], "E12345678", "series.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == set(stage_dicom.SERIES_COLUMNS)
    staged = [r for r in rows if r["decision"] == "staged"]
    assert len(staged) == 10  # 2,4,5,6,7,10,11a,11b,13,14
    text = open(path).read()
    assert "SHOULD" not in text and "MRN" not in text


def test_idempotent_rerun(study):
    ct = _row(study, 2)["ct_path"]
    mtime = os.path.getmtime(ct)
    stage_dicom.stage_case("E12345678", study["case"], study["work"], min_slices=20)
    assert os.path.getmtime(ct) == mtime


def test_dry_run_prints(study, capsys):
    stage_dicom.stage_case("E12345678", study["case"], study["work"], min_slices=20, dry_run=True)
    out = capsys.readouterr().out
    assert "Body 1.0 Br40" in out and "rejected" in out and "staged" in out


def test_sanitize():
    assert stage_dicom.sanitize("Body 1.0  Br40 / axial") == "Body_1-0_Br40_axial"
    assert stage_dicom.sanitize("") == "series"
    assert stage_dicom.sanitize("x" * 100).__len__() <= 48


def test_reconvert_when_source_grows(tmp_path):
    """Simulates a copy finishing after the first staging pass. Note the
    first pass must have *irregular* gaps: dropping every other slice looks
    exactly like a 2 mm series and is (correctly) not flagged."""
    import SimpleITK as sitk
    case = tmp_path / "E00000099"
    case.mkdir()
    uid = generate_uid()
    first = [float(i) for i in range(20)] + [float(i) for i in range(20, 60, 3)]   # 34 of 60
    write_series(str(case), 1, "Body", first, series_uid=uid)
    rows = stage_dicom.stage_case("E00000099", str(case), str(tmp_path / "w"), min_slices=20,
                                  max_missing_frac=0.25)
    assert rows[0]["decision"] == "rejected" and rows[0]["reasons"] == "incomplete_series=34/60"
    rows = stage_dicom.stage_case("E00000099", str(case), str(tmp_path / "w"), min_slices=20)
    assert rows[0]["decision"] == "staged" and "interpolated_slices=26" in rows[0]["flags"]
    ct = rows[0]["ct_path"]
    assert sitk.ReadImage(ct).GetSize()[2] == 60
    rest = [float(i) for i in range(20, 60) if (i - 20) % 3 != 0]                   # the other 26
    write_series(str(case), 1, "Body", rest, series_uid=uid, first_index=100)
    rows = stage_dicom.stage_case("E00000099", str(case), str(tmp_path / "w"), min_slices=20)
    assert rows[0]["decision"] == "staged" and rows[0]["flags"] == ""
    assert sitk.ReadImage(ct).GetSize()[2] == 60
    geom_path = os.path.join(os.path.dirname(ct), "geometry.json")
    assert json.load(open(geom_path))["n_source_frames"] == 60
    # third pass with unchanged source is a no-op
    mtime = os.path.getmtime(ct)
    stage_dicom.stage_case("E00000099", str(case), str(tmp_path / "w"), min_slices=20)
    assert os.path.getmtime(ct) == mtime
