#!/usr/bin/env python3
"""Export a Project's approved Annotations as an nnU-Net v2 dataset directory.

For every case_id in the project's CohortResolution where the head Annotation
matches --status (default approved) and has a seg_ref, this writes:

  <output_root>/Dataset<NNN>_<task>/imagesTr/<case_id>_0000.nrrd  ->  CT (from cohort)
  <output_root>/Dataset<NNN>_<task>/labelsTr/<case_id>.nrrd       ->  seg (from annotation)
  <output_root>/Dataset<NNN>_<task>/dataset.json                  -> nnU-Net manifest

Mode:
  --link-mode symlink  (default; same approach as lnq-retro6.py)
  --link-mode copy     (forks the data — slower, more disk)

Files referenced via local-uri are resolved with --base-canonical / --base-local
so paths stored as /media/share/LNQ-data/... map to the Mac mount when running
locally, the same way the Slicer module does it.

Typical use:

  bin/export-nnunet-dataset.py \\
      --project-id project:<uuid> \\
      --output-root /private/tmp/media/share/LNQ-data/Inguinal-LN/nnUNet_raw \\
      --dataset-id 1 \\
      --task-name LNQinguinal \\
      --label-name inguinal_ln

Reads chronicle.conf for DOMAIN_NAME and COUCHDB_ADMIN_PASSWORD.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

try:
    import requests
except ImportError:
    print("Need 'requests'.", file=sys.stderr)
    sys.exit(1)


def load_conf(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class Chronicle:
    def __init__(self, base_url: str, db: str, password: str):
        self.base = base_url.rstrip("/")
        self.db = db
        self.s = requests.Session()
        self.s.auth = ("admin", password)

    def get(self, doc_id: str) -> dict:
        r = self.s.get(f"{self.base}/{self.db}/{doc_id}")
        r.raise_for_status()
        return r.json()

    def list_by_type(self, type_name: str) -> list:
        prefix = {
            "Cohort": "cohort:", "Protocol": "protocol:", "Project": "project:",
            "CohortResolution": "cohortresolution:", "Annotation": "annotation:",
            "ModelGeneration": "modelgeneration:", "InferenceRun": "inferencerun:",
        }[type_name]
        r = self.s.get(
            f"{self.base}/{self.db}/_all_docs",
            params={
                "startkey": json.dumps(prefix),
                "endkey": json.dumps(prefix + "￰"),
                "include_docs": "true",
            })
        r.raise_for_status()
        return [row["doc"] for row in r.json().get("rows", [])
                if row.get("doc") and not row["id"].startswith("_design")]


_RESOLVABLE_LOCATION_KINDS = ("local-uri", "equivalent-volume")


def resolve_local_path(ref, chronicle, canonical: str, local: str) -> str | None:
    """Translate a file reference into a filesystem path on this machine.
    Accepts both shapes:
      - new: {"blob_id": "blob:..."}      -> look up Blob, walk locations
      - legacy: {"kind": "local-uri", "value": "file://..."}  -> direct
    Walks every entry in locations whose kind is one of:
      - "local-uri":          file at value is byte-identical to the Blob
                              (file_sha256 == Blob _id suffix)
      - "equivalent-volume":  file at value differs in encoding but its
                              decompressed voxel array is sha256-equivalent
                              to the canonical version (recorded as
                              voxel_sha256 on the entry). Used when an
                              upstream pipeline re-encoded the file but the
                              scientific content is unchanged.
    Applies the canonical->local mount rewrite at every hop. Does NOT follow
    symlinks; nnU-Net is happy with symlinked inputs."""
    if not ref:
        return None
    # Blob-keyed: pull the Blob doc, walk its locations.
    blob_id = ref.get("blob_id")
    if blob_id:
        try:
            blob = chronicle.get(blob_id)
        except Exception:
            return None
        if blob is None:
            return None
        for loc in blob.get("locations") or []:
            if loc.get("kind") not in _RESOLVABLE_LOCATION_KINDS:
                continue
            v = loc.get("value") or ""
            if v.startswith("file://"):
                v = v[len("file://"):]
            if canonical and local and v.startswith(canonical):
                candidate = local + v[len(canonical):]
                if os.path.lexists(candidate):
                    return candidate
            if os.path.lexists(v):
                return v
        return None
    # Legacy shape.
    value = ref.get("value")
    if not value:
        return None
    if value.startswith("file://"):
        value = value[len("file://"):]
    if canonical and local and value.startswith(canonical):
        candidate = local + value[len(canonical):]
        if os.path.lexists(candidate):
            return candidate
    return value


def link_or_copy(src: str, dst: str, mode: str) -> None:
    # Idempotent: if src and dst already resolve to the same file (e.g. the
    # source data is already living at the canonical Dataset###/imagesTr
    # path because conversion wrote it there), leave it alone. Without this
    # check, a symlink-mode export would point the file at itself and
    # nnUNet's verify_dataset_integrity blows up with "Too many levels of
    # symbolic links" on first read.
    src_abs = os.path.abspath(src)
    if os.path.lexists(dst):
        try:
            if os.path.samefile(src_abs, dst):
                return
        except OSError:
            pass
        os.remove(dst)
    if mode == "copy":
        shutil.copyfile(src, dst)
    else:
        # symlink to the absolute source path so the dataset dir is portable
        # within the mount layout (same canonical -> local override applies).
        os.symlink(src_abs, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True, help="Project doc _id")
    ap.add_argument("--output-root", required=True,
                    help="Where the Dataset<NNN>_<task> dir gets created.")
    ap.add_argument("--dataset-id", type=int, default=0,
                    help="Numeric dataset id (default: 0). Embedded in the directory name.")
    ap.add_argument("--task-name", required=True,
                    help="Short task identifier, e.g. 'LNQinguinal'.")
    ap.add_argument("--label-name", default=None,
                    help="Foreground label name in dataset.json (default: derived from protocol's first non-background color_table entry or 'lesion').")
    ap.add_argument("--status", default="approved",
                    help="Annotation status filter (default: approved).")
    ap.add_argument("--link-mode", default="symlink", choices=("symlink", "copy"))
    ap.add_argument("--base-canonical", default="/media/share/LNQ-data",
                    help="Canonical Manila mount prefix in seg_ref values.")
    ap.add_argument("--base-local", default="/private/tmp/media/share/LNQ-data",
                    help="Local mount prefix to substitute for canonical paths.")
    ap.add_argument("--conf", default="chronicle.conf")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.conf):
        print(f"Missing {args.conf}", file=sys.stderr)
        sys.exit(1)
    conf = load_conf(args.conf)
    chronicle = Chronicle(f"https://{conf.get('DOMAIN_NAME', 'lnq-chronicle.isomics.dev')}",
                          "lnq", conf["COUCHDB_ADMIN_PASSWORD"])

    project = chronicle.get(args.project_id)
    if project.get("type") != "Project":
        print(f"--project-id {args.project_id} is not a Project", file=sys.stderr)
        sys.exit(1)

    # Cohort + resolution.
    resolutions = [r for r in chronicle.list_by_type("CohortResolution")
                   if r.get("cohort_id") == project["cohort_id"]]
    if not resolutions:
        print(f"No CohortResolution for cohort {project['cohort_id']}", file=sys.stderr)
        sys.exit(1)
    resolutions.sort(key=lambda r: (r.get("resolved_at") or r.get("created_at") or ""))
    cases_by_id = {c["case_id"]: c for c in resolutions[-1].get("cases", [])}

    # Walk the project's predecessor chain backward — Annotations may live
    # under v1 even though the head we were given is v2 (chronicle typed-doc
    # immutability). Match annotations whose project_id is in the chain.
    project_chain = set()
    cur = args.project_id
    while cur and cur not in project_chain:
        project_chain.add(cur)
        try:
            d = chronicle.get(cur)
        except Exception:
            break
        cur = d.get("predecessor")

    # Latest Annotation per case across the project chain.
    annotations = [a for a in chronicle.list_by_type("Annotation")
                   if a.get("project_id") in project_chain]
    annotations.sort(key=lambda a: (a.get("case_id") or "", a.get("version") or 0))
    head_by_case: dict[str, dict] = {}
    for ann in annotations:
        head_by_case[ann["case_id"]] = ann   # later versions overwrite earlier

    # Resolve label name from the Protocol if not specified.
    label_name = args.label_name
    if label_name is None:
        try:
            protocol = chronicle.get(project["protocol_id"])
            ct = protocol.get("color_table") or []
            for row in ct:
                if str(row.get("label")) != "0" and row.get("name"):
                    label_name = (row["name"]
                                  .replace(" ", "_").replace("(", "")
                                  .replace(")", "").lower())
                    break
        except Exception:
            pass
    if not label_name:
        label_name = "lesion"

    # Build output paths.
    dataset_dirname = f"Dataset{args.dataset_id:03d}_{args.task_name}"
    dataset_dir = os.path.join(args.output_root, dataset_dirname)
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    if not args.dry_run:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

    included = []
    skipped_no_head = []
    skipped_wrong_status = []
    skipped_no_seg = []
    skipped_no_ct = []
    skipped_seg_missing = []
    skipped_ct_missing = []

    for case_id, case in sorted(cases_by_id.items()):
        head = head_by_case.get(case_id)
        if head is None:
            skipped_no_head.append(case_id)
            continue
        if head.get("status") != args.status:
            skipped_wrong_status.append((case_id, head.get("status")))
            continue
        seg_ref = head.get("seg_ref")
        if not seg_ref:
            skipped_no_seg.append(case_id)
            continue
        ct_path = resolve_local_path(case.get("ct_ref"), chronicle,
                                     args.base_canonical, args.base_local)
        seg_path = resolve_local_path(seg_ref, chronicle,
                                       args.base_canonical, args.base_local)
        if not ct_path:
            skipped_no_ct.append(case_id)
            continue
        if not os.path.lexists(ct_path):
            skipped_ct_missing.append((case_id, ct_path))
            continue
        if not os.path.lexists(seg_path):
            skipped_seg_missing.append((case_id, seg_path))
            continue
        if not args.dry_run:
            link_or_copy(ct_path, os.path.join(images_dir, f"{case_id}_0000.nrrd"), args.link_mode)
            link_or_copy(seg_path, os.path.join(labels_dir, f"{case_id}.nrrd"), args.link_mode)
        included.append(case_id)

    manifest = {
        "channel_names": {"0": "CT"},
        "labels": {"background": "0", label_name: "1"},
        "numTraining": len(included),
        "file_ending": ".nrrd",
    }
    if not args.dry_run:
        with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Dataset: {dataset_dir}")
    print(f"  included:                {len(included)}")
    print(f"  skipped no head ann:     {len(skipped_no_head)}")
    print(f"  skipped wrong status:    {len(skipped_wrong_status)}")
    print(f"  skipped no seg_ref:      {len(skipped_no_seg)}")
    print(f"  skipped no ct_ref:       {len(skipped_no_ct)}")
    print(f"  skipped seg file gone:   {len(skipped_seg_missing)}")
    print(f"  skipped ct file gone:    {len(skipped_ct_missing)}")
    if skipped_seg_missing[:3] or skipped_ct_missing[:3]:
        print()
        for c, p in (skipped_seg_missing + skipped_ct_missing)[:6]:
            print(f"    missing: {c}  -> {p}")
    if args.dry_run:
        print()
        print("(dry-run; no files written)")


if __name__ == "__main__":
    main()
