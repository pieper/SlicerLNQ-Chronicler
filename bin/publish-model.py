#!/usr/bin/env python3
"""publish-model.py — take a ModelGeneration doc + its weights on disk and
produce a publishable model bundle for the lnq-segmenter registry.

Inputs:
  --model-generation-id  Chronicle ModelGeneration doc to mint from.
  --source-path          Local path that exists and points at the nnU-Net
                         plans dir (the parent containing fold_0..fold_N
                         and dataset.json + plans.json). The script will
                         look up `weights_ref.value` on the doc and confirm
                         this path is consistent with it.
  --name                 Short slug for the model, e.g. inguinal-v1.
                         Becomes the bundle dirname, zip name, registry key.
  --version              Semver of THIS model release, e.g. 1.0.0.

Outputs (under --output-dir, default /tmp/lnq-publish):
  <name>/                      Bundle staged from --source-path.
  <name>.zip                   Compressed bundle for GitHub Release upload.
  <name>.sha256                Hash of the zip; for the registry entry.
  <name>.registry.json         JSON snippet to add to lnq-segmenter
                               _registry.json under "models".
  <name>.card.json             Model card (provenance, training set, eval
                               numbers) embedded in the bundle and read by
                               the Slicer extension's "Model info" panel.

Stdlib only — runs anywhere with python3 + the source dir mounted.

Typical workflow for inguinal-v1:
  1. Unshelve any trainer instance so Manila is reachable, OR rsync the
     nnUNet_results plans dir to the local filesystem.
  2. bin/publish-model.py \\
       --model-generation-id modelgeneration:ce6cf698-... \\
       --source-path /media/share/LNQ-data/Inguinal_Syed/nnUNet_results/Dataset001_LNQinguinal/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres \\
       --name inguinal-v1 \\
       --version 1.0.0 \\
       --display-name "Inguinal lymph nodes (Syed v1)" \\
       --license "CC BY 4.0"
     NOTE: all LNQ weights are CC BY 4.0. Do not pass an NC variant
     without updating MODELS.md in the lnq-segmenter repo first.
  3. gh release create inguinal-v1 /tmp/lnq-publish/inguinal-v1.zip \\
       --notes-file /tmp/lnq-publish/inguinal-v1.card.json
  4. Append <name>.registry.json into the lnq-segmenter package's
     _registry.json under "models", bump the package version, ship.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import pickle
import re
import shutil
import socket
import sys
import urllib.error
import urllib.request
import zipfile


def iso_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def load_conf(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, _, v = ln.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class Chronicle:
    def __init__(self, base, db, password):
        self.base = base.rstrip("/"); self.db = db
        tok = base64.b64encode(f"admin:{password}".encode()).decode()
        self.auth = f"Basic {tok}"

    def get(self, doc_id):
        url = f"{self.base}/{self.db}/{doc_id}"
        req = urllib.request.Request(url, headers={"Authorization": self.auth})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def discover_folds(plans_dir):
    """Return [(fold_int, fold_dir_path)] for each fold_N subdir that has
    checkpoint_final.pth. Sorted by fold number."""
    out = []
    for name in os.listdir(plans_dir):
        m = re.fullmatch(r"fold_(\d+)", name)
        if not m:
            continue
        fold_dir = os.path.join(plans_dir, name)
        if not os.path.isfile(os.path.join(fold_dir, "checkpoint_final.pth")):
            continue
        out.append((int(m.group(1)), fold_dir))
    out.sort()
    return out


class _StubUnpickler(pickle.Unpickler):
    """Unpickles a torch checkpoint's metadata without torch: every non-builtin class is a stub,
    and tensor storages come back as None. nnU-Net's checkpoint is a dict whose 'trainer_name'
    and 'init_args' are plain str/dict values, so those survive intact."""

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):          # a reduce's result may still get BUILD's setstate
            return _StubUnpickler._Stub()

        def __setstate__(self, state):
            pass

    def find_class(self, module, name):
        if module in ("builtins", "collections", "copyreg", "_codecs", "datetime"):
            return super().find_class(module, name)
        return self._Stub()

    def persistent_load(self, pid):
        return None


def checkpoint_meta(path):
    """(trainer_name, configuration) from an nnU-Net checkpoint_final.pth, stdlib only.

    A torch checkpoint is a zip holding <archive>/data.pkl; its top-level dict carries
    'trainer_name' and 'init_args' (with 'configuration'), which is what
    nnUNetPredictor.initialize_from_trained_model_folder rebuilds the model from."""
    with zipfile.ZipFile(path) as zf:
        member = next((n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"), None)
        if member is None:
            raise SystemExit(f"{path}: not a torch zip checkpoint (no data.pkl)")
        data = _StubUnpickler(zf.open(member)).load()
    trainer = data.get("trainer_name")
    config = (data.get("init_args") or {}).get("configuration")
    if not (isinstance(trainer, str) and isinstance(config, str)):
        raise SystemExit(f"{path}: no trainer_name / init_args.configuration in the checkpoint")
    return trainer, config


def model_folder_parts(source_plans_dir, plans_json_path, folds):
    """(trainer, plans, configuration) of the model being published.

    Read from the model itself - the first fold's checkpoint and plans.json's plans_name - not
    parsed from the source folder's name: abdominopelvic-v1, axillary-v1 and mediastinal-v1 were
    published from folders named after the model, the parse found nothing, and their card and
    registry entry shipped with all three blank (lnq-segmenter fixed the registry in 0.3.1). A
    source folder that IS named <trainer>__<plans>__<config> must agree with its contents."""
    with open(plans_json_path) as f:
        plans = json.load(f).get("plans_name")
    if not plans:
        raise SystemExit(f"{plans_json_path}: no plans_name")
    trainer, config = checkpoint_meta(os.path.join(folds[0][1], "checkpoint_final.pth"))
    parts = (trainer, plans, config)
    named = tuple(os.path.basename(source_plans_dir.rstrip("/")).split("__"))
    if len(named) == 3 and named != parts:
        raise SystemExit(f"{source_plans_dir}: the folder is named {'__'.join(named)} but "
                         f"its checkpoint and plans say {'__'.join(parts)}")
    return parts


def stage_bundle(source_plans_dir, staged_dir):
    """Copy the canonical nnU-Net structure into staged_dir/. Layout
    matches what nnUNetPredictor.initialize_from_trained_model_folder()
    expects:

        <staged_dir>/
          dataset.json
          plans.json
          <trainer>__<plans>__<configuration>/     (always; see model_folder_parts)
            fold_0/checkpoint_final.pth
            fold_1/checkpoint_final.pth
            ...
    """
    parent = os.path.dirname(source_plans_dir.rstrip("/"))
    # dataset.json + plans.json live next to OR inside the plans dir; nnU-Net
    # writes them inside Dataset<NNN>_<task>/, i.e. the parent of plans_dir.
    for fn in ("dataset.json", "plans.json"):
        candidates = [
            os.path.join(parent, fn),                # canonical
            os.path.join(source_plans_dir, fn),      # sometimes seen
        ]
        src = next((c for c in candidates if os.path.isfile(c)), None)
        if src is None:
            raise SystemExit(f"missing {fn} (looked in {candidates})")
        shutil.copy2(src, os.path.join(staged_dir, fn))
    # The plans subdir itself with all fold_X dirs (checkpoint_final.pth only —
    # we strip checkpoint_latest.pth, optimizer state, .log files, etc. to keep
    # the bundle small; predictor only needs final + dataset_fingerprint.json
    # + the plans.json above).
    # The model folder is staged under nnU-Net's stock name, whatever the source folder is
    # called, so tools that know a model folder by its name (haversack) can run the bundle.
    folds = discover_folds(source_plans_dir)
    if not folds:
        raise SystemExit(f"no fold_*/checkpoint_final.pth in {source_plans_dir}")
    parts = model_folder_parts(source_plans_dir, os.path.join(staged_dir, "plans.json"), folds)
    plans_subdir = "__".join(parts)
    bundle_plans = os.path.join(staged_dir, plans_subdir)
    os.makedirs(bundle_plans, exist_ok=True)
    for fold_i, fold_src in folds:
        fold_dst = os.path.join(bundle_plans, f"fold_{fold_i}")
        os.makedirs(fold_dst, exist_ok=True)
        shutil.copy2(
            os.path.join(fold_src, "checkpoint_final.pth"),
            os.path.join(fold_dst, "checkpoint_final.pth"),
        )
    # Sibling caches nnU-Net likes to have when initializing a predictor.
    # Same dual-location search as dataset/plans.
    for opt in ("dataset_fingerprint.json",):
        for candidate in (os.path.join(parent, opt),
                          os.path.join(source_plans_dir, opt)):
            if os.path.isfile(candidate):
                shutil.copy2(candidate, os.path.join(staged_dir, opt))
                break
    return folds, plans_subdir, parts


LICENSE_URLS = {
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
}

ATTRIBUTION_HOLDER = "Isomics, Inc. and the SlicerLNQ contributors"
PROJECT_URL = "https://github.com/pieper/lnq-segmenter"


def attribution_for(card):
    """The CC BY attribution string downstream tools should render verbatim.
    Kept in the registry so consumers don't have to rebuild it themselves —
    see MODELS.md in the lnq-segmenter repo."""
    return (f"{card['display_name']} ({card['name']}@{card['version']}), "
            f"{ATTRIBUTION_HOLDER}, {card['license']}, {PROJECT_URL}")


def make_card(mg, name, version, display_name, license_str, folds,
              plans_subdir, source_plans_dir, recommended_use,
              chronicle_url, parts):
    """Build the model card JSON. Lives inside the bundle so the Slicer
    extension's 'Model info' panel can read it without round-tripping to
    chronicle. Mirrors the registry entry's metadata but with more prose
    fields (license details, recommended use, known failure modes)."""
    trainer_class, plans_identifier, configuration = parts   # from the model: model_folder_parts
    return {
        "schema": "lnq-model-card/1",
        "name": name,
        "display_name": display_name,
        "version": version,
        "license": license_str,
        "task": "lymph_node_segmentation",
        "modality": "CT",
        "trainer_class": trainer_class,
        "plans_identifier": plans_identifier,
        "configuration": configuration,
        "folds": [f for f, _ in folds],
        "label_map": {"1": "lymph_node"},
        "training_provenance": {
            "model_generation_id": mg.get("_id"),
            "model_generation_label": mg.get("label"),
            "chronicle_url": chronicle_url,
            "training_started_at":  mg.get("training_started_at"),
            "training_finished_at": mg.get("training_finished_at"),
            "training_set_size":    len(mg.get("training_annotation_ids") or []),
            "training_annotation_ids": mg.get("training_annotation_ids") or [],
            "framework": mg.get("framework"),
            "notes": mg.get("notes"),
        },
        "recommended_use": recommended_use,
        "known_limitations": [],   # fill in after Monday's review
    }


def make_registry_entry(card, weights_url_template, plans_subdir,
                        weights_assets, source_plans_dir):
    """Single entry to be appended into _registry.json's 'models' array.
    weights_assets is the list returned by zip_assets — each item has role,
    filename, sha256, size_bytes. The downloader resolves URLs by filling
    {name}, {version}, {filename} in weights_url_template."""
    return {
        "name": card["name"],
        "display_name": card["display_name"],
        "version": card["version"],
        "license": card["license"],
        "license_url": LICENSE_URLS.get(card["license"]),
        "attribution": attribution_for(card),
        "task": card["task"],
        # Model names are "<anatomy>-v<N>"; strip the lineage suffix.
        "anatomy": re.sub(r"-v\d+$", "", card["name"]).lower() or None,
        "modality": card["modality"],
        "trainer_class": card["trainer_class"],
        "plans_identifier": card["plans_identifier"],
        "configuration": card["configuration"],
        "plans_subdir": plans_subdir,
        "folds": card["folds"],
        "label_map": card["label_map"],
        "weights_url_template": weights_url_template,
        "weights_assets": [
            {"role": a["role"], "filename": a["filename"],
             "sha256": a["sha256"], "size_bytes": a["size_bytes"]}
            for a in weights_assets
        ],
        "training_provenance": {
            k: card["training_provenance"][k]
            for k in ("model_generation_id", "chronicle_url",
                       "training_set_size", "training_finished_at")
            if k in card["training_provenance"]
        },
        "publish_provenance": {
            "published_at": iso_now(),
            "published_by": socket.gethostname(),
            "source_plans_dir_at_publish": source_plans_dir,
        },
        "recommended_use": card["recommended_use"],
    }


def _write_zip(zip_path, files):
    """Write a deterministic zip. `files` is [(arcname, src_path)]."""
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for arc, src in sorted(files):
            info = zipfile.ZipInfo(arc, date_time=(2020, 1, 1, 0, 0, 0))
            with open(src, "rb") as f:
                zf.writestr(info, f.read())


def zip_assets(staged_dir, output_dir, name, version, plans_subdir, folds):
    """Produce per-fold + meta zips under the GitHub Release 2 GiB asset cap.
    Each asset, when unzipped into a single dest dir, contributes part of the
    canonical nnU-Net layout — together they reconstruct it. Returns list of
    {role, filename, path, sha256, size_bytes} dicts."""
    assets = []
    meta_arcs = []
    for fn in ("dataset.json", "plans.json", "dataset_fingerprint.json",
               "card.json"):
        src = os.path.join(staged_dir, fn)
        if os.path.isfile(src):
            meta_arcs.append((fn, src))
    meta_name = f"{name}-{version}-meta.zip"
    meta_path = os.path.join(output_dir, meta_name)
    _write_zip(meta_path, meta_arcs)
    assets.append({
        "role": "meta", "filename": meta_name, "path": meta_path,
        "sha256": sha256_file(meta_path),
        "size_bytes": os.path.getsize(meta_path),
    })
    for fold_i, _ in folds:
        arc = f"{plans_subdir}/fold_{fold_i}/checkpoint_final.pth"
        src = os.path.join(staged_dir, plans_subdir, f"fold_{fold_i}",
                           "checkpoint_final.pth")
        fold_name = f"{name}-{version}-fold{fold_i}.zip"
        fold_path = os.path.join(output_dir, fold_name)
        _write_zip(fold_path, [(arc, src)])
        assets.append({
            "role": f"fold_{fold_i}", "filename": fold_name,
            "path": fold_path, "sha256": sha256_file(fold_path),
            "size_bytes": os.path.getsize(fold_path),
        })
    return assets


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-generation-id", required=True)
    ap.add_argument("--source-path", required=True,
                    help="Local path to the nnU-Net plans dir.")
    ap.add_argument("--name", required=True, help="Bundle/registry slug.")
    ap.add_argument("--version", required=True, help="Semver of this release.")
    ap.add_argument("--display-name", required=True)
    ap.add_argument("--license", default="CC BY 4.0", dest="license_str",
                    help="SPDX-ish license string for the weights. "
                         "Defaults to the project standard, CC BY 4.0.")
    ap.add_argument("--recommended-use", default="")
    ap.add_argument("--weights-url-template",
                    default="https://github.com/pieper/lnq-segmenter/releases/download/{name}-{version}/{filename}",
                    help="Templated URL where each published asset will live. "
                         "Filled per-asset with {name}, {version}, {filename}.")
    ap.add_argument("--output-dir", default="/tmp/lnq-publish")
    ap.add_argument("--conf", default="chronicle.conf")
    args = ap.parse_args()

    conf = load_conf(args.conf)
    chronicle_url = conf.get("CHRONICLE_URL")
    if not chronicle_url and conf.get("DOMAIN_NAME"):
        chronicle_url = "https://" + conf["DOMAIN_NAME"]
    if not chronicle_url:
        print("Missing CHRONICLE_URL / DOMAIN_NAME in chronicle.conf",
              file=sys.stderr); sys.exit(2)
    chronicle = Chronicle(chronicle_url, conf.get("CHRONICLE_DB", "lnq"),
                          conf["COUCHDB_ADMIN_PASSWORD"])

    mg = chronicle.get(args.model_generation_id)
    if mg.get("type") != "ModelGeneration":
        print(f"{args.model_generation_id} is not a ModelGeneration",
              file=sys.stderr); sys.exit(2)

    # Sanity: the MG's weights_ref should point at args.source_path (after
    # the optional 'file://' prefix). Don't fail on mismatch — the source
    # may have been moved/renamed; just warn.
    declared = ((mg.get("weights_ref") or {}).get("value") or "")
    if declared.startswith("file://"):
        declared = declared[len("file://"):]
    if declared and os.path.realpath(declared) != os.path.realpath(args.source_path):
        print(f"WARNING: MG.weights_ref={declared}", file=sys.stderr)
        print(f"         --source-path={args.source_path}", file=sys.stderr)
        print(f"         (proceeding with --source-path)", file=sys.stderr)

    if not os.path.isdir(args.source_path):
        print(f"--source-path missing or not a directory: {args.source_path}",
              file=sys.stderr); sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)
    staged_dir = os.path.join(args.output_dir, args.name)
    if os.path.exists(staged_dir):
        shutil.rmtree(staged_dir)
    os.makedirs(staged_dir)
    print(f"[publish] staging into {staged_dir}", flush=True)
    folds, plans_subdir, parts = stage_bundle(args.source_path, staged_dir)
    print(f"[publish] folds = {[f for f, _ in folds]}", flush=True)
    print(f"[publish] model folder = {plans_subdir}", flush=True)

    card = make_card(
        mg=mg, name=args.name, version=args.version,
        display_name=args.display_name, license_str=args.license_str,
        folds=folds, plans_subdir=plans_subdir,
        source_plans_dir=args.source_path,
        recommended_use=args.recommended_use,
        chronicle_url=chronicle_url, parts=parts,
    )
    card_path = os.path.join(staged_dir, "card.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    print(f"[publish] writing per-fold + meta zips to {args.output_dir}",
          flush=True)
    assets = zip_assets(staged_dir, args.output_dir, args.name, args.version,
                        plans_subdir, folds)
    for a in assets:
        print(f"  {a['role']:8s}  {a['size_bytes']:>13_d} bytes  "
              f"{a['sha256'][:16]}…  {a['filename']}", flush=True)

    registry_entry = make_registry_entry(card, args.weights_url_template,
                                         plans_subdir, assets,
                                         args.source_path)
    registry_path = os.path.join(
        args.output_dir, f"{args.name}-{args.version}.registry.json")
    with open(registry_path, "w") as f:
        json.dump(registry_entry, f, indent=2)
    sha_path = os.path.join(
        args.output_dir, f"{args.name}-{args.version}.sha256")
    with open(sha_path, "w") as f:
        for a in assets:
            f.write(f"{a['sha256']}  {a['filename']}\n")

    print("\n[publish] artifacts ready:")
    print(f"  bundle dir:  {staged_dir}")
    for a in assets:
        print(f"  {a['role']:8s}    {a['path']}")
    print(f"  sha256:      {sha_path}")
    print(f"  card:        {card_path}")
    print(f"  registry:    {registry_path}")
    asset_paths = " \\\n       ".join(a["path"] for a in assets)
    print("\n[publish] next steps:")
    print(f"  1. Verify the bundle loads with nnUNetPredictor pointing at "
          f"{staged_dir}.")
    print(f"  2. gh release create {args.name}-{args.version} \\")
    print(f"       {asset_paths} \\")
    print(f"       --title \"{args.display_name} {args.version}\" \\")
    print(f"       --notes-file {card_path}")
    print(f"  3. Append {os.path.basename(registry_path)} into "
          f"lnq-segmenter/_registry.json under 'models'.")
    print(f"  4. Bump lnq-segmenter package version, publish to PyPI.")
    print(f"  5. Bump LymphNodeQuantifier extension if it ships its own copy "
          f"of the registry.")


if __name__ == "__main__":
    main()
