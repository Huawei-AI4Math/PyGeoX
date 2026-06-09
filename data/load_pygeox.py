#!/usr/bin/env python3
"""
Download and prepare the PyGeoX datasets from the Hugging Face Hub.

    https://huggingface.co/datasets/rafaelcabral96/PyGeoX

Quick start (from an empty folder):

    pip install huggingface_hub
    python load_pygeox.py --root ./PyGeoX        # download everything + extract archives

Then, in your own code:

    from load_pygeox import load_subset, resolve_source_file, reconstruct_sft_variant
    rl = list(load_subset("gcs-rl", root="./PyGeoX"))
    path = resolve_source_file(rl[0], root="./PyGeoX")     # -> problem JSON for reward scoring
    mse_sd = list(reconstruct_sft_variant("mse_sd", root="./PyGeoX"))   # one SFT training set
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

REPO_ID = "rafaelcabral96/PyGeoX"

# subset name -> path within the downloaded repo
SUBSETS = {
    "gcs-sft":     "data/PyGeoX-GCS-SFT.jsonl",
    "codegen-sft": "data/PyGeoX-CodeGen-SFT.jsonl",
    "gcs-rl":      "data/PyGeoX-GCS-RL.jsonl",
    "bench":       "data/PyGeoX-Bench.jsonl",
    "wild":        "data/PyGeoX-Wild.jsonl",
}

# archives that must be extracted in place (under data/) so that the relative
# `source_file` paths inside the jsonl files resolve to real files on disk
ZIPS = [
    "data/PyGeoX-GCS-RL-Code.zip",   # -> data/PyGeoX-GCS-RL-Code/<id>.json   (RL + bench reward ground truth)
    "data/PyGeoX-Wild-Code.zip",     # -> data/PyGeoX-Wild-Code/problem_<id>.json  (Wild ground-truth code)
]


def download(root: str = "./PyGeoX") -> str:
    """Download the whole dataset repo into `root`, preserving the data/ layout."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=root)


def extract_zips(root: str = "./PyGeoX") -> None:
    """Extract the code/answer archives next to themselves under data/."""
    for rel in ZIPS:
        zpath = Path(root) / rel
        if not zpath.exists():
            print(f"  skip (missing): {rel}")
            continue
        with zipfile.ZipFile(zpath) as z:
            z.extractall(zpath.parent)          # each zip already nests its own top-level folder
        print(f"  extracted {rel} -> {zpath.parent / zpath.stem}/")


def load_jsonl(path):
    """Yield one dict per line of a .jsonl file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_subset(name: str, root: str = "./PyGeoX"):
    """Yield records from a subset by name (one of SUBSETS)."""
    if name not in SUBSETS:
        raise ValueError(f"unknown subset {name!r}; choose from {list(SUBSETS)}")
    yield from load_jsonl(Path(root) / SUBSETS[name])


def resolve_source_file(record: dict, root: str = "./PyGeoX") -> str:
    """Absolute path to a record's problem JSON (valid after extract_zips)."""
    return str(Path(root) / record["source_file"])


def reconstruct_sft_variant(variant: str, root: str = "./PyGeoX"):
    """Yield (messages, weight) for one SFT training set from the merged gcs-sft master.

    `variant` is one of: sar, sar_sd, mse, mse_sd, sparse, full.
    """
    for r in load_subset("gcs-sft", root):
        s = r.get("splits", {}).get(variant)
        if s and s.get("in"):
            yield r["messages"], s.get("weight", 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download + prepare PyGeoX datasets from the HF Hub.")
    ap.add_argument("--root", default="./PyGeoX", help="local directory to download into (default: ./PyGeoX)")
    ap.add_argument("--no-extract", action="store_true", help="do not extract the code/answer archives")
    args = ap.parse_args()

    print(f"downloading {REPO_ID} -> {args.root} ...")
    download(args.root)
    if not args.no_extract:
        print("extracting code/answer archives ...")
        extract_zips(args.root)

    print("\nsummary (rows per subset):")
    for name in SUBSETS:
        try:
            n = sum(1 for _ in load_subset(name, args.root))
        except FileNotFoundError:
            n = "missing"
        print(f"  {name:12s} {n}")
    print(f"\nready. e.g.:  from load_pygeox import load_subset; "
          f"list(load_subset('gcs-rl', root={args.root!r}))")
