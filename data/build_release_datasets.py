#!/usr/bin/env python3
"""
Build the three PyGeoX training datasets into PyGeoX-cleaned/data/:

  1. PyGeoX-GCS-SFT.jsonl
       Merged geometry-constraint-solving SFT master. The 5 rejection-sampling SFT
       variants (plus the `full` base) are collapsed into ONE deduplicated file, keyed
       by (problem, completion). Each row carries a per-variant `splits` map so every
       original training set is a deterministic filter:  keep rows where
       splits[<name>].in == True, scale the loss by splits[<name>].weight.

  2. PyGeoX-CodeGen-SFT.jsonl
       NL -> PyGeoX code SFT. Each problem JSON becomes a (user: diagram description,
       assistant: ```python ... ```) pair that teaches the model to emit PyGeoX library
       code for a described diagram.

  3. PyGeoX-GCS-RL.jsonl
       RL prompts built from the problem JSONs in data/PyGeoX-GCS-RL-Code/
       (one merged file, `difficulty` field).

  4. PyGeoX-Bench.jsonl
       Lean evaluation benchmark. Each record is self-contained with only the fields
       needed to score a model: the symbolic scene (`Objs`/`Rels`/`Points`/`extra_rel`)
       that `create_scene_from_json` rebuilds for reward, plus `nl_description`,
       `pygeox_code`, `possible_solution`, and `unique_id`. Generation metadata and
       `llm_messages` are dropped.

All `source_file` paths are written RELATIVE to the project root (PyGeoX-cleaned/), so
they are invariant to where the repo lives:
    data/PyGeoX-GCS-RL-Code/<name>.json

Usage:
    python build_release_datasets.py [all|gcs_sft|codegen_sft|rl|bench]   (default: all)
"""

import json
import os
import re
import sys
import hashlib
from pathlib import Path

# ---- paths (resolved relative to this script => project-root invariant) ----
DATA_DIR = Path(__file__).resolve().parent                 # .../PyGeoX-cleaned/data
PROJECT_ROOT = DATA_DIR.parent                             # .../PyGeoX-cleaned
RS_DIR = Path(
    "/mnt/disk0/r00922822/PyGeoX-original/model_training/old_data/rejection_sampling"
)
CODE_REL = "data/PyGeoX-GCS-RL-Code"  # rel-to-root location of the problem JSONs
CODE_ABS = PROJECT_ROOT / CODE_REL
SYS_PROMPT_PATH = PROJECT_ROOT / "model_training" / "system_prompt_rl.md"

GCS_SFT_OUT = DATA_DIR / "PyGeoX-GCS-SFT.jsonl"          # merged rejection-sampling SFT master
CODEGEN_SFT_OUT = DATA_DIR / "PyGeoX-CodeGen-SFT.jsonl"  # NL -> PyGeoX code SFT
RL_OUT = DATA_DIR / "PyGeoX-GCS-RL.jsonl"

BENCH_SRC = DATA_DIR / "PyGeoX benchmark" / "json_fixed"  # source benchmark problem JSONs
BENCH_OUT = DATA_DIR / "PyGeoX-Bench.jsonl"               # lean evaluation benchmark
# fields kept in the lean benchmark (scoring-needed + nl/code/id reference)
BENCH_FIELDS = [
    "unique_id", "nl_description", "pygeox_code",
    "Objs", "Rels", "Points", "extra_rel", "possible_solution",
]

# ---- SFT variant mapping:  split-name -> rejection-sampling file (script that uses it) ----
VARIANTS = {
    "full":   "rs_full_cleaned.jsonl",                 # base / superset (no weight)
    "sar":    "rs_dense_SAR.jsonl",                    # sft_sar.sh
    "sar_sd": "rs_dense_cleaned.jsonl",                # sft_sar_sd.sh (== Qwen_32B_teacher_data/rs_dense.jsonl)
    "mse":    "rs_sse_weighted_exp_sse_fixed.jsonl",   # sft_mse.sh
    "mse_sd": "rs_mixed_sse_sparse.jsonl",             # sft_mse_sd.sh
    "sparse": "rs_sparse_cleaned.jsonl",               # sft_sparse.sh
}
TRAINABLE = [s for s in VARIANTS if s != "full"]       # the 5 reconstructable training sets


def difficulty_of(name: str):
    if name.startswith("1obj"):
        return "easy"
    if name.startswith("2obj"):
        return "medium"
    if name.startswith("3obj"):
        return "hard"
    return None


def assistant_of(messages):
    ac = ""
    for m in messages:
        if m.get("role") == "assistant":
            ac = m.get("content", "")
    return ac


def build_gcs_sft():
    print(f"[GCS-SFT] building from {len(VARIANTS)} rejection-sampling files ...")
    master = {}   # (problem_id, completion_hash) -> record
    order = []    # first-seen order
    per_split_rows = {s: 0 for s in VARIANTS}

    for split, fn in VARIANTS.items():
        path = RS_DIR / fn
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d.get("messages", [])
                pid = os.path.basename(d.get("source_file", "") or "UNKNOWN")
                h = hashlib.md5(
                    assistant_of(msgs).encode("utf-8", "ignore")
                ).hexdigest()[:12]
                key = (pid, h)
                if key not in master:
                    master[key] = {
                        "messages": msgs,
                        "problem_id": pid,
                        "source_file": f"{CODE_REL}/{pid}",
                        "difficulty": difficulty_of(pid) or d.get("difficulty"),
                        "splits": {},
                    }
                    order.append(key)
                entry = {"in": True, "reward": d.get("reward")}
                if d.get("weight") is not None:
                    entry["weight"] = d.get("weight")
                if split not in master[key]["splits"]:   # keep first completion seen per split
                    master[key]["splits"][split] = entry
                    per_split_rows[split] += 1
                n += 1
        print(f"    {split:7s} <- {fn:34s} rows={n}")

    fc = 0
    diff_dist = {"easy": 0, "medium": 0, "hard": 0, None: 0}
    with open(GCS_SFT_OUT, "w", encoding="utf-8") as out:
        for key in order:
            rec = master[key]
            present = rec["splits"]
            # fully_correct: base scheme reward == 10 (max). Unknown (null) if not in `full`.
            full_r = present.get("full", {}).get("reward")
            rec["fully_correct"] = (full_r is not None and full_r >= 9.999) if "full" in present else None
            rec["in_splits"] = [s for s in TRAINABLE if s in present]
            # materialize in:false for absent splits so every row has the full map
            for s in VARIANTS:
                present.setdefault(s, {"in": False})
            if rec["fully_correct"]:
                fc += 1
            diff_dist[rec["difficulty"] if rec["difficulty"] in diff_dist else None] += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[GCS-SFT] wrote {len(order)} rows -> {GCS_SFT_OUT}")
    print(f"      per-split (in:true): " + ", ".join(f"{s}={per_split_rows[s]}" for s in VARIANTS))
    print(f"      fully_correct(base==10): {fc}")
    print(f"      difficulty: {dict(diff_dist)}")
    return len(order)


def build_codegen_sft():
    """NL -> PyGeoX code SFT (one (user, assistant) pair per problem JSON)."""
    if not CODE_ABS.is_dir():
        print(f"[CodeGen-SFT] ERROR: {CODE_ABS} not found (extract the zip first).", file=sys.stderr)
        return None
    files = sorted(CODE_ABS.glob("*.json"))
    print(f"[CodeGen-SFT] building from {len(files)} problem JSONs ...")
    processed = skipped = 0
    diff_dist = {"easy": 0, "medium": 0, "hard": 0, None: 0}
    with open(CODEGEN_SFT_OUT, "w", encoding="utf-8") as out:
        for jf in files:
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                skipped += 1
                continue
            nl = d.get("nl_description", "")
            thinking = d.get("pygeox_sft_thinking", "")        # used as a quality filter
            code = d.get("pygeox_code_fixed", "") or d.get("pygeox_code", "")
            # normalize code (drop scene.get_object('x') -> x, add line_ prefix, fix quoted attr)
            code = re.sub(r"scene.get_object\('([^']+)'\)", r"\1", code)
            code = re.sub(r"\bline\s+([A-Z]{2,})\b", r"line_\1", code)
            code = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\.", r"\1.", code)
            if not nl or not thinking or not code:
                skipped += 1
                continue
            user = nl if nl.endswith(".") else nl + "."
            user += " Please generate PyGeoX code that represents this diagram."
            diff = difficulty_of(jf.name)
            entry = {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": f"```python\n{code}\n```"},
                ],
                "problem_id": jf.name,
                "source_file": f"{CODE_REL}/{jf.name}",
                "difficulty": diff,
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            processed += 1
            diff_dist[diff if diff in diff_dist else None] += 1
    print(f"[CodeGen-SFT] wrote {processed} rows (skipped {skipped}) -> {CODEGEN_SFT_OUT}")
    print(f"      difficulty: {dict(diff_dist)}")
    return processed


def build_rl():
    if not CODE_ABS.is_dir():
        print(f"[RL] ERROR: {CODE_ABS} not found (extract the zip first).", file=sys.stderr)
        return None
    sys_prompt = SYS_PROMPT_PATH.read_text(encoding="utf-8")
    files = sorted(CODE_ABS.glob("*.json"))
    print(f"[RL] building from {len(files)} problem JSONs ...")
    stats = {"processed": 0, "skipped": 0, "by_difficulty": {"easy": 0, "medium": 0, "hard": 0}}
    with open(RL_OUT, "w", encoding="utf-8") as out:
        for jf in files:
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                stats["skipped"] += 1
                continue
            desc = d.get("nl_description", "")
            sol = d.get("possible_solution")
            diff = difficulty_of(jf.name)
            if not desc or not sol or not diff:
                stats["skipped"] += 1
                continue
            epd = {p: [None, None] for p in sol.get("points", {}).keys()}
            ecd = {c: None for c in sol.get("circles", {}).keys()}
            user_prompt = (
                "Write python code to find the coordinates and circle radiuses for:\n"
                f"{desc}\n\n"
                "Required format of 'points' dictionary:\n"
                f"{json.dumps(epd)}\n\n"
                "Required format of 'circles' dictionary:\n"
                f"{json.dumps(ecd)}"
            )
            entry = {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "source_file": f"{CODE_REL}/{jf.name}",
                "difficulty": diff,
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stats["processed"] += 1
            stats["by_difficulty"][diff] += 1
    print(f"[RL] wrote {stats['processed']} rows (skipped {stats['skipped']}) -> {RL_OUT}")
    print(f"     by_difficulty: {stats['by_difficulty']}")
    return stats


def build_bench():
    """Lean evaluation benchmark: keep only scoring-needed fields + nl/code/id reference."""
    if not BENCH_SRC.is_dir():
        print(f"[Bench] ERROR: {BENCH_SRC} not found.", file=sys.stderr)
        return None
    files = sorted(BENCH_SRC.glob("*.json"))
    print(f"[Bench] building from {len(files)} benchmark JSONs ...")
    written = skipped = 0
    with open(BENCH_OUT, "w", encoding="utf-8") as out:
        for jf in files:
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                skipped += 1
                continue
            if not all(k in d for k in ("Objs", "Rels", "Points")):  # required to rebuild the scene
                skipped += 1
                continue
            rec = {k: d.get(k, [] if k == "extra_rel" else None) for k in BENCH_FIELDS}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"[Bench] wrote {written} rows (skipped {skipped}) -> {BENCH_OUT}")
    return written


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("gcs_sft", "all"):
        build_gcs_sft()
    if which in ("codegen_sft", "all"):
        build_codegen_sft()
    if which in ("rl", "all"):
        build_rl()
    if which in ("bench", "all"):
        build_bench()
