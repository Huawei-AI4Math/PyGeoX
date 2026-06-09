# PyGeoX Datasets — Download & Usage Guide

Datasets for training and evaluating language models on **geometric constraint solving**
(read a natural-language diagram description → produce point coordinates / circle radii
that satisfy the constraints), plus a supervised set for generating PyGeoX scene code.

> Hosted on the Hugging Face Hub: **https://huggingface.co/datasets/rafaelcabral96/PyGeoX**
> This guide assumes you are starting from an **empty folder** and want to fetch and use the data.

## Contents

| File (in `data/`) | Subset | Rows | Notes |
|---|---|---|---|
| `PyGeoX-GCS-SFT.jsonl` | `gcs-sft` | 15,738 | SFT — solve geometry (NL → coordinates), merged master with per-variant labels |
| `PyGeoX-CodeGen-SFT.jsonl` | `codegen-sft` | 46,977 | SFT — NL diagram description → PyGeoX code |
| `PyGeoX-GCS-RL.jsonl` | `gcs-rl` | 46,977 | RL — prompts for geometry constraint solving |
| `PyGeoX-Bench.jsonl` | `bench` | 300 | Evaluation benchmark (self-contained) |
| `PyGeoX-Wild.jsonl` | `wild` | 200 | Evaluation — real-world school-geometry questions |
| `PyGeoX-GCS-RL-Code.zip` | — | 46,977 | Problem-definition JSONs (reward ground truth) for `gcs-rl` / `bench` |
| `PyGeoX-Wild-Code.zip` | — | 200 | Ground-truth `full_code` answers for `wild` |

---

## 1. Download

### Option A — helper script (recommended: downloads **and** extracts the archives)

```bash
pip install huggingface_hub
python load_pygeox.py --root ./PyGeoX
```

This fetches everything into `./PyGeoX/` and unzips the two code/answer archives in place,
so the relative `source_file` paths resolve. (`load_pygeox.py` is in this dataset repo.)

### Option B — Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
hf download rafaelcabral96/PyGeoX --repo-type dataset --local-dir ./PyGeoX
cd ./PyGeoX/data && unzip -q PyGeoX-GCS-RL-Code.zip && unzip -q PyGeoX-Wild-Code.zip
```

### Option C — stream individual subsets with `datasets`

```python
from datasets import load_dataset
ds = load_dataset("rafaelcabral96/PyGeoX", "gcs-rl")    # or: gcs-sft, codegen-sft, bench, wild
```

> **Note.** Several subsets contain free-form nested objects (`Objs`, `possible_solution`,
> `splits`) whose keys vary per example. The reward / eval harness **reads the raw `.jsonl`
> line-by-line with `json`** — that always works; the HF viewer may not fully render those
> fields. Options A/B give you the raw files for exactly this.

### Resolving `source_file`

Every `source_file` is **relative to the download root** (e.g.
`data/PyGeoX-GCS-RL-Code/<id>.json`), so the data is invariant to where you put it.
After extracting the archives, resolve a path by joining with the root — or run
reward/eval jobs with the working directory set to the root.

---

## 2. Load & process — `load_pygeox.py`

```python
from load_pygeox import (
    download, extract_zips, load_subset, resolve_source_file, reconstruct_sft_variant,
)

# one-time fetch (or use the CLI in step 1):
download("./PyGeoX"); extract_zips("./PyGeoX")

# iterate any subset:
for rec in load_subset("gcs-rl", root="./PyGeoX"):
    src = resolve_source_file(rec, root="./PyGeoX")   # absolute path to the reward ground-truth JSON
    ...

# rebuild any of the 5 SFT training sets from the merged gcs-sft master:
for messages, weight in reconstruct_sft_variant("mse_sd", root="./PyGeoX"):
    ...   # train on `messages`, scale loss by `weight`
```

---

## 3. Subset details

### `gcs-sft` — geometry-solving SFT (merged master)

One deduplicated file keyed by (problem, completion). Each row's `splits` map records, per
training variant, whether that variant includes the row and with what reward/weight, so every
original training set is a deterministic filter.

````json
{
  "messages": [ {"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", "content": "<think>...</think>\n```python\n...\n```"} ],
  "problem_id": "1obj_2rel_2extra_gen17338.json",
  "source_file": "data/PyGeoX-GCS-RL-Code/1obj_2rel_2extra_gen17338.json",
  "difficulty": "easy",
  "fully_correct": true,
  "in_splits": ["sar", "sar_sd", "mse", "mse_sd", "sparse"],
  "splits": {
    "full":   {"in": true, "reward": 10.0},
    "sar":    {"in": true, "reward": 6.0,  "weight": 1.0},
    "sar_sd": {"in": true, "reward": 10.0, "weight": 1.0},
    "mse":    {"in": true, "reward": 10.0, "weight": 1.0},
    "mse_sd": {"in": true, "reward": 10.0, "weight": 1.0},
    "sparse": {"in": true, "reward": 10.0}
  }
}
````

| split | reward scheme | rows |
|---|---|---|
| `sar` | dense (SAR) | 10,248 |
| `sar_sd` | dense | 10,781 |
| `mse` | SSE-weighted | 9,809 |
| `mse_sd` | mixed SSE + sparse | 9,809 |
| `sparse` | sparse / binary | 4,971 |
| `full` | base superset | 15,666 |

`fully_correct` is `true` when the base (`full`) reward is maximal (`== 10`); `null` for the
few rows not present in `full`.

### `codegen-sft` — NL → PyGeoX code

`messages` = (user: diagram description + "Please generate PyGeoX code…", assistant:
a ```python``` block), plus `problem_id`, `source_file`, `difficulty`.

### `gcs-rl` — RL prompts

`messages` (system + user), `source_file` (→ reward ground truth in `PyGeoX-GCS-RL-Code/`),
`difficulty` (`easy`/`medium`/`hard` for 1/2/3 primary objects).

### `bench` — evaluation benchmark (self-contained)

Each record carries everything needed to score a model inline:
`unique_id, nl_description, pygeox_code, Objs, Rels, Points, extra_rel, possible_solution`.
Rebuild the scene directly from a record (no file needed) and score predicted coordinates:

```python
from pygeox.synthetic.llm_client import create_scene_from_json
scene = create_scene_from_json(domain=10, json_data=record, generate_objective_function=True)
reward, details = scene.reward.reward_function(pred_points, pred_circles)
```

### `wild` — real-world school geometry

200 questions from MathVerse, ZhongKaoGeo, and MathVista (`id, question, source, source_id`).
Ground truth is executable PyGeoX `full_code`, shipped in `PyGeoX-Wild-Code.zip` (one
`problem_<id>.json` per question, keyed 1:1 by `id`, each with `full_code` +
`possible_solution`); scored by *executing* that code.

---

## 4. Computing rewards (RL training / benchmark evaluation)

```python
from model_training.reward_func_openrlhf import llm_reward_function_new

rewards = llm_reward_function_new(
    prompts=[""] * len(completions),   # not used
    completions=completions,           # each must contain <think>...</think> and a ```python block
    source_file=source_files,          # resolve each row's relative source_file against the download root
    max_workers=8,
    dense_reward=True,                 # continuous reward (-5..10) instead of binary
)
```

The reward rebuilds the scene from the problem JSON's `Objs` / `Rels` / `Points` /
`extra_rel` fields (via `create_scene_from_json`), so make sure
`PyGeoX-GCS-RL-Code.zip` has been extracted.

---

## Citation

```bibtex
@software{pygeox_datasets,
  title  = {PyGeoX: Datasets for Geometric Constraint Solving},
  author = {Rafael Cabral},
  year   = {2026},
  url    = {https://huggingface.co/datasets/rafaelcabral96/PyGeoX}
}
```

## License

MIT
