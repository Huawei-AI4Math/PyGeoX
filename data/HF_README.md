---
license: mit
language:
  - en
  - zh
task_categories:
  - text-generation
tags:
  - geometry
  - geometric-constraint-solving
  - reasoning
  - synthetic-data
  - math
pretty_name: PyGeoX
size_categories:
  - 10K<n<100K
configs:
  - config_name: gcs-sft
    data_files: data/PyGeoX-GCS-SFT.jsonl
  - config_name: codegen-sft
    data_files: data/PyGeoX-CodeGen-SFT.jsonl
  - config_name: gcs-rl
    data_files: data/PyGeoX-GCS-RL.jsonl
  - config_name: bench
    data_files: data/PyGeoX-Bench.jsonl
  - config_name: wild
    data_files: data/PyGeoX-Wild.jsonl
---

# PyGeoX

Datasets for training and evaluating language models on **geometric constraint solving** —
reading a natural-language description of a diagram and producing the point coordinates /
circle radii that satisfy the stated constraints — plus a supervised set for generating
PyGeoX scene code.

## Subsets

| Config | Rows | Task |
|--------|------|------|
| `gcs-sft` | 15,738 | SFT — solve geometry (NL → coordinates), merged master with per-variant reward labels |
| `codegen-sft` | 46,977 | SFT — NL diagram description → PyGeoX code |
| `gcs-rl` | 46,977 | RL — prompts for geometry constraint solving |
| `bench` | 300 | Evaluation benchmark (self-contained) |
| `wild` | 200 | Evaluation — real-world school-geometry questions |

```python
from datasets import load_dataset
ds = load_dataset("rafaelcabral96/PyGeoX", "gcs-rl")
```

> **How this data is meant to be used.** Several subsets contain free-form nested objects
> (e.g. `Objs`, `possible_solution`) whose keys vary per example. The reward / evaluation
> harness **downloads the raw `.jsonl` files and parses them line-by-line with `json`** —
> that always works. The Hugging Face viewer may not fully render the nested fields.

## Download (raw files + helper)

For training/eval you want the raw files (see the note above). Either use the helper
script shipped in this repo, or the CLI:

```bash
pip install huggingface_hub
python load_pygeox.py --root ./PyGeoX        # downloads everything + extracts the archives
# --- or ---
hf download rafaelcabral96/PyGeoX --repo-type dataset --local-dir ./PyGeoX
cd ./PyGeoX/data && unzip -q PyGeoX-GCS-RL-Code.zip && unzip -q PyGeoX-Wild-Code.zip
```

See `README_DATA.md` in this repo for the full download & usage guide.

## Reward ground truth (`PyGeoX-GCS-RL-Code.zip`)

The problem definitions used to score RL and benchmark outputs are shipped as
`data/PyGeoX-GCS-RL-Code.zip`. The `source_file` field of each `gcs-rl` row points into
this folder. After downloading the repo, extract it **inside `data/`** so the relative
paths resolve:

```bash
cd data && unzip PyGeoX-GCS-RL-Code.zip      # -> data/PyGeoX-GCS-RL-Code/<id>.json
```

All `source_file` paths are written relative to the repo root
(`data/PyGeoX-GCS-RL-Code/<name>.json`), so they are invariant to where the repo lives.

## Subset details

### `gcs-sft` — geometry-solving SFT (merged master)
One deduplicated file keyed by (problem, completion). Each row's `splits` map records, per
training variant (`sar`, `sar_sd`, `mse`, `mse_sd`, `sparse`, plus the `full` base),
whether that variant includes it and with what reward/weight. Reconstruct any variant:

```python
rows    = [r for r in ds if r["splits"]["mse_sd"]["in"]]
weights = [r["splits"]["mse_sd"].get("weight", 1.0) for r in rows]
```

### `codegen-sft` — NL → PyGeoX code
`messages` = (user: diagram description + "Please generate PyGeoX code…", assistant:
```python … ``` block), plus `problem_id`, `source_file`, `difficulty`.

### `gcs-rl` — RL prompts
`messages` (system + user), `source_file` (→ reward ground truth), `difficulty`
(`easy`/`medium`/`hard` for 1/2/3 primary objects).

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
200 questions from MathVerse, ZhongKaoGeo, and MathVista (`id, question, source,
source_id`). Ground truth is executable PyGeoX `full_code`, shipped in
`data/PyGeoX-Wild-Code.zip` (one `problem_<id>.json` per question, keyed 1:1 by `id`,
each with `full_code` + `possible_solution`). Extract inside `data/`:

```bash
cd data && unzip PyGeoX-Wild-Code.zip    # -> data/PyGeoX-Wild-Code/problem_<id>.json
```

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
