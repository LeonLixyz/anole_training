#!/usr/bin/env python
"""
merge_datasets.py  (v2)

Aggregate several top‑level Hugging Face **dataset repos** – pulling *every*
config (a.k.a. sub‑dataset) under each – and merge them into one shuffled
`all_datasets.json` for interleaved text–image reasoning.

Why v2?
--------
The first cut tried to load a sub‑dataset by concatenating
`repo_id/config_name` (which is not how `datasets.load_dataset` works).
This release keeps the correct two‑argument form:

```python
load_dataset(repo_id, config_name)
```

Other tweaks
------------
* Uses an explicit `(repo_id, config)` tuple list under the hood.
* Generates deterministic *safe* folder names like
  `vlm-reasoning-cot__Visual_Logic_and_Strategic_Games__Tetris`.
* `processed` tracking keyed on the same tuple so incremental runs work.

Usage example
-------------
```bash
python merge_datasets.py \
  --repo_ids \
      vlm-reasoning-cot/Scientific_Reasoning \
      vlm-reasoning-cot/2D_Visual_Reasoning \
      vlm-reasoning-cot/3D_Visual_Reasoning \
      vlm-reasoning-cot/Visual_Logic_and_Strategic_Games
```
Add `--debug` for a 5‑row smoke test per config.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Set, Tuple

from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from PIL import Image  # noqa: F401  # Pillow may still be needed downstream

###############################################################################
# Helper utilities
###############################################################################

def _serialise(example: Dict[str, Any]) -> Dict[str, Any]:
    """Strip PIL objects & keep only JSON‑friendly pieces."""
    return {
        "input_text": example["input_text"],
        "input_img_paths": example["input_img_paths"],
        "label_text": example["label_text"],
        "label_img_paths": example["label_img_paths"],
        "task": example["task"],
        "train_task": example["train_task"],
        "dataset_source": example.get("dataset_source", "unknown"),
    }

###############################################################################
# Core formatting logic (unchanged)
###############################################################################

def format_dataset(dataset, output_dir: str):
    """Format a single HF split into Zebra‑CoT style; return list of dicts."""
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    out: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset):
        q = item.get("question", "")
        r = item.get("reasoning", "")
        a = item.get("answer", "")

        # -------- images (problem & reasoning) -----------------------------
        def _grab_images(prefix: str):
            imgs, paths = [], []
            for i in range(1, 5):
                img = item.get(f"{prefix}_image_{i}")
                if img is None or not hasattr(img, "save"):
                    continue
                pth = os.path.join(img_dir, f"{prefix}_{idx}_{i}.jpg")
                try:
                    img.save(pth)
                    imgs.append(img)
                    paths.append(pth)
                except Exception as e:  # noqa: BLE001
                    print(f"    ⚠️  {prefix} image {i}: {e}")
            return imgs, paths

        p_imgs, p_paths = _grab_images("problem")
        r_imgs, r_paths = _grab_images("reasoning")

        q_text = q
        for i in range(1, len(p_imgs) + 1):
            q_text = q_text.replace(
                f"<image_start>[problem_image_{i}]<image_end>", "<image>"
            )
        r_text = r
        for i in range(1, len(r_imgs) + 1):
            r_text = r_text.replace(
                f"<image_start>[reasoning_image_{i}]<image_end>", "<image>"
            )
        if a and "FINAL ANSWER" not in r_text:
            r_text += f"\n\nFINAL ANSWER: {a}"

        out.append(
            {
                "input_text": f"QUESTION:\n{q_text}",
                "input_imgs": p_imgs,
                "input_img_paths": p_paths,
                "label_text": r_text,
                "label_imgs": r_imgs,
                "label_img_paths": r_paths,
                "task": "reasoning",
                "train_task": "interleaved_reasoning",
            }
        )

    with open(os.path.join(output_dir, "formatted_data.json"), "w") as fh:
        json.dump({"train": [_serialise(x) for x in out]}, fh, indent=2)
    print(f"  → Saved {len(out)} formatted examples → {output_dir}/formatted_data.json")
    return out

###############################################################################
# Main driver
###############################################################################

def main() -> None:
    p = argparse.ArgumentParser("Merge HF repos (all configs) → Zebra‑CoT JSON")
    p.add_argument("--repo_ids", nargs="+", required=True, help="Top‑level dataset repos")
    p.add_argument("--output_dir", default="formatted_data")
    p.add_argument("--output_file", default="train_dataset.json", help="Name of the combined output JSON file")
    p.add_argument("--debug", action="store_true", help="5‑row smoke test per config")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    combined_json = os.path.join(args.output_dir, args.output_file)

    # ------------------------------------------------------------------
    # Enumerate (repo, cfg) pairs
    # ------------------------------------------------------------------
    datasets_to_load: List[Tuple[str, str | None]] = []
    for repo in args.repo_ids:
        try:
            cfgs = get_dataset_config_names(repo, trust_remote_code=True)
        except Exception:
            cfgs = None
        if cfgs:
            datasets_to_load.extend([(repo, cfg) for cfg in cfgs])
        else:
            datasets_to_load.append((repo, None))

    print("Discovered", len(datasets_to_load), "dataset splits:")
    for rep, cfg in datasets_to_load:
        print("  •", rep if cfg is None else f"{rep} :: {cfg}")

    # ------------------------------------------------------------------
    # Recover previous state (incremental runs)
    # ------------------------------------------------------------------
    processed: Set[Tuple[str, str | None]] = set()
    all_examples: List[Dict[str, Any]] = []
    if os.path.exists(combined_json):
        with open(combined_json) as fh:
            obj = json.load(fh)
        all_examples.extend(obj.get("train", []))
        for ex in all_examples:
            if ex.get("dataset_source"):
                parts = ex.get("dataset_source", "").split(":")
                repo_id = parts[0]
                config = parts[1] if len(parts) > 1 else None
                processed.add((repo_id, config))

    # also honour existing folders
    for repo, cfg in datasets_to_load:
        safe = repo.replace("/", "__") + (f"__{cfg}" if cfg else "")
        if os.path.isdir(os.path.join(args.output_dir, safe)):
            processed.add((repo, cfg))

    # ------------------------------------------------------------------
    # Iterate
    # ------------------------------------------------------------------
    for repo, cfg in datasets_to_load:
        if (repo, cfg) in processed:
            print(f"⏩  Skipping {repo} :: {cfg or '[default]'} (already done)")
            continue

        print("\n" + "=" * 60)
        print(f"Processing {repo} :: {cfg or '[default]'}")
        print("=" * 60)

        safe = repo.replace("/", "__") + (f"__{cfg}" if cfg else "")
        out_dir = os.path.join(args.output_dir, safe)
        os.makedirs(out_dir, exist_ok=True)

        # load
        try:
            if cfg is None:
                ds_dict = load_dataset(repo, trust_remote_code=True)
            else:
                ds_dict = load_dataset(repo, cfg, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Failed to load {repo} :: {cfg}: {e}")
            continue

        train_split = ds_dict.get("train") or next(iter(ds_dict.values()))
        print(f"  → Loaded {len(train_split)} rows")
        if args.debug:
            train_split = train_split.select(range(min(5, len(train_split))))
            print("  • Debug: using", len(train_split), "rows")

        formatted = format_dataset(train_split, out_dir)
        for ex in formatted:
            ex["dataset_source"] = f"{repo}:{cfg or 'default'}"
        all_examples.extend(formatted)

    # ------------------------------------------------------------------
    # Shuffle & dump combined
    # ------------------------------------------------------------------
    random.shuffle(all_examples)
    with open(combined_json, "w") as fh:
        json.dump({"train": [_serialise(x) for x in all_examples]}, fh, indent=2)

    print("\n✅  Done – combined dataset has", len(all_examples), f"examples → {combined_json}")


if __name__ == "__main__":
    main()
