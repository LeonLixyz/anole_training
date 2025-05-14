#!/usr/bin/env python
"""
A utility script to merge multiple HuggingFace datasets into a single JSON file
suitable for interleaved text–image reasoning tasks.

Key features
------------
1. **Skip already‑processed datasets** by checking both `all_datasets.json` *and*
   the presence of a dataset‑specific folder.
2. **Shuffle the combined dataset on *every* run**, whether or not new data is
   added.  This ensures training randomness without requiring an extra step.

Usage (example):
    python format_datasets.py --output_dir formatted_data

Add `--debug` to limit each dataset to 5 examples while iterating.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
from io import BytesIO
from typing import Any, Dict, List, Set

from PIL import Image
from datasets import concatenate_datasets, load_dataset

###############################################################################
# Helper utilities
###############################################################################

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert a base‑64 string (optionally with a data‑URI prefix) to a PIL image."""
    if not base64_str.startswith("data:"):
        base64_str = f"data:image/jpeg;base64,{base64_str}"
    _, encoded = base64_str.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _serialise(example: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce an example to a JSON‑serialisable form."""
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
# Core formatting logic
###############################################################################

def format_dataset(dataset, output_dir: str) -> List[Dict[str, Any]]:
    """Format a single HuggingFace dataset split into Zebra‑CoT JSON structure."""

    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    formatted: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset):
        question = item.get("question", "")
        reasoning = item.get("reasoning", "")
        answer = item.get("answer", "")

        print(f"  • Example {idx + 1}/{len(dataset)}")

        # ----- problem images -------------------------------------------------
        p_imgs, p_paths = [], []
        for i in range(1, 5):
            b64_key, img_key = f"problem_image_{i}_base64", f"problem_image_{i}"
            try:
                if b64_key in item and item[b64_key]:
                    img = base64_to_image(item[b64_key])
                elif img_key in item and item[img_key] is not None and hasattr(item[img_key], "size"):
                    img = item[img_key]
                else:
                    continue
                path = os.path.join(image_dir, f"problem_{idx}_{i}.jpg")
                img.save(path)
                p_imgs.append(img)
                p_paths.append(path)
            except Exception as e:
                print(f"    ⚠️  Problem image {i}: {e}")

        # ----- reasoning images ----------------------------------------------
        r_imgs, r_paths = [], []
        for i in range(1, 5):
            b64_key, img_key = f"reasoning_image_{i}_base64", f"reasoning_image_{i}"
            try:
                if b64_key in item and item[b64_key]:
                    img = base64_to_image(item[b64_key])
                elif img_key in item and item[img_key] is not None and hasattr(item[img_key], "size"):
                    img = item[img_key]
                else:
                    continue
                path = os.path.join(image_dir, f"reasoning_{idx}_{i}.jpg")
                img.save(path)
                r_imgs.append(img)
                r_paths.append(path)
            except Exception as e:
                print(f"    ⚠️  Reasoning image {i}: {e}")

        # ----- placeholder replacement ---------------------------------------
        q_text = question
        for i in range(1, len(p_imgs) + 1):
            q_text = q_text.replace(f"<image_start>[problem_image_{i}]<image_end>", "<image>")

        r_text = reasoning
        for i in range(1, len(r_imgs) + 1):
            r_text = r_text.replace(f"<image_start>[reasoning_image_{i}]<image_end>", "<image>")

        if answer and "FINAL ANSWER" not in r_text:
            r_text += f"\n\nFINAL ANSWER: {answer}"

        formatted.append(
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

    # Write per‑dataset JSON snapshot
    with open(os.path.join(output_dir, "formatted_data.json"), "w") as fh:
        json.dump({"train": [_serialise(x) for x in formatted]}, fh, indent=2)
    print(f"  → Saved {len(formatted)} formatted examples to {output_dir}/formatted_data.json")
    return formatted

###############################################################################
# Main driver
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and shuffle VLM reasoning datasets")
    parser.add_argument("--dataset_names", nargs="+", type=str, default=[
        "vlm-reasoning-cot/ARC-AGI",
        "vlm-reasoning-cot/visual_jigsaw",
        "vlm-reasoning-cot/graph",
        "vlm-reasoning-cot/Mazes",
        "vlm-reasoning-cot/Physics",
        "vlm-reasoning-cot/Tetris",
        "vlm-reasoning-cot/MATH_geometry",
        "vlm-reasoning-cot/chem",
        "vlm-reasoning-cot/robot_planning",
        "vlm-reasoning-cot/checker",
        "vlm-reasoning-cot/connect_4",
        "vlm-reasoning-cot/cipher",
        "vlm-reasoning-cot/embodied_cot",
        "vlm-reasoning-cot/visual_search",
        "vlm-reasoning-cot/chess",
    ])
    parser.add_argument("--output_dir", default="formatted_data", type=str, help="Output folder")
    parser.add_argument("--debug", action="store_true", help="Process only 5 examples per dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    combined_json = os.path.join(args.output_dir, "all_datasets.json")

    # ------------------------------------------------------------------
    # Load existing combined data (if any)
    # ------------------------------------------------------------------
    processed: Set[str] = set()
    all_examples: List[Dict[str, Any]] = []
    if os.path.exists(combined_json):
        print(f"Found existing combined file: {combined_json} — loading…")
        with open(combined_json, "r") as fh:
            obj = json.load(fh)
        all_examples.extend(obj.get("train", []))
        processed.update({ex.get("dataset_source", "unknown") for ex in all_examples})
        print(f"  → {len(processed)} datasets already recorded")

    # Also skip datasets whose folder already exists (guards partial runs)
    for name in args.dataset_names:
        if os.path.isdir(os.path.join(args.output_dir, name.split("/")[-1])):
            processed.add(name)

    # ------------------------------------------------------------------
    # Iterate over requested datasets
    # ------------------------------------------------------------------
    for dname in args.dataset_names:
        if dname in processed:
            print(f"⏩  Skipping {dname} (already processed)")
            continue

        print("\n" + "=" * 60)
        print(f"Processing {dname}")
        print("=" * 60)

        out_dir = os.path.join(args.output_dir, dname.split("/")[-1])
        os.makedirs(out_dir, exist_ok=True)

        # Load dataset (handle multi‑config gracefully)
        try:
            try:
                ds_dict = load_dataset(dname, trust_remote_code=True)
            except Exception as e:
                if "Config name is missing" in str(e):
                    cfgs = re.search(r"Please pick one among the available configs: \[(.*?)\]", str(e))
                    if not cfgs:
                        raise
                    cfg_list = [c.strip("'\"") for c in cfgs.group(1).split(", ")]
                    parts = [load_dataset(dname, cfg, trust_remote_code=True)["train"] for cfg in cfg_list]
                    ds_dict = {"train": concatenate_datasets(parts)}
                else:
                    raise
        except Exception as e:
            print(f"⚠️  Failed to load {dname}: {e}")
            continue

        train_split = ds_dict["train"]
        print(f"  → Loaded {len(train_split)} rows")
        if args.debug:
            train_split = train_split.select(range(min(5, len(train_split))))
            print(f"  • Debug: using {len(train_split)} rows")

        formatted = format_dataset(train_split, out_dir)
        for ex in formatted:
            ex["dataset_source"] = dname
        all_examples.extend(formatted)

    # ------------------------------------------------------------------
    # SHUFFLE!  Always randomise order, even if no new data were added.
    # ------------------------------------------------------------------
    random.shuffle(all_examples)

    # ------------------------------------------------------------------
    # Write combined JSON
    # ------------------------------------------------------------------
    with open(combined_json, "w") as fh:
        json.dump({"train": [_serialise(ex) for ex in all_examples]}, fh, indent=2)

    print("\n✅  All done — combined dataset contains", len(all_examples), "examples")
    print("    Shuffled and saved to", combined_json)


if __name__ == "__main__":
    main()
