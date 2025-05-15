#!/usr/bin/env python
"""
Utility to aggregate HuggingFace VLM-reasoning datasets into a single JSON with
(train / eval) splits.

2025-05-14  –  *chunk-aware* revision
2025-05-15  –  *cached-source* hot-fix #1
2025-05-15  –  *cached-source* hot-fix #2  ← this file
     · overwrite stale dataset_source values from old caches
     · drop stale category so it’s recomputed
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List

from PIL import Image
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

###############################################################################
# Console helpers
###############################################################################
_lock = threading.RLock()


def log(*args, **kw):
    with _lock:
        print(*args, **kw)


###############################################################################
# Path helpers
###############################################################################


def done_flag(dir_: str) -> bool:
    fp = os.path.join(dir_, "formatted_data.json")
    return os.path.isfile(fp) and os.path.getsize(fp) > 10


def canon_category(src: str) -> str:
    """Extremely simple string-based category assignment that merges chunks."""
    s = re.sub(r"/chunk_\d+$", "", src.lower())
    s = re.sub(r"/default$", "", s)

    checks = {
        "visual search": "visual_search",
        "visual_jigsaw": "visual_jigsaw",
        "arc-agi": "arc_agi",
        "graph": "graph",
        "mazes": "mazes",
        "physics": "physics",
        "tetris": "tetris",
        "math_geometry": "math_geometry",
        "chem": "chem",
        "robot_planning": "robot_planning",
        "checkers": "checkers",
        "connect4": "connect4",
        "ciphers": "ciphers",
        "embodied-cot": "embodied_cot",
        "chess": "chess",
        "raven_cot": "raven_cot",
        "competitive_cs": "competitive_cs",
        "zebra-cot": "zebra_cot",
    }
    for k, v in checks.items():
        if k in s:
            return v

    if "vlm-reasoning-cot/" in s:
        return s.split("vlm-reasoning-cot/")[1].split("/")[0].replace("-", "_")

    return s.split("/")[-1].replace("-", "_")


###############################################################################
# Image + serialisation utils
###############################################################################


def b64_to_img(b64: str):
    if not b64.startswith("data:"):
        b64 = f"data:image/jpeg;base64,{b64}"
    return Image.open(BytesIO(base64.b64decode(b64.split(",", 1)[1]))).convert("RGB")


def pick(ex: Dict[str, Any]):
    keep = (
        "input_text",
        "input_img_paths",
        "label_text",
        "label_img_paths",
        "task",
        "train_task",
        "dataset_source",
        "category",
    )
    return {k: ex[k] for k in keep}


###############################################################################
# Core formatter
###############################################################################


def format_split(split, out_dir: str, ds_src: str):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    res = []
    total = len(split)
    for i, row in enumerate(split):
        if i < 10 or i % 100 == 0 or i + 1 == total:
            log(f"      • {i + 1}/{total}")

        q, r, a = row.get("question", ""), row.get("reasoning", ""), row.get("answer", "")

        def save(kind):
            paths = []
            for j in range(1, 5):
                for key in (f"{kind}_image_{j}_base64", f"{kind}_image_{j}"):
                    blob = row.get(key)
                    if not blob:
                        continue
                    try:
                        img = b64_to_img(blob) if "base64" in key else blob
                        p = os.path.join(img_dir, f"{kind}_{i}_{j}.jpg")
                        img.save(p)
                        paths.append(p)
                        break
                    except Exception as e:
                        log(f"        ⚠️  {kind} {j}: {e}")
            return paths

        p_paths, r_paths = save("problem"), save("reasoning")
        for j in range(1, len(p_paths) + 1):
            q = q.replace(f"<image_start>[problem_image_{j}]<image_end>", "<image>")
        for j in range(1, len(r_paths) + 1):
            r = r.replace(f"<image_start>[reasoning_image_{j}]<image_end>", "<image>")
        if a and "FINAL ANSWER" not in r:
            r += f"\n\nFINAL ANSWER: {a}"

        res.append(
            {
                "input_text": f"QUESTION:\n{q}",
                "input_img_paths": p_paths,
                "label_text": r,
                "label_img_paths": r_paths,
                "task": "reasoning",
                "train_task": "interleaved_reasoning",
                "dataset_source": ds_src,
                "category": canon_category(ds_src),
            }
        )

    json.dump({"train": [pick(x) for x in res]}, open(os.path.join(out_dir, "formatted_data.json"), "w"), indent=2)
    log(f"      → Saved {len(res)} examples → {out_dir}/formatted_data.json")
    return res


###############################################################################
# Download worker
###############################################################################


def handle_dataset(name: str, cfg: str | None, odir: str, args, bucket):
    src = f"{name}/{cfg}" if cfg else name
    log("\n" + "=" * 60, f"\nProcessing {src}\n", "=" * 60)
    try:
        data = load_dataset(name, cfg, trust_remote_code=True)
        split = data["train"] if "train" in data else concatenate_datasets(list(data.values()))
        if args.debug:
            split = split.select(range(min(5, len(split))))
        bucket.extend(format_split(split, odir, src))
    except Exception as e:
        log(f"⚠️  Failed {src}: {e}")


###############################################################################
# Main
###############################################################################


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_file", default="datasets.json")
    ap.add_argument("--output_dir", default="formatted_data")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg_json = json.load(open(args.datasets_file))
    root_out = args.output_dir
    os.makedirs(root_out, exist_ok=True)
    datasets = cfg_json["datasets"]
    explicit_sub = cfg_json.get("subdatasets", {})

    merged: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = []
        for ds in datasets:
            # ------------------------------------------------------------------
            # A) datasets that list their sub-configs explicitly (e.g. Zebra-CoT)
            # ------------------------------------------------------------------
            if ds in explicit_sub:
                ds_dir = os.path.join(root_out, ds.split("/")[-1])
                os.makedirs(ds_dir, exist_ok=True)
                for sub in explicit_sub[ds]:
                    out_dir = os.path.join(ds_dir, sub.replace("/", "_"))
                    if done_flag(out_dir):
                        merged.extend(load_cached_examples(out_dir, f"{ds}/{sub}"))
                        continue
                    futs.append(pool.submit(handle_dataset, ds, sub, out_dir, args, merged))
                continue

            # ------------------------------------------------------------------
            # B) everything else – maybe multi-config, maybe not
            # ------------------------------------------------------------------
            try:
                cfgs = get_dataset_config_names(ds)
            except Exception:
                cfgs = []

            # B-1) single-config
            if not cfgs:
                out_dir = os.path.join(root_out, ds.split("/")[-1])
                if done_flag(out_dir):
                    merged.extend(load_cached_examples(out_dir, ds))
                else:
                    futs.append(pool.submit(handle_dataset, ds, None, out_dir, args, merged))
                continue

            # B-2) multi-config
            base_dir = os.path.join(root_out, ds.split("/")[-1])
            os.makedirs(base_dir, exist_ok=True)
            for cfg_name in cfgs:
                out_dir = os.path.join(base_dir, cfg_name)
                if done_flag(out_dir):
                    merged.extend(load_cached_examples(out_dir, f"{ds}/{cfg_name}"))
                    continue
                futs.append(pool.submit(handle_dataset, ds, cfg_name, out_dir, args, merged))

        for f in futs:
            f.result()

    # ------------------------------------------------------------------
    # Shuffle & stratified 16-shot eval split per category
    # ------------------------------------------------------------------
    random.shuffle(merged)
    cat_map = defaultdict(list)
    for ex in merged:
        cat = ex.get("category") or canon_category(ex["dataset_source"])
        ex["category"] = cat
        cat_map[cat].append(ex)

    train, eval_ = [], []
    for li in cat_map.values():
        eval_.extend(li[:16])
        train.extend(li[16:])

    json.dump([pick(x) for x in train], open(os.path.join(root_out, "train_dataset.json"), "w"), indent=2)
    json.dump([pick(x) for x in eval_], open(os.path.join(root_out, "test_datasets.json"), "w"), indent=2)

    log("\n" + "-" * 60, "\nCategory counts (train / eval / total):")
    for c in sorted(cat_map):
        log(f"  • {c}: {max(0, len(cat_map[c]) - 16)} / {min(16, len(cat_map[c]))} / {len(cat_map[c])}")
    log("-" * 60)
    log(f"✅  Finished – {len(train)} train + {len(eval_)} test examples written.")


###############################################################################
# Cache helpers
###############################################################################


def cached_example_cnt(dir_: str) -> int | None:
    fp = os.path.join(dir_, "formatted_data.json")
    if not os.path.isfile(fp):
        return None
    try:
        data = json.load(open(fp))
        if isinstance(data, dict):
            return sum(len(v) for v in data.values() if isinstance(v, list))
    except Exception:
        pass
    return None


def load_cached_examples(dir_: str, src_hint: str | None = None) -> List[Dict[str, Any]]:
    """
    Load examples already on disk.  If *src_hint* is provided we **force-overwrite**
    each record’s ``dataset_source`` with that path and delete any stale
    ``category`` so it can be recomputed cleanly.
    """
    fp = os.path.join(dir_, "formatted_data.json")
    if not os.path.isfile(fp):
        return []

    try:
        data = json.load(open(fp))
        exs: List[Dict[str, Any]] = []
        for v in data.values():
            if isinstance(v, list):
                exs.extend(v)

        if src_hint is not None:
            for ex in exs:
                ex["dataset_source"] = src_hint
                if "category" in ex:
                    del ex["category"]

        return exs
    except Exception as e:
        log(f"⚠️  Could not load cached examples from {fp}: {e}")
        return []


if __name__ == "__main__":
    main()
