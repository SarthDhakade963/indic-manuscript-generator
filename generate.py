"""
generate.py
───────────
Main entry point for the Synthetic Manuscript Generator.

Usage:
    python generate.py                          # use config.yaml defaults
    python generate.py --count 10              # quick test run
    python generate.py --scripts devanagari    # single script
    python generate.py --upload                # upload to Hugging Face after generation

What it produces (per script, inside output_dir/):
    {script}/train/Image_{N:03d}.png  + Image_{N:03d}.md
    {script}/validation/...
    {script}/test/...
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.background_generator import generate_manuscript_background
from src.layout import compose, ink_page, outside_fraction


# Helpers ────────────

def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def aging_to_level(aging: str) -> float:
    """Converts an aging profile name to a 0-1 float the ink engine understands.
    The ink engine uses this to decide how browned and faded the ink should look."""
    return {"light": 0.15, "moderate": 0.5, "heavy": 0.9}.get(aging, 0.5)


def split_name(idx: int, splits: dict) -> str:
    """Returns which split folder this image belongs in based on its index.
    First 85 go to train, next 10 to validation, last 5 to test."""
    train_end = splits["train"]
    val_end   = train_end + splits["validation"]
    if idx < train_end:
        return "train"
    if idx < val_end:
        return "validation"
    return "test"


def generate_one(
    script:      str,
    font_path:   str,
    font_index:  int,
    font_size:   int,
    width:       int,
    height:      int,
    material:    str,
    aging:       str,
    seed:        int,
    damage:      bool,
) -> tuple:
    """Generate one manuscript image.

    Returns
    -------
    image : PIL.Image (RGB)
    md    : str  – ground-truth annotation text
    meta  : dict – generation metadata
    """
    # 1. Background (paper texture + aging + damage)
    img, alpha, meta, maps = generate_manuscript_background(
        width=width,
        height=height,
        material=material,
        aging=aging,
        seed=seed,
        damage=damage,
        return_material_maps=True,
    )

    page_rgb = np.asarray(img.convert("RGB"))
    alpha_arr = np.asarray(alpha.convert("L"))

    # 2. Page layout (text mask + margin note + folio + rubrication)
    comp = compose(
        script=script,
        alpha=alpha_arr,
        font_path=font_path,
        font_index=font_index,
        width=width,
        height=height,
        seed=seed,
        font_size=font_size,
    )

    # 3. Ink compositing (iron-gall + vermilion highlight)
    out_rgb = ink_page(
        page_rgb=page_rgb,
        alpha=alpha_arr,
        comp=comp,
        maps=maps,
        material=material,
        aging_level=aging_to_level(aging),
        seed=seed,
    )

    # Safety check – no ink should spill outside the torn boundary
    overflow = outside_fraction(
        np.maximum(comp["black"], comp["red"]), alpha_arr
    )

    meta.update({
        "script":    script,
        "aging":     aging,
        "seed":      seed,
        "overflow":  round(overflow, 6),
        "lines":     len(comp["layout"]["lines"]),
        "red_words": len(comp["highlighted"]),
    })

    return Image.fromarray(out_rgb), comp["md"], meta


# Main generation loop ──────────────────────────────────────────────────────

def run(cfg: dict, scripts: list, count: int, upload: bool):
    out_root = Path(cfg["output_dir"])
    splits   = cfg["splits"]
    W, H     = cfg["width"], cfg["height"]
    damage   = cfg.get("damage", True)

    materials     = cfg["materials"]
    aging_levels  = cfg["aging_levels"]

    total      = len(scripts) * count
    done       = 0
    t0         = time.time()
    errors     = []

    print(f"\nGenerating {total} images  ({len(scripts)} scripts x {count})\n")

    for script in scripts:
        fcfg       = cfg["fonts"][script]
        font_path  = fcfg["path"]
        font_index = int(fcfg.get("index", 0))
        font_size  = int(fcfg.get("size", 38))

        for i in range(count):
            split = split_name(i, splits)
            img_id = f"Image_{i:03d}"

            out_dir = out_root / script / split
            out_dir.mkdir(parents=True, exist_ok=True)

            img_path = out_dir / f"{img_id}.png"
            md_path  = out_dir / f"{img_id}.md"

            # Deterministic seed: unique per script + index
            seed = abs(hash(script)) % (2**20) + i * 97

            # Cycle through materials and aging levels
            rng_choice = np.random.default_rng(seed)
            material   = str(rng_choice.choice(materials))
            aging      = str(rng_choice.choice(aging_levels))

            try:
                image, md_text, meta = generate_one(
                    script=script,
                    font_path=font_path,
                    font_index=font_index,
                    font_size=font_size,
                    width=W,
                    height=H,
                    material=material,
                    aging=aging,
                    seed=seed,
                    damage=damage,
                )

                image.save(img_path)
                md_path.write_text(md_text, encoding="utf-8")

                done += 1
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (total - done) / rate if rate > 0 else 0

                print(
                    f"  [{done:>3}/{total}]  {script}/{split}/{img_id}"
                    f"  {material}/{aging}"
                    f"  lines={meta['lines']}"
                    f"  red={meta['red_words']}"
                    f"  overflow={meta['overflow']}"
                    f"  ETA {eta/60:.1f}min"
                )

            except Exception as exc:
                errors.append((script, i, str(exc)))
                print(f"  ERROR  {script}/{img_id}: {exc}", file=sys.stderr)

    # Summary
    elapsed = time.time() - t0
    print(f"\nDone.  {done}/{total} images in {elapsed:.1f}s  ({elapsed/max(done,1):.1f}s each)")
    print(f"Output: {out_root.resolve()}")

    if errors:
        print(f"\nFailed ({len(errors)}):")
        for s, i, e in errors:
            print(f"  {s}/Image_{i:03d}: {e}")

    if upload:
        hf_upload(cfg, out_root)

    return done, errors


# Hugging Face upload 

def hf_upload(cfg: dict, out_root: Path):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    hf_cfg   = cfg.get("huggingface", {})
    repo_id  = hf_cfg.get("repo_id", "")
    if not repo_id or repo_id.startswith("your-username"):
        print("\nSkipping upload: set huggingface.repo_id in config.yaml first.")
        return

    repo_type = hf_cfg.get("repo_type", "dataset")
    private   = bool(hf_cfg.get("private", False))

    api = HfApi()

    print(f"\nUploading to Hugging Face: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type,
                        private=private, exist_ok=True)
    except Exception as e:
        print(f"  Could not create repo: {e}")
        return

    # Upload every file under out_root
    files_uploaded = 0
    for f in sorted(out_root.rglob("*")):
        if not f.is_file():
            continue
        path_in_repo = f.relative_to(out_root).as_posix()
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            files_uploaded += 1
            if files_uploaded % 50 == 0:
                print(f"  Uploaded {files_uploaded} files...")
        except Exception as e:
            print(f"  Failed {path_in_repo}: {e}")

    print(f"  Upload complete: {files_uploaded} files -> {repo_id}")


# CLI 

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Indic manuscript images."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config (default: config.yaml)"
    )
    parser.add_argument(
        "--scripts", nargs="+",
        help="Override scripts to generate (e.g. --scripts devanagari modi)"
    )
    parser.add_argument(
        "--count", type=int,
        help="Override count per script"
    )
    parser.add_argument(
        "--output", type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload to Hugging Face after generation"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.scripts:
        scripts = args.scripts
    else:
        scripts = cfg["scripts"]

    if args.count:
        count = args.count
    else:
        count = cfg["count_per_script"]

    if args.output:
        cfg["output_dir"] = args.output

    run(cfg, scripts, count, upload=args.upload)


if __name__ == "__main__":
    main()

