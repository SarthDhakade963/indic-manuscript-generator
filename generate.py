"""Synthetic Indic manuscript dataset generator (GEN_V1).
Usage: python generate.py [--config config.json] [--count N] [--out DIR] [--workers K] [--scripts devanagari modi sharada] [--seed S]"""
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.background_generator import generate_manuscript_background
from src.layout import compose, ink_page, outside_fraction
from src.deform import deform

DEFAULTS = {
    "scripts": ["devanagari", "modi", "sharada"],
    "count_per_script": 100,
    "split": {"train": 0.85, "validation": 0.10, "test": 0.05},
    "seed": 1234,
    "output_dir": "output/dataset",
    "workers": 1,
    "font_size": 38,
    "deform_strength": 0.75,
    "max_attempts": 12,
    "materials": {"paper": 0.45, "parchment": 0.15, "palmleaf": 0.40},
    "aging": {"light": 0.3, "moderate": 0.4, "heavy": 0.3},
    "ink_level": {"light": [0.1, 0.3], "moderate": [0.4, 0.6], "heavy": [0.7, 0.9]},
    "sizes": {"page": [1400, 1900], "palmleaf": [2000, 500]},
    "fonts": {
        "devanagari": ["assets/fonts/devanagiri/NIRMALA.TTC", 0],
        "modi": ["assets/fonts/modi/NotoSansModi-Regular.ttf", 0],
        "sharada": ["assets/fonts/sharada/NotoSansSharada-Regular.ttf", 0],
    },
    "backdrops": [[40, 38, 36], [70, 66, 60], [20, 20, 22], [110, 100, 88]],
}


def _probs(d):
    p = np.array(list(d.values()), dtype=np.float64)
    return p / p.sum()


def _backdrop(h, w, base, rng):
    g = rng.standard_normal((h // 80 + 3, w // 80 + 3)).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    n /= n.std() + 1e-6
    return np.clip(np.array(base, np.float32)[None, None, :] * (1.0 + 0.06 * n[..., None]), 0, 255)


def make_page(job):
    cfg, script, si, idx = job["cfg"], job["script"], job["si"], job["idx"]
    prng = np.random.default_rng([cfg["seed"], si, idx])
    mats = list(cfg["materials"])
    mat = str(prng.choice(mats, p=_probs(cfg["materials"])))
    ages = list(cfg["aging"])
    aging = str(prng.choice(ages, p=_probs(cfg["aging"])))
    lo, hi = cfg["ink_level"][aging]
    level = float(prng.uniform(lo, hi))
    W, H = cfg["sizes"]["palmleaf" if mat == "palmleaf" else "page"]
    fp, fi = cfg["fonts"][script]
    last = "none"
    for attempt in range(cfg["max_attempts"]):
        seed = int(prng.integers(1, 2 ** 31 - 1))
        try:
            img, alpha, meta, maps = generate_manuscript_background(W, H, mat, aging, seed, return_material_maps=True)
            page, al = np.asarray(img.convert("RGB")), np.asarray(alpha.convert("L"))
            comp = compose(script, al, fp, fi, W, H, seed, cfg["font_size"],
                           mode="palm" if mat == "palmleaf" else None)
            if comp["overflow"]:
                last = "overflow"
                continue
            res = ink_page(page, al, comp, maps, mat, level, seed)
            r2, a2, (b2, rd2), info = deform(res, al, np.random.default_rng(seed + 99), mat,
                                             masks=[comp["black"], comp["red"]],
                                             strength=cfg["deform_strength"])
            of = outside_fraction(np.maximum(b2, rd2), a2)
            if of > 0.0:
                last = f"outside {of:.5f}"
                continue
        except RuntimeError as e:
            last = str(e)
            continue
        break
    else:
        raise RuntimeError(f"{script} #{idx + 1}: no valid page after {cfg['max_attempts']} attempts ({last})")

    if comp["mode"] == "palm":
        md = comp["md"]
    else:
        rows = [L["text"] for L in comp["layout"]["lines"]]
        rows += [t for t in (comp["side_note"], comp["folio"]) if t]
        md = "\n".join(rows) + "\n"

    bg = _backdrop(H, W, cfg["backdrops"][int(prng.integers(len(cfg["backdrops"])))], prng)
    a = (a2.astype(np.float32) / 255.0)[..., None]
    final = np.clip(r2.astype(np.float32) * a + bg * (1.0 - a), 0, 255).astype(np.uint8)

    name = f"Image_{idx + 1}"
    sdir = Path(job["stage"]) / script
    sdir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(sdir / f"{name}.png")
    (sdir / f"{name}.md").write_text(md, encoding="utf-8")
    return {"name": name, "text": md.rstrip("\n"), "script": script, "material": mat, "aging": aging,
            "ink_level": round(level, 3), "mode": comp["mode"], "font": Path(fp).name, "seed": seed,
            "attempts": attempt + 1, "rubricated_words": comp["highlighted"],
            "side_note": comp["side_note"], "folio": comp["folio"], "curl": info["curl"],
            "fold": info["fold"], "width": W, "height": H}


def contact_sheet(paths, dest, cols=6, tw=230, th=300):
    paths = paths[::max(1, len(paths) // 24)][:24]
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 6) + 6, rows * (th + 6) + 6), (30, 30, 30))
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((tw, th), Image.LANCZOS)
        sheet.paste(im, (6 + (k % cols) * (tw + 6) + (tw - im.width) // 2,
                         6 + (k // cols) * (th + 6) + (th - im.height) // 2))
    sheet.save(dest)


def split_counts(n, split):
    nv, nt = int(round(n * split["validation"])), int(round(n * split["test"]))
    return {"train": n - nv - nt, "validation": nv, "test": nt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--count", type=int)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--scripts", nargs="+")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = json.loads(json.dumps(DEFAULTS))
    if Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key, val in (("count_per_script", args.count), ("output_dir", args.out), ("workers", args.workers),
                     ("scripts", args.scripts), ("seed", args.seed)):
        if val is not None:
            cfg[key] = val

    out = Path(cfg["output_dir"])
    stage = out / "_staging"
    if out.exists():
        shutil.rmtree(out)
    stage.mkdir(parents=True)
    n = cfg["count_per_script"]
    t0 = time.time()
    summary = {}

    for si, script in enumerate(cfg["scripts"]):
        jobs = [{"cfg": cfg, "script": script, "si": si, "idx": i, "stage": str(stage)} for i in range(n)]
        rows = []
        if cfg["workers"] > 1:
            with ProcessPoolExecutor(cfg["workers"]) as ex:
                for k, r in enumerate(ex.map(make_page, jobs)):
                    rows.append(r)
                    print(f"[{script}] {k + 1}/{n} {r['material']} {r['aging']} {r['mode']} ({time.time() - t0:.0f}s)", flush=True)
        else:
            for k, job in enumerate(jobs):
                r = make_page(job)
                rows.append(r)
                print(f"[{script}] {k + 1}/{n} {r['material']} {r['aging']} {r['mode']} ({time.time() - t0:.0f}s)", flush=True)

        srng = np.random.default_rng([cfg["seed"], si, 777])
        perm = [int(i) for i in srng.permutation(n)]
        counts = split_counts(n, cfg["split"])
        start = 0
        summary[script] = {}
        for split, c in counts.items():
            chosen = sorted(perm[start:start + c])
            start += c
            if not chosen:
                continue
            d = out / script / split
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "metadata.jsonl", "w", encoding="utf-8") as f:
                for i in chosen:
                    r = dict(rows[i])
                    name = r.pop("name")
                    for ext in (".png", ".md"):
                        shutil.move(str(stage / script / f"{name}{ext}"), str(d / f"{name}{ext}"))
                    f.write(json.dumps({"file_name": f"{name}.png", **r}, ensure_ascii=False) + "\n")
            summary[script][split] = len(chosen)
        contact_sheet(sorted((out / script).glob("*/*.png")), out.parent / f"{out.name}_contact_{script}.png")

    shutil.rmtree(stage)
    cards = "\n".join(
        f"- config_name: {s}\n  data_files:\n" + "\n".join(
            f"  - split: {sp}\n    path: {s}/{sp}/**" for sp in summary[s]) for s in cfg["scripts"])
    (out / "README.md").write_text(
        f"---\nconfigs:\n{cards}\n---\n\n# Synthetic Indic Manuscripts\n\n"
        "Synthetic folios (Devanagari, Modi, Sharada) with line-level text labels. "
        "Modi and Sharada text is transliterated from Devanagari source text by Unicode block mapping "
        "and has not been verified by a native reader.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()