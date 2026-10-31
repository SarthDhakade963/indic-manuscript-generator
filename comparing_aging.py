"""
compare_aging.py
Generate light / moderate / heavy aging with the SAME seed, with damage OFF,
so we isolate the effect of aging strength. Prints stats and saves a
side-by-side contact sheet.
"""
import sys
import cv2
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.background_generator import generate_manuscript_background

SEED = 42
W, H = 1200, 1600
OUT = ROOT / "output" / "compare"
OUT.mkdir(parents=True, exist_ok=True)

LEVELS = ["light", "moderate", "heavy"]
MATERIAL = sys.argv[1] if len(sys.argv) > 1 else "paper"

def stats(img: Image.Image) -> dict:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    # Large-scale variation: blur-free proxy by block-averaging 40x40 tiles
    h, w = lum.shape
    bh, bw = h // 40, w // 40
    tiles = lum[: bh * 40, : bw * 40].reshape(40, bh, 40, bw).mean(axis=(1, 3))
    # Fine-scale variation: pixel std minus nothing, just raw std
    return {
        "mean_rgb": tuple(int(round(v)) for v in a.reshape(-1, 3).mean(0)),
        "mean_lum": float(lum.mean()),
        "lum_std_total": float(lum.std()),
        "lum_std_broad": float(tiles.std()),   # blotchiness / stains
        "lum_min": float(np.percentile(lum, 1)),
        "lum_max": float(np.percentile(lum, 99)),
        "b_over_r": float(a[..., 2].mean() / a[..., 0].mean()),  # lower = yellower
        "fine_std": float((lum - cv2.GaussianBlur(lum, (0, 0), 3)).std()),
    }


def main():
    images, rows = [], []
    for level in LEVELS:
        img, mask, meta = generate_manuscript_background(
            width=W, height=H, material=MATERIAL,
            aging=level, seed=SEED, damage=False,
        )
        img = img.convert("RGB")
        img.save(OUT / f"{MATERIAL}_{level}_seed{SEED}.png")
        images.append(img)
        rows.append((level, stats(img)))

    # Print table
    print(f"\nSame seed = {SEED}, damage=False, material={MATERIAL}\n")
    print(f"{'level':10} {'mean RGB':>18} {'lum':>7} {'std':>6} {'broad':>6} "
          f"{'p1':>6} {'p99':>6} {'B/R':>6}")
    for level, s in rows:
        print(f"{level:10} {str(s['mean_rgb']):>18} {s['mean_lum']:7.1f} "
              f"{s['lum_std_total']:6.2f} {s['lum_std_broad']:6.2f} "
              f"{s['lum_min']:6.1f} {s['lum_max']:6.1f} {s['b_over_r']:6.3f}")

    # Contact sheet (each page scaled to 50%)
    tw, th = W // 2, H // 2
    sheet = Image.new("RGB", (tw * 3 + 40, th + 60), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for i, (img, level) in enumerate(zip(images, LEVELS)):
        x = 10 + i * (tw + 10)
        sheet.paste(img.resize((tw, th), Image.LANCZOS), (x, 40))
        d.text((x, 12), level.upper(), fill=(255, 255, 255))
    sheet.save(OUT / f"contact_sheet_{MATERIAL}_seed{SEED}.png")
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()