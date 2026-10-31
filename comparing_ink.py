"""Throwaway ink test. Run from the project root:  python -m comparing_ink"""
import os
import numpy as np
from PIL import Image, ImageDraw

import src.ink_engine as ink_engine
from src.background_generator import generate_manuscript_background
from src.ink_engine import render_ink

W, H = 1400, 1900
SS = 3
SEED = 1234
OUT = os.path.join("output", "ink_test")
LEVELS = {"light": 0.15, "moderate": 0.5, "heavy": 0.9}
MATERIALS = ["paper", "parchment"]
CROP = (140, 200, 840, 600)


def bezier(p0, p1, p2, n=20):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.array, (p0, p1, p2))
    return [tuple(p) for p in ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2)]


def stroke(draw, pts, w):
    n = len(pts)
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        wi = max(1.0, w * (0.6 + 0.8 * np.sqrt(np.sin(np.pi * t) + 1e-3)))
        draw.line([pts[i], pts[i + 1]], fill=255, width=int(round(wi)))
        r = wi / 2
        x, y = pts[i]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    for (x, y) in (pts[0], pts[-1]):
        r = w * 0.75
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)


def draw_word(draw, x, y, width, rng):
    s = SS
    bar = [(x + i * width / 12, y + rng.normal(0, 0.5) * s) for i in range(13)]
    stroke(draw, bar, 3.0 * s)
    n = max(2, int(width // 34))
    for g in range(n):
        gx = x + (g + 0.5) * width / n
        kind = rng.integers(0, 3)
        if kind == 0:
            p1 = (gx + rng.normal(0, 2) * s, y + rng.uniform(34, 52) * s)
            stroke(draw, bezier((gx, y), ((gx + p1[0]) / 2 + 2 * s, y + 20 * s), p1), 3.0 * s)
        elif kind == 1:
            stroke(draw, bezier((gx - 9 * s, y + 6 * s), (gx - 16 * s, y + 52 * s),
                                (gx + 10 * s, y + 24 * s)), 3.2 * s)
        else:
            stroke(draw, bezier((gx, y + 4 * s), (gx + 14 * s, y + 30 * s),
                                (gx - 8 * s, y + 46 * s)), 2.8 * s)


def make_stroke_mask(seed):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(img)
    for i in range(13):
        y = (300 + i * 120) * SS
        d.line([(120 * SS, y), (1280 * SS, y)], fill=140, width=int(1.6 * SS))
    d.line([(120 * SS, 200 * SS), (120 * SS, 1760 * SS)], fill=140, width=int(1.6 * SS))
    for i in range(13):
        y = (240 + i * 120) * SS
        x = 160
        while x < 1180:
            wd = int(rng.integers(60, 190))
            if x + wd > 1250:
                break
            draw_word(d, x * SS, y, wd * SS, rng)
            x += wd + 24
    img = img.resize((W, H), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    print("ink_engine file:", ink_engine.__file__)
    os.makedirs(OUT, exist_ok=True)
    mask = make_stroke_mask(SEED)
    crops = {}
    for mat in MATERIALS:
        for level, lv in LEVELS.items():
            img, alpha, meta, maps = generate_manuscript_background(
                width=W, height=H, material=mat, aging=level, seed=SEED,
                damage=True, return_material_maps=True)
            page = np.asarray(img.convert("RGB"))
            a = np.asarray(alpha.convert("L"))
            rng = np.random.default_rng(SEED + 7)
            out = render_ink(page, a, mask, maps["absorbency"], maps["roughness"],
                             rng, ink="iron_gall", fiber_map=maps["fiber_map"],
                             aging_level=lv, material=mat)
            im = Image.fromarray(out)
            im.save(os.path.join(OUT, f"{mat}_{level}.png"))
            crops[(mat, level)] = im.crop(CROP)
            print("done", mat, level)

    cw, ch = CROP[2] - CROP[0], CROP[3] - CROP[1]
    sheet = Image.new("RGB", (cw * 2, ch * 3), (255, 255, 255))
    for r, level in enumerate(LEVELS):
        for c, mat in enumerate(MATERIALS):
            sheet.paste(crops[(mat, level)], (c * cw, r * ch))
    sheet.save(os.path.join(OUT, "ink_crops.png"))
    print("saved", os.path.join(OUT, "ink_crops.png"))


if __name__ == "__main__":
    main()