"""Throwaway diagnostic: damage=True at fixed seeds, composited on a neutral
background so the page boundary is visible. Run: python -m src.comparing_damage"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.background_generator import generate_manuscript_background

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "damage"
OUT.mkdir(parents=True, exist_ok=True)

W, H, SEED = 1200, 1600, 42
CASES = [("paper", "moderate"), ("paper", "heavy"), ("parchment", "moderate")]
BG = (70, 90, 110)

tw, th = W // 2, H // 2
sheet = Image.new("RGB", (tw * 3 + 40, th * 2 + 80), (30, 30, 30))
d = ImageDraw.Draw(sheet)

for i, (mat, lvl) in enumerate(CASES):
    img, mask, _ = generate_manuscript_background(
        width=W, height=H, material=mat, aging=lvl, seed=SEED, damage=True
    )
    comp = Image.new("RGB", (W, H), BG)
    comp.paste(img.convert("RGB"), (0, 0), mask)
    comp.save(OUT / f"{mat}_{lvl}_damage_seed{SEED}.png")
    mask.save(OUT / f"{mat}_{lvl}_mask_seed{SEED}.png")

    x = 10 + i * (tw + 10)
    d.text((x, 12), f"{mat} {lvl}", fill=(255, 255, 255))
    sheet.paste(comp.resize((tw, th), Image.LANCZOS), (x, 30))
    sheet.paste(mask.convert("RGB").resize((tw, th), Image.LANCZOS),
                (x, th + 50))

sheet.save(OUT / f"damage_sheet_seed{SEED}.png")
print("Saved to:", OUT)