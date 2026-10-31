import os
import numpy as np
from PIL import Image
from src.background_generator import generate_manuscript_background
from src.layout import compose, ink_page, outside_fraction

W, H = 1400, 1900
OUT = os.path.join("output", "layout_test")
os.makedirs(OUT, exist_ok=True)
CASES = [
    ("devanagari", r"assets\fonts\devanagiri\NIRMALA.TTC", "paper", "moderate", 0.5),
    ("modi", r"assets\fonts\modi\NotoSansModi-Regular.ttf", "parchment", "moderate", 0.5),
    ("sharada", r"assets\fonts\sharada\NotoSansSharada-Regular.ttf", "paper", "heavy", 0.9),
]
for i, (script, font, material, aging, level) in enumerate(CASES):
    seed = 1000 + i
    img, alpha, meta, maps = generate_manuscript_background(
        width=W, height=H, material=material, aging=aging, seed=seed,
        damage=True, return_material_maps=True)
    page = np.asarray(img.convert("RGB"))
    a = np.asarray(alpha.convert("L"))
    comp = compose(script, a, font, 0, W, H, seed)
    out = ink_page(page, a, comp, maps, material, level, seed)
    Image.fromarray(out).save(os.path.join(OUT, f"{script}.png"))
    with open(os.path.join(OUT, f"{script}.md"), "w", encoding="utf-8") as f:
        f.write(comp["md"])
    outside = outside_fraction(np.maximum(comp["black"], comp["red"]), a)
    print(script, "| lines", len(comp["layout"]["lines"]), "| red words", len(comp["highlighted"]),
          "| outside", round(outside, 5), "| margins", comp["margins"], "| safe", comp["safe"])