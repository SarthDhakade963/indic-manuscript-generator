import os
import numpy as np
from PIL import Image
from src.background_generator import generate_manuscript_background
from src.layout import compose, ink_page, outside_fraction, MODES

W, H = 1400, 1900
OUT = os.path.join("output", "layout_test")
os.makedirs(OUT, exist_ok=True)
FONTS = {
    "devanagari": r"assets\fonts\devanagiri\NIRMALA.TTC",
    "modi": r"assets\fonts\modi\NotoSansModi-Regular.ttf",
    "sharada": r"assets\fonts\sharada\NotoSansSharada-Regular.ttf",
}
MATS = ["paper", "parchment"]
AGES = [("light", 0.15), ("moderate", 0.5), ("heavy", 0.9)]

thumbs, i = [], 0
for script, font in FONTS.items():
    for mode in MODES:
        seed = 1000 + i
        material = MATS[i % 2]
        aging, level = AGES[i % 3]
        img, alpha, meta, maps = generate_manuscript_background(
            width=W, height=H, material=material, aging=aging, seed=seed,
            damage=True, return_material_maps=True)
        page = np.asarray(img.convert("RGB"))
        a = np.asarray(alpha.convert("L"))
        comp = compose(script, a, font, 0, W, H, seed, mode=mode)
        out = ink_page(page, a, comp, maps, material, level, seed)
        name = f"{script}_{mode}"
        Image.fromarray(out).save(os.path.join(OUT, name + ".png"))
        with open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8") as f:
            f.write(comp["md"])
        outside = outside_fraction(np.maximum(comp["black"], comp["red"]), a)
        print(f"{name:22s} lines {len(comp['layout']['lines']):2d} | red {len(comp['highlighted']):2d} | "
              f"outside {outside:.5f} | overflow {comp['overflow']} | {material}/{aging}")
        thumbs.append(Image.fromarray(out).resize((350, 475), Image.LANCZOS))
        i += 1

sheet = Image.new("RGB", (350 * 4, 475 * 3), (255, 255, 255))
for k, t in enumerate(thumbs):
    sheet.paste(t, ((k % 4) * 350, (k // 4) * 475))
sheet.save(os.path.join(OUT, "contact_sheet.png"))
print("saved contact_sheet.png")