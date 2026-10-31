"""Throwaway: real text -> ink on all aging levels. Run:  python -m comparing_text"""
import os
import numpy as np
from PIL import Image

import src.ink_engine as ink_engine
from src.background_generator import generate_manuscript_background
from src.ink_engine import render_ink
from src.text_renderer import render_text_mask, render_ruling, handwrite

W, H = 1400, 1900
SEED = 1234
OUT = os.path.join("output", "text_test")
FONT = r"C:\Windows\Fonts\Nirmala.ttc"     # swap for a font in assets/fonts/devanagari to compare
FONT_SIZE = 38
LEVELS = {"light": 0.15, "moderate": 0.5, "heavy": 0.9}
MATERIALS = ["paper", "parchment"]
CROP = (140, 200, 840, 600)

TEXT = [
    "श्री गणेशाय नमः।",
    "आज सकाळपासूनच आभाळ भरून आले होते आणि दुपारपर्यंत पावसाने चांगलाच जोर धरला. गावाबाहेरील ओढ्याला पाणी आले असून शेतकरी पेरणीच्या तयारीत गुंतले आहेत.",
    "बाजारात धान्याचे भाव स्थिर आहेत, परंतु कापसाची आवक अद्याप कमीच आहे. व्यापाऱ्यांनी पुढील आठवड्यात नवी खरेदी सुरू करण्याचे ठरवले आहे.",
    "सायंकाळी सर्वजण मंदिरात जमले आणि देवाला नैवेद्य अर्पण करून आरती म्हटली. लहान मुलांनी दिवे लावले आणि वडीलधाऱ्यांनी आशीर्वाद दिले.",
]


def main():
    print("ink_engine version:", getattr(ink_engine, "VERSION", "?"))
    print("font:", FONT, "size", FONT_SIZE)
    os.makedirs(OUT, exist_ok=True)

    text_mask, layout = render_text_mask(W, H, TEXT, FONT, font_size=FONT_SIZE, seed=SEED)
    text_mask = handwrite(text_mask, np.random.default_rng(SEED + 1))  
    ruling = render_ruling(W, H, layout, text_mask)
    stroke_mask = np.maximum(text_mask, ruling)
    print("text lines:", len(layout["lines"]))

    # Raw mask, black on white: use this to check glyph shaping before any ink.
    prev = Image.fromarray((255 - np.clip(text_mask, 0, 1) * 255).astype(np.uint8))
    prev.save(os.path.join(OUT, "mask_preview.png"))
    prev.crop(CROP).save(os.path.join(OUT, "mask_crop.png"))

    crops = {}
    for mat in MATERIALS:
        for level, lv in LEVELS.items():
            img, alpha, meta, maps = generate_manuscript_background(
                width=W, height=H, material=mat, aging=level, seed=SEED,
                damage=True, return_material_maps=True)
            page = np.asarray(img.convert("RGB"))
            a = np.asarray(alpha.convert("L"))
            rng = np.random.default_rng(SEED + 7)
            out = render_ink(page, a, stroke_mask, maps["absorbency"], maps["roughness"],
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
    sheet.save(os.path.join(OUT, "text_crops.png"))
    print("saved", os.path.join(OUT, "text_crops.png"))


if __name__ == "__main__":
    main()