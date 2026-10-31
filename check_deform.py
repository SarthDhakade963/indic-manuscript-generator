import numpy as np
from pathlib import Path
from PIL import Image
from src.background_generator import generate_manuscript_background
from src.layout import compose, ink_page, outside_fraction
from src.deform import deform

FONTS = {"devanagari": (r"assets\fonts\devanagiri\NIRMALA.TTC", 0),
         "modi": (r"assets\fonts\modi\NotoSansModi-Regular.ttf", 0),
         "sharada": (r"assets\fonts\sharada\NotoSansSharada-Regular.ttf", 0)}
LEVEL = {"light": 0.2, "moderate": 0.5, "heavy": 0.8}
out = Path("output") / "deform_test"
out.mkdir(parents=True, exist_ok=True)

cases = [("paper", "devanagari", "moderate", 1400, 1900), ("parchment", "modi", "heavy", 1400, 1900),
         ("paper", "sharada", "light", 1400, 1900), ("palmleaf", "devanagari", "moderate", 2000, 500),
         ("palmleaf", "modi", "heavy", 2000, 500)]
for i, (mat, script, aging, W, H) in enumerate(cases):
    fp, fi = FONTS[script]
    seed = 5000 + i
    img, alpha, meta, maps = generate_manuscript_background(W, H, mat, aging, seed, return_material_maps=True)
    page, al = np.asarray(img.convert("RGB")), np.asarray(alpha.convert("L"))
    comp = compose(script, al, fp, fi, W, H, seed, 38, mode="palm" if mat == "palmleaf" else None)
    res = ink_page(page, al, comp, maps, mat, LEVEL[aging], seed)
    r2, a2, (b2,), info = deform(res, al, np.random.default_rng(seed + 99), mat, masks=[comp["black"]])
    of = outside_fraction(b2, a2)
    name = f"{mat}_{script}_{aging}"
    grey = np.full_like(r2, 90)
    Image.fromarray(np.where(a2[..., None] > 127, r2, grey)).save(out / f"{name}.png")
    print(f"{name}: outside {of:.5f} overflow {comp['overflow']} curl {info['curl']} fold {info['fold']}")