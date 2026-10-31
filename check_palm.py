import numpy as np
from pathlib import Path
from PIL import Image
from src.background_generator import generate_manuscript_background
from src.layout import compose, ink_page, outside_fraction

FONTS = {"devanagari": (r"assets\fonts\devanagiri\NIRMALA.TTC", 0),
         "modi": (r"assets\fonts\modi\NotoSansModi-Regular.ttf", 0),
         "sharada": (r"assets\fonts\sharada\NotoSansSharada-Regular.ttf", 0)}
LEVEL = {"light": 0.2, "moderate": 0.5, "heavy": 0.8}
W, H = 2000, 500
out = Path("output") / "palm_test"
out.mkdir(parents=True, exist_ok=True)

for i, aging in enumerate(["light", "moderate", "heavy"]):
    for j, (script, (fp, fi)) in enumerate(FONTS.items()):
        seed = 3000 + i * 10 + j
        img, alpha, meta, maps = generate_manuscript_background(
            W, H, "palmleaf", aging, seed, return_material_maps=True)
        page = np.asarray(img.convert("RGB"))
        al = np.asarray(alpha.convert("L"))
        comp = compose(script, al, fp, fi, W, H, seed, 38, mode="palm")
        res = ink_page(page, al, comp, maps, "palmleaf", LEVEL[aging], seed)
        of = outside_fraction(comp["black"], al)
        name = f"palm_{script}_{aging}"
        grey = np.full_like(res, 90)
        a3 = (al[..., None] > 127)
        Image.fromarray(np.where(a3, res, grey)).save(out / f"{name}.png")
        print(f"{name}: outside {of:.5f} overflow {comp['overflow']} lines {len(comp['md'].splitlines())}")