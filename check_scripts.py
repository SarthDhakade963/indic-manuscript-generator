import numpy as np
from PIL import Image
from src.text_renderer import render_text_mask, handwrite
from src.corpus import get_paragraphs

W, H, SEED = 1400, 1900, 1234
FONTS = {
    "devanagari": (r"assets\fonts\devanagiri\NIRMALA.TTC", 0),
    "modi": (r"assets\fonts\modi\NotoSansModi-Regular.ttf", 0),
    "sharada": (r"assets\fonts\sharada\NotoSansSharada-Regular.ttf", 0),
}

for name, (path, idx) in FONTS.items():
    rng = np.random.default_rng(SEED)
    paras, _ = get_paragraphs(name, rng)
    mask, layout = render_text_mask(W, H, paras, path, font_size=38, seed=SEED, font_index=idx)
    mask = handwrite(mask, np.random.default_rng(SEED + 1))
    img = (255 * (1 - np.clip(np.asarray(mask, dtype=np.float32), 0, 1))).astype(np.uint8)
    Image.fromarray(img).save(rf"output\text_test\script_{name}.png")
    print("saved", name)