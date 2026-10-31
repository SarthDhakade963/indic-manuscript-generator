"""Page composer: safe area from the torn edge, main block, marginal note, folio number,
rubricated (red) words, and the matching .md label."""
import numpy as np
import cv2

import src.ink_engine as ink_engine
from src.ink_engine import render_ink
from src.text_renderer import render_text_mask, render_ruling, handwrite
from src.corpus import get_paragraphs
from src.translit import convert

ink_engine.INKS.setdefault("vermilion", {
    "fresh": (150, 28, 22), "aged": (118, 52, 34), "brown_amount": 0.6,
    "max_opacity": 0.92, "bleed": 1.0, "fade": 1.0})

MARKERS = set("।॥") | {"\U00011641", "\U00011642", "\U000111C5", "\U000111C6"}
NOTES = ["श्री", "पाठांतर", "शुद्ध पाठ", "राम राम", "ॐ नमः शिवाय", "हे पुन्हा पाहावे", "इति", "टीका"]


def _alpha01(alpha):
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    return a / 255.0 if a.max() > 1.5 else a


def safe_area(alpha, erode=28, band=150):
    """Distances (left, right, top, bottom) from the image edge to the worst point of the torn edge."""
    inside = cv2.erode((_alpha01(alpha) > 0.5).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1)))
    h, w = inside.shape
    rows = inside[band:h - band]
    has = rows.any(axis=1)
    first = np.where(has, rows.argmax(axis=1), w)
    last = np.where(has, w - 1 - rows[:, ::-1].argmax(axis=1), 0)
    cols = inside[:, band:w - band]
    hasc = cols.any(axis=0)
    firstc = np.where(hasc, cols.argmax(axis=0), h)
    lastc = np.where(hasc, h - 1 - cols[::-1].argmax(axis=0), 0)
    sl, sr = int(first.max()), int(w - 1 - last.min())
    st, sb = int(firstc.max()), int(h - 1 - lastc.min())
    return min(sl, w // 4), min(sr, w // 4), min(st, h // 4), min(sb, h // 4)


def outside_fraction(ink, alpha, erode=6):
    """Share of ink pixels that fall outside the page. Must be 0.0."""
    inside = cv2.erode((_alpha01(alpha) > 0.5).astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8))
    px = ink > 0.15
    return float((px & (inside == 0)).sum()) / max(1, int(px.sum()))


def _crop(m, pad=4):
    ys, xs = np.where(m > 0.05)
    if len(ys) == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
    return m[y0:y1, x0:x1]


def _stamp(canvas, patch, x, y):
    ph, pw = patch.shape
    x = int(np.clip(x, 0, canvas.shape[1] - pw))
    y = int(np.clip(y, 0, canvas.shape[0] - ph))
    reg = canvas[y:y + ph, x:x + pw]
    np.maximum(reg, patch, out=reg)


def _small(text, font_path, font_index, size, seed, w, h):
    m, _ = render_text_mask(w, h, [text], font_path, font_size=size, margins=(10, 10, 16, 10),
                            fill_page=False, seed=seed, font_index=font_index)
    m = handwrite(m, np.random.default_rng(seed + 1), warp=0.8, fine=0.4)
    return _crop(m)


def _rule_baseline(width, height, layout, value=0.5):
    """Ruling under each baseline (for Modi, which has no headline)."""
    left, right, top, bottom = layout["margins"]
    arr = np.zeros((height, width), np.float32)
    xl, xr = int(left - 30), int(width - right + 30)
    ys = [int(L["baseline"] + 8) for L in layout["lines"]]
    for y in ys:
        cv2.line(arr, (xl, y), (xr, y), value, 1, cv2.LINE_AA)
    if ys:
        cv2.line(arr, (xl, ys[0] - int(layout["ascent"]) - 10),
                 (xl, ys[-1] + int(layout["descent"]) + 10), value, 1, cv2.LINE_AA)
    return arr


def _is_marker(tok):
    return len(tok) >= 3 and tok[0] in MARKERS and tok[-1] in MARKERS


def _rubricate(mask, layout, rng, p_marker=0.7, extra=(3, 7)):
    """Move chosen words (verse markers + a few random words) from the black mask to a red mask."""
    h, w = mask.shape
    asc, desc = layout["ascent"], layout["descent"]
    words = [wd for L in layout["lines"] for wd in L["words"]]
    chosen = [i for i, wd in enumerate(words) if _is_marker(wd["text"]) and rng.random() < p_marker]
    plain = [i for i, wd in enumerate(words) if len(wd["text"]) >= 4 and not _is_marker(wd["text"])]
    k = min(len(plain), int(rng.integers(extra[0], extra[1] + 1)))
    if k:
        chosen += [int(i) for i in rng.choice(plain, size=k, replace=False)]
    red = np.zeros_like(mask)
    black = mask.copy()
    names = []
    for i in chosen:
        wd = words[i]
        x0, x1 = int(max(0, wd["x0"] - 3)), int(min(w, wd["x1"] + 3))
        y0, y1 = int(max(0, wd["y"] - asc - 8)), int(min(h, wd["y"] + desc + 8))
        red[y0:y1, x0:x1] = np.maximum(red[y0:y1, x0:x1], mask[y0:y1, x0:x1])
        black[y0:y1, x0:x1] = 0
        names.append(wd["text"])
    return red, black, names


def _md(layout, note, folio):
    out, prev = [], None
    for L in layout["lines"]:
        if prev is not None and L["para"] != prev:
            out.append("")
        out.append(L["text"])
        prev = L["para"]
    out += ["", note, folio]
    return "\n".join(out) + "\n"


def compose(script, alpha, font_path, font_index=0, width=1400, height=1900, seed=0, font_size=38):
    rng = np.random.default_rng(seed)
    sl, sr, st, sb = safe_area(alpha)
    left, right = max(sl + 120, 190), max(sr + 50, 110)
    top, bottom = max(st + 110, 190), max(sb + 60, 170)
    paras, _ = get_paragraphs(script, rng, n_blocks=8)
    mask, layout = render_text_mask(width, height, paras, font_path, font_size=font_size,
                                    margins=(left, right, top, bottom), fill_page=False,
                                    seed=seed, font_index=font_index)
    if "words" not in layout["lines"][0]:
        raise RuntimeError("LAY_V1 is missing from src/text_renderer.py")
    mask = handwrite(mask, np.random.default_rng(seed + 1))

    rule = (_rule_baseline(width, height, layout) if script == "modi"
            else render_ruling(width, height, layout, mask))
    red, mask, highlighted = _rubricate(mask, layout, rng)

    note = str(rng.choice(NOTES))
    note = convert(note, script)
    patch = _small(note, font_path, font_index, 26, seed + 3, 520, 90)
    if patch is not None:
        patch = np.rot90(patch)
        ph = patch.shape[0]
        y = top + rng.uniform(0, max(1.0, (height - bottom - top - ph) * 0.5))
        _stamp(mask, patch, sl + 18, y)

    num = "".join(chr(0x0966 + int(d)) for d in str(int(rng.integers(1, 300))))
    folio = convert("।" + num + "।", script)
    fp = _small(folio, font_path, font_index, 30, seed + 5, 260, 100)
    if fp is not None:
        _stamp(mask, fp, width - right - fp.shape[1] - 10, st + 40)

    return {"black": np.maximum(mask, rule).astype(np.float32), "red": red.astype(np.float32),
            "layout": layout, "md": _md(layout, note, folio), "highlighted": highlighted,
            "side_note": note, "folio": folio, "margins": (left, right, top, bottom),
            "safe": (sl, sr, st, sb)}


def ink_page(page_rgb, alpha, comp, maps, material, aging_level, seed):
    out = render_ink(page_rgb, alpha, comp["black"], maps["absorbency"], maps["roughness"],
                     np.random.default_rng(seed + 7), ink="iron_gall", fiber_map=maps["fiber_map"],
                     aging_level=aging_level, material=material)
    if comp["red"].max() > 0.02:
        out = render_ink(out, alpha, comp["red"], maps["absorbency"], maps["roughness"],
                         np.random.default_rng(seed + 8), ink="vermilion", fiber_map=maps["fiber_map"],
                         aging_level=aging_level, material=material)
    return out