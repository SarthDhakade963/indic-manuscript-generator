"""Page composer (LAY_V3): safe area from the torn edge, four layout modes, wavy ruling,
sparse rubrication, margin note, folio, yellow wash, and the matching .md label."""
import numpy as np
import cv2

import src.ink_engine as ink_engine
from src.ink_engine import render_ink
from src.text_renderer import render_text_mask, handwrite
from src.corpus import get_paragraphs, HEADERS
from src.translit import convert
from src.damage_engine import PALM_HOLE_FRACS, PALM_CLEAR


ink_engine.INKS.setdefault("vermilion", {
    "fresh": (150, 28, 22), "aged": (118, 52, 34), "brown_amount": 0.6,
    "max_opacity": 0.92, "bleed": 1.0, "fade": 1.0})
ink_engine.INKS.setdefault("lampblack", {  
    "fresh": (28, 22, 16), "aged": (46, 34, 22), "brown_amount": 0.5,
    "max_opacity": 0.93, "bleed": 0.6, "fade": 0.35})
ink_engine.MATERIAL_BLEED.setdefault("palmleaf", 0.5)  

MARKERS = set("।॥") | {"\U00011641", "\U00011642", "\U000111C5", "\U000111C6"}
NOTES = ["श्री", "पाठांतर", "शुद्ध पाठ", "राम राम", "ॐ नमः शिवाय", "हे पुन्हा पाहावे", "इति", "टीका",
         "श्रीगुरवे नमः", "अधिक पाठ", "पाहा पुढे", "शुभं भवतु", "हरि ॐ", "सही", "पान उलटावे", "येथे खंड"]
COLOPHONS = ["इति श्रीकथासारः समाप्तः । शुभं भवतु । श्रीरामार्पणमस्तु ।",
             "इति श्री अध्याय समाप्त । श्रीकृष्णार्पणमस्तु । शके १७४२ ।",
             "संवत् १८८४ मिती कार्तिक शुद्ध ५ । लेखक पाठक । श्री ।",
             "इति श्रीमद्ग्रंथे द्वितीयोऽध्यायः समाप्तः । शुभमस्तु ।",
             "श्री हरिहरार्पणमस्तु । लिखितं रामचंद्रेण । शुभं भवतु ।"]
CLOSERS = ["॥ श्री ॥", "इति समाप्तम् ।", "श्री रामार्पणमस्तु ।", "शुभं भवतु ।", "॥ हरि ॐ ॥"]
MODES = ["single", "two_col", "colophon", "short"]


def _alpha01(alpha):
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    return a / 255.0 if a.max() > 1.5 else a


def safe_area(alpha, erode=20, band=150):
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


def _seg(canvas, p, q, color):
    cv2.line(canvas, (int(p[0] * 16), int(p[1] * 16)), (int(q[0] * 16), int(q[1] * 16)),
             int(color), 1, cv2.LINE_AA, 4)


def _ruling(width, height, layout, mask, script, rng, value=(0.32, 0.58)):
    """Hand-ruled lines: own slope, slow waviness, strength and length per line, occasional gaps."""
    left, right, top, bottom = layout["margins"]
    asc = layout["ascent"]
    canvas = np.zeros((height, width), np.uint8)
    ys = []
    for L in layout["lines"]:
        b = L["baseline"]
        if script == "modi":
            ry = b + 8
        else:
            y0, y1 = int(max(0, b - asc)), int(min(height, b))
            x0, x1 = int(L["x0"]), int(min(width, L["x1"]))
            band = mask[y0:y1, x0:x1].sum(axis=1) if y1 > y0 and x1 > x0 else np.zeros(1)
            ry = y0 + int(np.argmax(band)) - 7
        ys.append(ry)
        xa = left - 30 + rng.uniform(-14, 14)
        xb = width - right + 30 + rng.uniform(-14, 14)
        slope = rng.normal(0, 2.0) / max(1.0, xb - xa)
        lam, ph, amp = rng.uniform(180, 420), rng.uniform(0, 6.28), rng.uniform(0.4, 1.4)
        v = rng.uniform(*value)
        xs = np.arange(xa, xb, 25.0)
        gap = 0
        for i in range(len(xs) - 1):
            if gap > 0:
                gap -= 1
                continue
            if rng.random() < 0.025:
                gap = int(rng.integers(2, 6))
                continue
            def pt(x):
                return (x, ry + slope * (x - xa) + amp * np.sin(2 * np.pi * x / lam + ph))
            vv = v * (0.85 + 0.15 * np.sin(i * 0.7 + ph))
            _seg(canvas, pt(xs[i]), pt(xs[i + 1]), 255 * vv)
    if ys:
        xl = left - 30 + rng.uniform(-3, 3)
        ya, yb = ys[0] - 20, ys[-1] + layout["ascent"] + layout["descent"]
        lam, ph = rng.uniform(250, 500), rng.uniform(0, 6.28)
        yy = np.arange(ya, yb, 25.0)
        for i in range(len(yy) - 1):
            def pv(y):
                return (xl + 1.5 * np.sin(2 * np.pi * y / lam + ph), y)
            _seg(canvas, pv(yy[i]), pv(yy[i + 1]), 255 * 0.45)
    return canvas.astype(np.float32) / 255.0


def _box(shape, box, rng, value=0.5):
    """Double-ruled hand-drawn rectangle."""
    canvas = np.zeros(shape, np.uint8)
    for inset in (0, 5):
        x0, y0, x1, y1 = box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        for a, b in zip(corners[:-1], corners[1:]):
            n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 20))
            pts = [(a[0] + (b[0] - a[0]) * t + rng.normal(0, 0.8),
                    a[1] + (b[1] - a[1]) * t + rng.normal(0, 0.8)) for t in np.linspace(0, 1, n + 1)]
            for p, q in zip(pts[:-1], pts[1:]):
                _seg(canvas, p, q, 255 * value)
    return canvas.astype(np.float32) / 255.0


def _is_marker(tok):
    return len(tok) >= 3 and tok[0] in MARKERS and tok[-1] in MARKERS


def _rubricate(mask, layout, rng, p_marker=0.55, extra=(0, 2)):
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


def _wash(shape, layout, rng, n=(1, 2)):
    """Turmeric-yellow highlight behind 1-2 ordinary words."""
    words = [wd for L in layout["lines"] for wd in L["words"]
             if len(wd["text"]) >= 5 and not _is_marker(wd["text"])]
    m = np.zeros(shape, np.float32)
    k = min(len(words), int(rng.integers(n[0], n[1] + 1)))
    if k:
        for i in rng.choice(len(words), size=k, replace=False):
            wd = words[int(i)]
            x0, x1 = max(0, int(wd["x0"] - 6)), int(wd["x1"] + 6)
            y0, y1 = max(0, int(wd["y"] - layout["ascent"] * 0.9)), int(wd["y"] + layout["descent"] * 0.6)
            m[y0:y1, x0:x1] = 1.0
    return cv2.GaussianBlur(m, (0, 0), 3.0)


def _md(layout, note, folio):
    out, prev = [], None
    for L in layout["lines"]:
        if prev is not None and L["para"] != prev:
            out.append("")
        out.append(L["text"])
        prev = L["para"]
    tail = [t for t in (note, folio) if t]
    if tail:
        out += [""] + tail
    return "\n".join(out) + "\n"


def _block(paras, font_path, font_index, width, height, margins, font_size, seed):
    slack = 28                                                                   
    inner = (margins[0], margins[1] + slack, margins[2], margins[3])             
    mask, layout = render_text_mask(width, height, paras, font_path, font_size=font_size,
                                    margins=inner, fill_page=False, seed=seed, font_index=font_index)
    layout["margins"] = margins                                                  
    if layout["lines"]:
        if "words" not in layout["lines"][0]:
            raise RuntimeError("LAY_V1 is missing from src/text_renderer.py")
        mask = handwrite(mask, np.random.default_rng(seed + 1))
    layout["overflow"] = any(L["x1"] > width - margins[1] + 2 for L in layout["lines"])   
    return mask, layout


def _strip_header(paras, script):
    heads = {convert(h, script) for h in HEADERS}
    return paras[1:] if paras and paras[0] in heads else paras


def _shift(mask, layout, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    for L in layout["lines"]:
        L["x0"] += dx
        L["x1"] += dx
        for wd in L["words"]:
            wd["x0"] += dx
            wd["x1"] += dx
    return mask

def _compose_palm(script, alpha, font_path, font_index, width, height, seed, font_size): 
    rng = np.random.default_rng(seed)
    fs = int(font_size * 0.8)
    top = 62 + int(rng.integers(0, 14))
    bottom = 62 + int(rng.integers(0, 60))
    inside = cv2.erode((_alpha01(alpha) > 0.5).astype(np.uint8), np.ones((31, 31), np.uint8))
    rows = inside[top:height - bottom]
    if not rows.any(axis=1).all():
        raise RuntimeError("palm: page edge intrudes into text rows")
    first = rows.argmax(axis=1)
    last = width - 1 - rows[:, ::-1].argmax(axis=1)
    left = int(first.max()) + 25
    right = int(width - 1 - last.min()) + 25
    hx = [f * width for f in PALM_HOLE_FRACS]
    segs = [(left, hx[0] - PALM_CLEAR), (hx[0] + PALM_CLEAR, hx[1] - PALM_CLEAR),
            (hx[1] + PALM_CLEAR, width - right)]
    blocks = []
    for k, (a, b) in enumerate(segs):
        paras, _ = get_paragraphs(script, rng, 3)
        text = " ".join(_strip_header(paras, script))
        m, lay = _block([text], font_path, font_index, width, height,
                        (int(round(a)), int(round(width - b)), top, bottom), fs, seed + 41 * (k + 1))
        blocks.append((m, lay))
    counts = {len(lay["lines"]) for _, lay in blocks}
    if len(counts) != 1 or 0 in counts:
        raise RuntimeError(f"palm: unequal line counts {counts}")
    n = counts.pop()
    md = "\n".join(" ".join(blocks[k][1]["lines"][i]["text"] for k in range(3)) for i in range(n)) + "\n"
    lines = [blocks[k][1]["lines"][i] for i in range(n) for k in range(3)]
    ref = blocks[0][1]
    layout = {"lines": lines, "ascent": ref["ascent"], "descent": ref["descent"],
              "margins": (left, right, top, bottom), "font_size": fs}
    zero = np.zeros((height, width), np.float32)
    return {"black": np.maximum.reduce([m for m, _ in blocks]).astype(np.float32), "red": zero,
            "wash": zero, "layout": layout, "md": md, "highlighted": [], "side_note": None,
            "folio": None, "mode": "palm", "overflow": any(lay["overflow"] for _, lay in blocks),
            "margins": (left, right, top, bottom), "safe": (left, right, top, bottom)}

def compose(script, alpha, font_path, font_index=0, width=1400, height=1900, seed=0,
            font_size=38, mode=None):
    rng = np.random.default_rng(seed)
    if mode == "palm":
        return _compose_palm(script, alpha, font_path, font_index, width, height, seed, font_size)
    sl, sr, st, sb = safe_area(alpha)
    left, right = max(sl + 120, 190), max(sr + 50, 110)
    top, bottom = max(st + 110, 190), max(sb + 35, 130)
    if mode is None:
        mode = str(rng.choice(MODES, p=[0.35, 0.25, 0.25, 0.15]))
    fs = int(font_size * 0.94) if mode == "two_col" else font_size

    blocks, box = [], None          # blocks: (mask, layout, ruled)
    if mode == "two_col":
        gutter = 56
        colw = (width - left - right - gutter) // 2
        p1, _ = get_paragraphs(script, rng, 7)
        p2, _ = get_paragraphs(script, rng, 7)
        p2 = _strip_header(p2, script)
        m1, l1 = _block(p1, font_path, font_index, width, height,
                        (left, width - left - colw, top, bottom), fs, seed)
        m2, l2 = _block(p2, font_path, font_index, width, height,
                        (left + colw + gutter, right, top, bottom), fs, seed + 11)
        for L in l2["lines"]:
            L["para"] += 1000
        blocks = [(m1, l1, True), (m2, l2, True)]
    elif mode == "colophon":
        paras, _ = get_paragraphs(script, rng, 8)
        m1, l1 = _block(paras, font_path, font_index, width, height,
                        (left, right, top, bottom + 230), fs, seed)
        blocks = [(m1, l1, True)]
        box_top = (l1["lines"][-1]["baseline"] + l1["descent"] + 55) if l1["lines"] else top
        text = convert(str(rng.choice(COLOPHONS)), script)
        mc, lc = _block([text], font_path, font_index, width, height,
                        (left + 40, right + 40, int(box_top + 30), bottom), fs, seed + 21)
        for L in lc["lines"]:
            L["para"] = 2000
        blocks.append((mc, lc, False))
        if lc["lines"]:
            box = (left - 14, box_top, width - right + 14,
                   lc["lines"][-1]["baseline"] + lc["descent"] + 24)
    elif mode == "short":
        paras, _ = get_paragraphs(script, rng, 8)
        usable = height - top - bottom
        mb = max(int(height - (top + rng.uniform(0.66, 0.86) * usable)), bottom)
        m1, l1 = _block(paras, font_path, font_index, width, height, (left, right, top, mb), fs, seed)
        blocks = [(m1, l1, True)]
        ctop = (l1["lines"][-1]["baseline"] + l1["descent"] + 60) if l1["lines"] else top
        closer = convert(str(rng.choice(CLOSERS)), script)
        mc, lc = _block([closer], font_path, font_index, width, height,
                        (left, right, int(ctop), bottom), fs, seed + 21)
        for L in lc["lines"]:
            L["para"] = 3000
        if lc["lines"]:
            dx = (left + (width - right)) / 2 - (lc["lines"][0]["x0"] + lc["lines"][0]["x1"]) / 2
            mc = _shift(mc, lc, dx)
        blocks.append((mc, lc, False))
    else:
        paras, _ = get_paragraphs(script, rng, 8)
        m1, l1 = _block(paras, font_path, font_index, width, height, (left, right, top, bottom), fs, seed)
        blocks = [(m1, l1, True)]

    lines = [L for _, lay, _ in blocks for L in lay["lines"]]
    ref = blocks[0][1]
    layout = {"lines": lines, "ascent": ref["ascent"], "descent": ref["descent"],
              "margins": ref["margins"], "font_size": fs}
    mask = np.maximum.reduce([b[0] for b in blocks])

    rrng = np.random.default_rng(seed + 31)
    rule = np.zeros_like(mask)
    for m, lay, ruled in blocks:
        if ruled and lay["lines"]:
            rule = np.maximum(rule, _ruling(width, height, lay, m, script, rrng))
    if box is not None:
        rule = np.maximum(rule, _box(mask.shape, box, rrng))

    red, mask, highlighted = _rubricate(mask, layout, rng)
    wash = _wash(mask.shape, layout, rng)

    note = convert(str(rng.choice(NOTES)), script) if rng.random() < 0.80 else None
    if note:
        patch = _small(note, font_path, font_index, 26, seed + 3, 520, 90)
        if patch is not None:
            patch = np.rot90(patch)
            y = top + rng.uniform(0, max(1.0, (height - bottom - top - patch.shape[0]) * 0.5))
            _stamp(mask, patch, sl + 18, y)

    folio = None
    if rng.random() < 0.85:
        num = "".join(chr(0x0966 + int(d)) for d in str(int(rng.integers(1, 300))))
        folio = convert("।" + num + "।", script)
        fp = _small(folio, font_path, font_index, 30, seed + 5, 260, 100)
        if fp is not None:
            _stamp(mask, fp, width - right - fp.shape[1] - 10, st + 40)

    return {"black": np.maximum(mask, rule).astype(np.float32), "red": red.astype(np.float32),
            "wash": wash, "layout": layout, "md": _md(layout, note, folio),
            "highlighted": highlighted, "side_note": note, "folio": folio, "mode": mode,
            "overflow": any(b[1]["overflow"] for b in blocks),
            "margins": (left, right, top, bottom), "safe": (sl, sr, st, sb)}


def ink_page(page_rgb, alpha, comp, maps, material, aging_level, seed):
    if material == "palmleaf": 
        g = cv2.GaussianBlur(comp["black"], (0, 0), 2.2)
        a = (np.clip(g * 0.5, 0, 0.35) * _alpha01(alpha))[..., None]
        tint = np.array([0.60, 0.48, 0.32], np.float32)
        f = page_rgb.astype(np.float32) / 255.0
        page_rgb = (np.clip(f * (1.0 - a * (1.0 - tint)), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    ink_name = "lampblack" if material == "palmleaf" else "iron_gall" 
    out = render_ink(page_rgb, alpha, comp["black"], maps["absorbency"], maps["roughness"],
                     np.random.default_rng(seed + 7), ink=ink_name, fiber_map=maps["fiber_map"],
                     aging_level=aging_level, material=material)
    if comp["red"].max() > 0.02:
        out = render_ink(out, alpha, comp["red"], maps["absorbency"], maps["roughness"],
                         np.random.default_rng(seed + 8), ink="vermilion", fiber_map=maps["fiber_map"],
                         aging_level=aging_level, material=material)
    wash = comp.get("wash")
    if wash is not None and wash.max() > 0.02:
        a = (wash * 0.55 * _alpha01(alpha))[..., None]
        tint = np.array([1.0, 0.80, 0.28], np.float32)
        f = out.astype(np.float32) / 255.0
        out = (np.clip(f * (1.0 - a + a * tint), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return out