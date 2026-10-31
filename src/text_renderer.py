"""Text -> stroke_mask. No ink here: this only decides where ink goes."""
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFont, features


def load_font(path, size, index=0):
    if features.check("raqm"):
        return ImageFont.truetype(path, size, index=index, layout_engine=ImageFont.Layout.RAQM)
    # RAQM not available – fall back to basic layout engine.
    # Simple scripts (Modi, Sharada) still render; Devanagari conjuncts may be imperfect.
    return ImageFont.truetype(path, size, index=index, layout_engine=ImageFont.Layout.BASIC)


def _wrap(words, font, max_w, space_w):
    lines, cur, cur_w = [], [], 0.0
    for w in words:
        ww = font.getlength(w)
        add = ww if not cur else space_w + ww
        if cur and cur_w + add > max_w:
            lines.append(cur)
            cur, cur_w = [w], ww
        else:
            cur.append(w)
            cur_w += add
    if cur:
        lines.append(cur)
    return lines


def render_text_mask(width, height, paragraphs, font_path, font_size=38, weight=0.4,
                     margins=(150, 150, 220, 200), line_spacing=1.55, para_gap=0.4,
                     fill_page=True, ss=3, seed=0, font_index=0):
    """Returns (mask float32 HxW in 0..1, layout dict).

    margins = (left, right, top, bottom) in output pixels.
    weight  = extra pen thickness in output pixels (0 = font's own weight).
    Words are drawn one at a time so each gets its own tiny baseline wobble,
    gap and ink load. Each line also gets a slight overall slope.
    """
    rng = np.random.default_rng(seed)
    font = load_font(font_path, int(font_size * ss), font_index)
    ascent, descent = font.getmetrics()
    line_h = (ascent + descent) * line_spacing
    left, right, top, bottom = [m * ss for m in margins]
    max_w = width * ss - left - right
    space_w = font.getlength(" ")
    stroke = max(0, int(round(weight * ss)))

    img = Image.new("L", (width * ss, height * ss), 0)
    d = ImageDraw.Draw(img)

    y = top + ascent
    lines_out = []
    paras = list(paragraphs)
    body = paras[1:] if len(paras) > 1 else paras
    pi = 0
    full = False
    while not full:
        if pi >= len(paras) and not fill_page:
            break
        para = paras[pi] if pi < len(paras) else body[(pi - len(paras)) % len(body)]
        pi += 1
        for line in _wrap(para.split(), font, max_w, space_w):
            if y + descent > height * ss - bottom:
                full = True
                break
            slope = rng.normal(0, 1.5 * ss) / max_w
            x = left
            words_out = []
            for w in line:
                ww = font.getlength(w)
                yy = y + slope * (x - left) + rng.normal(0, 0.5 * ss)
                load = int(255 * rng.uniform(0.88, 1.0))
                pad = 6 * ss                                  
                cw = int(ww) + 4 * pad
                chh = int(ascent + descent) + 2 * pad
                wi = Image.new("L", (cw, chh), 0)
                ImageDraw.Draw(wi).text(
                    (2 * pad, pad + ascent), w, font=font, fill=load,
                    anchor="ls", stroke_width=stroke, stroke_fill=load)
                wi = wi.rotate(float(rng.normal(0, 0.9)), resample=Image.BICUBIC,
                               center=(2 * pad, pad + ascent))
                ox = int(round(x - 2 * pad))
                oy = int(round(yy - pad - ascent))
                box = (ox, oy, ox + cw, oy + chh)
                img.paste(ImageChops.lighter(img.crop(box), wi), (ox, oy))
                words_out.append({"text": w, "x0": x / ss, "x1": (x + ww) / ss, "y": yy / ss})   # LAY_V1
                x += ww + space_w * rng.uniform(0.85, 1.3)
            lines_out.append({"baseline": y / ss, "x0": left / ss,
                              "x1": words_out[-1]["x1"],               # LAY_V1 true right edge
                              "text": " ".join(line), "words": words_out, "para": pi - 1})
            y += line_h
        y += line_h * para_gap

    mask = np.asarray(img.resize((width, height), Image.LANCZOS), dtype=np.float32) / 255.0
    layout = {"lines": lines_out, "ascent": ascent / ss, "descent": descent / ss,
              "margins": margins, "font_size": font_size}
    return mask, layout


def render_ruling(width, height, layout, text_mask, ss=3, value=0.55, gap=7):
    """Ruled lines sit just above each line's headline (shirorekha); text hangs from them."""
    left, right, top, bottom = layout["margins"]
    img = Image.new("L", (width * ss, height * ss), 0)
    d = ImageDraw.Draw(img)
    fill = int(255 * value)
    lw = max(1, int(1.5 * ss))
    xl, xr = (left - 30) * ss, (width - right + 30) * ss
    rows = []
    for L in layout["lines"]:
        b = L["baseline"]
        y0 = int(max(0, b - layout["ascent"]))
        y1 = int(min(height, b))
        x0 = int(L["x0"])
        x1 = int(min(width, L["x1"]))
        band = text_mask[y0:y1, x0:x1].sum(axis=1)
        hy = y0 + int(np.argmax(band))        # densest row = the headline
        ry = hy - gap
        rows.append(ry)
        d.line([(xl, ry * ss), (xr, ry * ss)], fill=fill, width=lw)
    if rows:
        y_top = (rows[0] - 20) * ss
        y_bot = (rows[-1] + layout["ascent"] + layout["descent"]) * ss
        d.line([(xl, y_top), (xl, y_bot)], fill=fill, width=lw)
    img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0

def handwrite(mask, rng, warp=1.6, fine=0.6, pen=1.0):   # HW_V1
    """Wander the strokes and vary pen width. Input/output: float32 0..1."""
    h, w = mask.shape

    def field(cell, amp):
        g = rng.standard_normal((h // cell + 3, w // cell + 3)).astype(np.float32)
        return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC) * amp

    dx = field(70, warp) + field(16, fine)
    dy = field(70, warp) + field(16, fine)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    out = cv2.remap(mask.astype(np.float32), xs + dx, ys + dy,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    k = np.clip(field(120, 0.5), -1.0, 1.0)               # gentle pressure drift
    soft = cv2.GaussianBlur(out, (0, 0), 1.0)
    thick = np.clip((soft - (0.5 - 0.06 * pen * k)) * 3.0 + 0.5, 0, 1)
    res = 0.5 * out + 0.5 * thick
    res[out < 0.02] = 0.0                                  # no ink where there was no stroke
    return np.clip(res, 0, 1).astype(np.float32)