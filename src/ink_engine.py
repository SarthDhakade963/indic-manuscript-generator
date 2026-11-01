"""Ink compositing for synthetic manuscripts (iron-gall / carbon)."""
import numpy as np
from PIL import Image

INKS = {
    "iron_gall": {
        "fresh": (20, 16, 18),
        "aged": (92, 58, 34),
        "brown_amount": 1.0,
        "max_opacity": 0.94,
        "bleed": 1.0,
        "fade": 1.0,
    },
    "carbon": {
        "fresh": (14, 14, 16),
        "aged": (24, 24, 26),
        "brown_amount": 0.3,
        "max_opacity": 0.97,
        "bleed": 0.45,
        "fade": 0.25,
    },
}

# Ink sits on parchment, soaks into paper.
MATERIAL_BLEED = {"paper": 1.0, "parchment": 0.3}


def _gauss(a, sigma):
    if sigma < 0.3:
        return a
    r = int(np.ceil(sigma * 3))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    h, w = a.shape
    p = np.pad(a, ((0, 0), (r, r)), mode="edge")
    acc = np.zeros_like(a)
    for i, wt in enumerate(k):
        acc += wt * p[:, i:i + w]
    p = np.pad(acc, ((r, r), (0, 0)), mode="edge")
    out = np.zeros_like(a)
    for i, wt in enumerate(k):
        out += wt * p[i:i + h, :]
    return out


def _norm(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [2, 98])
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _lowfreq(shape, cell, rng):
    h, w = shape
    gh, gw = h // cell + 3, w // cell + 3
    g = rng.random((gh, gw)).astype(np.float32)
    im = Image.fromarray(g).resize((gw * cell, gh * cell), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32)[:h, :w]
    return (a - a.mean()) / (a.std() + 1e-6)


def _sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def render_ink(page_rgb, alpha, stroke_mask, absorbency, roughness, rng,
               ink="iron_gall", fiber_map=None, aging_level=0.5, material="paper"):
    cfg = INKS[ink]
    bleed = cfg["bleed"] * MATERIAL_BLEED.get(material, 1.0)
    page = page_rgb.astype(np.float32) / 255.0
    al = alpha.astype(np.float32)
    if al.max() > 1.5:
        al = al / 255.0
    m = np.clip(stroke_mask.astype(np.float32), 0, 1)
    shape = m.shape
    ab = _norm(absorbency)
    rg = _norm(roughness)
    age = float(np.clip(aging_level, 0, 1))

    # Spread: more blur where the page is absorbent.
    s_lo = _gauss(m, 0.6)
    s_hi = _gauss(m, 1.6)
    t = np.clip(ab * bleed, 0, 1)
    blur = s_lo * (1 - t) + s_hi * t

    # Hairy edges: jitter the ink threshold with fibre structure (paper)
    # or fine pitting (parchment), so the boundary itself becomes ragged.
    if fiber_map is not None and float(np.std(fiber_map)) > 1e-6:
        f = (fiber_map - fiber_map.mean()) / (fiber_map.std() + 1e-6)
        f = np.clip(f, -2.0, 2.0)
        jit = 0.05 * bleed * f
        halo_mod = 0.7 + 0.3 * np.clip(f, -1, 1)
    else:
        n = _gauss(rng.standard_normal(shape).astype(np.float32), 0.8)
        n = n / (n.std() + 1e-6)
        jit = 0.04 * np.clip(n, -2, 2)
        halo_mod = 1.0
    lo = 0.08 + 0.10 * (1.0 - bleed)
    e0 = np.maximum(lo + jit, 0.03)
    e1 = np.maximum(0.50 + jit, e0 + 0.20)
    s = np.clip(_sstep(e0, e1, blur), 0, 1)
    s = s * _sstep(0.0, 0.02, blur)

    # Wicking halo around strokes, strongest on absorbent paper.
    halo = _gauss(m, 3.0) * ab * 0.12 * bleed * halo_mod

    # Density variation: darker rim, dry patches, granulation, tooth.
    edge = 4.0 * s * (1.0 - s)
    dens = 1.0 + 0.12 * edge
    dry = 0.5 * _lowfreq(shape, 48, rng) + 0.5 * _lowfreq(shape, 9, rng)
    dens *= 1.0 - 0.40 * _sstep(0.2, 1.4, dry)
    gran = np.clip(_lowfreq(shape, 3, rng), -2, 2)
    dens *= 1.0 - 0.06 * gran
    dens *= 1.0 - 0.18 * rg
    dens = np.clip(dens, 0.35, 1.15)

    d = s * dens
    d = d + halo * (1.0 - d)
    d = np.clip(d * cfg["max_opacity"], 0, 1)

    # Fade where the page is stained / dark, stronger on older pages.
    lum = page.mean(axis=2)
    inside = al > 0.5
    ref = np.percentile(lum[inside], 90) if inside.any() else 1.0
    stain = np.clip((ref - lum) / max(ref, 1e-3), 0, 1)
    fade = np.clip(stain * 3.0, 0, 1) * 0.65 * (0.3 + age) * cfg["fade"]
    d = d * (1.0 - fade)

    d = d * al

    # Colour
    fresh = np.array(cfg["fresh"], dtype=np.float32)
    aged = np.array(cfg["aged"], dtype=np.float32)
    col = fresh + (aged - fresh) * (age * cfg["brown_amount"])
    tint = (col / 255.0)[None, None, :]
    out = page * (1.0 - d[..., None] * (1.0 - tint))
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)