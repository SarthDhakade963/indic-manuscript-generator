"""Page deformation (DEFORM_V1): smooth warp, edge curl and fold applied to RGB, alpha and label masks together."""
import numpy as np
import cv2


def _field(h, w, rng, cell):
    g = rng.standard_normal((h // cell + 3, w // cell + 3)).astype(np.float32)
    f = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return f / (f.std() + 1e-6)


def _sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def deform(rgb, alpha, rng, kind="paper", masks=(), strength=1.0):
    """Returns (rgb_uint8, alpha_uint8, [warped masks], info). out(x, y) = src(x + dx, y + dy)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    alpha = np.ascontiguousarray(alpha, dtype=np.uint8)
    h, w = alpha.shape[:2]
    s = float(min(h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    shade = 0.03 * _field(h, w, rng, max(40, int(s // 2)))
    info = {"curl": None, "fold": None}
    palm = kind == "palmleaf"

    # 1. smooth surface warp
    amp = (0.006 if palm else 0.010) * s * strength
    cell = max(40, int(s // 3))
    dx += amp * _field(h, w, rng, cell)
    dy += amp * _field(h, w, rng, cell)
    if palm:  # a leaf bows along its length and catches light unevenly
        dy += rng.uniform(-8, 8) * strength * np.sin(np.pi * xx / w + rng.uniform(-0.4, 0.4))
        shade += rng.uniform(-0.05, 0.05) * (yy / h - 0.5) * 2.0

    # 2. edge curl (paper / parchment only)
    if not palm and rng.random() < 0.7:
        side = int(rng.integers(0, 4))                       # 0 left, 1 right, 2 top, 3 bottom
        d = [xx, w - 1 - xx, yy, h - 1 - yy][side]
        t = [yy / h, yy / h, xx / w, xx / w][side]
        cw = rng.uniform(0.06, 0.12) * s
        c, half = rng.uniform(0.25, 0.75), rng.uniform(0.2, 0.4)
        env = _sstep(1.0 - np.abs(t - c) / half)
        shift = 0.35 * cw * (1.0 - np.clip(d / cw, 0, 1)) ** 2 * env * strength
        if side == 0:
            dx -= shift
        elif side == 1:
            dx += shift
        elif side == 2:
            dy -= shift
        else:
            dy += shift
        shade += env * (0.12 * np.exp(-((d - 0.15 * cw) / (0.12 * cw)) ** 2)
                        - 0.20 * np.exp(-((d - 1.1 * cw) / (0.25 * cw)) ** 2))
        info["curl"] = ["left", "right", "top", "bottom"][side]

    # 3. fold (paper / parchment only)
    if not palm and rng.random() < 0.45:
        vertical = rng.random() < 0.45
        pos, tilt = rng.uniform(0.3, 0.7), rng.uniform(-0.04, 0.04)
        sd = (xx - pos * w + tilt * (yy - h / 2)) if vertical else (yy - pos * h + tilt * (xx - w / 2))
        step = rng.uniform(4, 9) * strength * np.tanh(sd / (0.05 * s))
        if vertical:
            dx += step
        else:
            dy += step
        shade += (-0.20 * np.exp(-(sd / (0.010 * s)) ** 2)
                  + 0.08 * np.exp(-((sd - 0.02 * s) / (0.03 * s)) ** 2)
                  + 0.05 * np.tanh(sd / (0.10 * s)))
        info["fold"] = "vertical" if vertical else "horizontal"

    mx, my = xx + dx, yy + dy
    out = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = np.clip(out * (1.0 + shade[..., None]), 0, 255).astype(np.uint8)
    a = cv2.remap(alpha, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ms = [cv2.remap(np.asarray(m, np.float32), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) for m in masks]
    return out, a, ms, info