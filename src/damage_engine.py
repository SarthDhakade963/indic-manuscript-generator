import cv2
import numpy as np

PALM_HOLE_FRACS = (0.33, 0.67)   # string-hole x positions as a fraction of width
PALM_HOLE_R = 15
PALM_CLEAR = 85                  # text keeps this far from each hole centre


def _noise(width, height, rng, scale):
    """Smooth noise whose features are about `scale` pixels across."""
    gw = max(2, int(round(width / scale)))
    gh = max(2, int(round(height / scale)))
    n = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    n = cv2.resize(n, (width, height), interpolation=cv2.INTER_CUBIC)
    n -= n.mean()
    n /= n.std() + 1e-6
    return n


def _smooth_1d(length, rng, knots):
    """1D smooth random curve, ~unit variance, `knots` control points."""
    knots = max(2, int(knots))
    k = rng.normal(0, 1, knots).astype(np.float32).reshape(1, -1)
    return cv2.resize(k, (length, 1), interpolation=cv2.INTER_CUBIC).ravel()


# creases

def add_crease_shadows(paper, rng, count=3):
    """Simulates physical folds in the paper. Each crease has a dark valley on
    one side, a bright ridge beside it (where the paper bent outward), a subtle
    brightness step across the fold, and a worn highlight along the fold line itself.
    The crease follows a gently curving path — no straight lines here."""
    height, width = paper.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    tone = np.array([1.0, 0.97, 0.90], dtype=np.float32)

    for _ in range(count):
        horizontal = rng.random() < 0.6
        long_side = width if horizontal else height
        short_side = height if horizontal else width

        # Build the fold path: a baseline position, a slight diagonal tilt,
        # a few gentle bends across the page, and fine jitter for realism.
        t = np.arange(long_side, dtype=np.float32)
        pos = rng.uniform(0.2, 0.8) * short_side
        tilt = rng.uniform(-0.05, 0.05)
        curve = (
            pos
            + tilt * (t - long_side / 2)
            + 10.0 * _smooth_1d(long_side, rng, 6)
            + 1.2 * _smooth_1d(long_side, rng, long_side // 14)
        )

        # Folds often don't reach all the way to the page edge — they fade out
        # gradually at the start and end rather than stopping abruptly.
        start = rng.uniform(0.0, 0.2) * long_side
        end = long_side - rng.uniform(0.0, 0.2) * long_side
        u = np.minimum(
            np.clip((t - start) / 160.0, 0, 1),
            np.clip((end - t) / 160.0, 0, 1),
        )
        taper = u * u * (3.0 - 2.0 * u)
        taper *= np.clip(0.8 + 0.25 * _smooth_1d(long_side, rng, 8), 0.4, 1.2)

        if horizontal:
            side = yy - curve[None, :]
            taper2d = taper[None, :]
        else:
            side = xx - curve[:, None]
            taper2d = taper[:, None]

        core = np.exp(-(side / 1.8) ** 2)
        valley = np.exp(-(side / 2.4) ** 2)
        shade = np.exp(-(side / 10.0) ** 2)
        ridge = np.exp(-((side - 5.5) / 2.6) ** 2)
        step = np.tanh(side / 5.0) * np.exp(-np.abs(side) / 70.0)

        grit = _noise(width, height, rng, 3)
        patch = _noise(width, height, rng, 40)
        broken = np.clip(0.55 + 0.6 * patch, 0.2, 1.2)
        wear = core * np.clip(0.3 + 0.7 * grit, 0.0, 1.0) * broken

        s = float(rng.uniform(0.8, 1.2))
        delta = (
            -20.0 * valley * broken
            - 8.0 * shade
            + 11.0 * ridge
            + 4.0 * step
            + 15.0 * wear
        ) * taper2d * s

        paper += delta[:, :, None] * tone[None, None, :]

    return paper


# torn edge

def generate_torn_edge(width, height, rng):
    """Generates an irregular torn paper boundary. Returns two things:
    - alpha (uint8): the page mask — white inside, black outside
    - signed (float32): signed distance from the torn edge (positive = inside the page).
      We use the signed distance later to add darkening and fiber details right at the edge."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.minimum.reduce([x, width - 1 - x, y, height - 1 - y])

    # The basic offset from the rectangular boundary — large drift, medium bumps, fine fiber roughness
    offset = (
        12.0
        + 9.0 * _noise(width, height, rng, 260)    # slow drift over hundreds of pixels
        + 4.0 * _noise(width, height, rng, 45)     # medium-scale bumps
        + 1.2 * _noise(width, height, rng, 6)      # tiny fiber-scale roughness
    )

    # Add random bites taken out of each edge — places where the paper tore more deeply.
    # Every side gets at least one bite, plus a few random extras.
    n_bites = int(rng.integers(4, 8))
    sides = [int(s) for s in rng.permutation(4)]
    sides += [int(s) for s in rng.integers(0, 4, n_bites - 4)]
    for side in sides:
        t = rng.uniform(0.1, 0.9)
        cx = [t * width, t * width, 0, width - 1][side]
        cy = [0, height - 1, t * height, t * height][side]
        radius = rng.uniform(40, 110)
        amp = rng.uniform(12, 35)
        d2 = ((x - cx) ** 2 + (y - cy) ** 2) / radius ** 2
        offset += amp * np.exp(-d2)

    offset = np.maximum(offset, 1.0)
    signed = dist - offset

    alpha = np.clip(signed / 1.5 + 0.5, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8), signed.astype(np.float32)


def generate_torn_mask(width, height, rng):
    return generate_torn_edge(width, height, rng)[0]


def add_torn_edge_darkening(paper, signed, rng, strength):
    """The paper near a torn edge gets darker and browner from handling and exposure.
    We measure from the torn boundary (not the rectangular page edge), so bites
    and bumps in the tear correctly show as darker, not as pale unexposed paper."""
    height, width = signed.shape

    band = 38.0 * np.clip(1.0 + 0.5 * _noise(width, height, rng, 120), 0.4, 1.8)
    a = np.clip(1.0 - signed / band, 0.0, 1.0) ** 2
    a *= np.clip(0.8 + 0.35 * _noise(width, height, rng, 70), 0.3, 1.4)
    a = np.clip(strength * a, 0.0, 0.9)

    tint = np.array([0.72, 0.54, 0.32], dtype=np.float32)
    return paper * (1.0 - a[:, :, None] * (1.0 - tint[None, None, :]))


def add_torn_rim(paper, signed, rng):
    """When paper tears, the cross-section exposes pale fibrous pulp along the fracture.
    This looks like a thin bright rim right at the edge. But it's not uniform —
    some stretches stay darker, and the fiber texture makes it look ragged."""
    height, width = signed.shape

    rim_w = 7.0 * np.clip(1.0 + 0.4 * _noise(width, height, rng, 50), 0.5, 1.6)
    fibres = np.clip(0.65 + 0.35 * _noise(width, height, rng, 3), 0.0, 1.3)
    patch = np.clip(0.55 + 0.9 * _noise(width, height, rng, 35), 0.0, 1.2)

    rim = np.clip(1.0 - signed / rim_w, 0.0, 1.0) ** 1.5
    rim = np.where(signed > -1.0, rim, 0.0) * fibres * patch
    a = np.clip(0.55 * rim, 0.0, 0.7)[:, :, None]

    pale = np.clip(paper * 1.22 + 16.0, 0, 255)
    return paper * (1.0 - a) + pale * a


#---- entry

def apply_damage(
    paper,
    rng,
    crease_count=3,
    torn_edges=True,
    edge_strength=0.4,
):
    paper = np.asarray(paper, dtype=np.float32).copy()

    paper = add_crease_shadows(paper, rng, crease_count)

    height, width = paper.shape[:2]

    if torn_edges:
        alpha, signed = generate_torn_edge(width, height, rng)
        paper = add_torn_edge_darkening(paper, signed, rng, edge_strength)
        paper = add_torn_rim(paper, signed, rng)
    else:
        alpha = np.full((height, width), 255, dtype=np.uint8)

    return np.clip(paper, 0, 255), alpha


def generate_palm_edge(width, height, rng):
    """Straight-cut long edges, ragged split short edges, two transparent string holes."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    top_off = 6.0 + 2.5 * _smooth_1d(width, rng, 14) + 0.8 * _smooth_1d(width, rng, width // 9)
    bot_off = 6.0 + 2.5 * _smooth_1d(width, rng, 14) + 0.8 * _smooth_1d(width, rng, width // 9)
    sides = []
    for _ in range(2):
        off = 16.0 + 9.0 * _smooth_1d(height, rng, 10) + 4.0 * _smooth_1d(height, rng, height // 6)
        for _ in range(int(rng.integers(1, 4))):
            c = int(rng.integers(20, height - 20))
            w2 = int(rng.integers(2, 5))
            lo, hi = max(0, c - w2), min(height, c + w2)
            off[lo:hi] += rng.uniform(12, 30) * rng.uniform(0.4, 1.0, hi - lo)
        sides.append(np.maximum(off, 2.0))
    d_top = y - top_off[None, :]
    d_bot = (height - 1 - y) - bot_off[None, :]
    d_left = x - sides[0][:, None]
    d_right = (width - 1 - x) - sides[1][:, None]
    signed = np.minimum.reduce([d_top, d_bot, d_left, d_right])  
    for f in PALM_HOLE_FRACS:  
        ang = np.arctan2(y - height / 2, x - f * width)  
        ph = rng.uniform(0, 6.28, 3)  
        wob = 1.2 * np.sin(2 * ang + ph[0]) + 0.8 * np.sin(3 * ang + ph[1]) + 0.5 * np.sin(5 * ang + ph[2])  
        signed = np.minimum(signed, np.hypot(x - f * width, y - height / 2) - PALM_HOLE_R - wob)  
    alpha = np.clip(signed / 1.5 + 0.5, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8), signed.astype(np.float32)


def apply_palm_damage(paper, rng, edge_strength=0.4):
    paper = np.asarray(paper, dtype=np.float32).copy()
    height, width = paper.shape[:2]
    alpha, signed = generate_palm_edge(width, height, rng)
    paper = add_torn_edge_darkening(paper, signed, rng, edge_strength)
    paper = add_torn_rim(paper, signed, rng)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    tint = np.array([0.55, 0.42, 0.28], dtype=np.float32)
    for f in PALM_HOLE_FRACS:
        d = np.hypot(x - f * width, y - height / 2) - PALM_HOLE_R
        a = 0.8 * np.exp(-(np.maximum(d, 0) / 12.0) ** 2)
        paper = paper * (1.0 - a[:, :, None] * (1.0 - tint[None, None, :]))
    return np.clip(paper, 0, 255), alpha