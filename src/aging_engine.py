import cv2
import numpy as np


# How much of each aging effect shows up at each level.
# Strengths are fractions (0..1) used multiplicatively, not raw pixel offsets.
AGING_PROFILES = {
    "light": {
        "yellowing": 0.05,
        "foxing": 25,
        "water": 1,
        "water_strength": 0.50,
        "edge": 0.30,
    },
    "moderate": {
        "yellowing": 0.14,
        "foxing": 90,
        "water": 2,
        "water_strength": 0.75,
        "edge": 0.60,
    },
    "heavy": {
        "yellowing": 0.26,
        "foxing": 260,
        "water": 4,
        "water_strength": 1.00,
        "edge": 0.95,
    },
}

MAX_SPOTS = 400    # we always generate this many foxing spots and just use the first N — that way
                   # heavy aging is a superset of moderate which is a superset of light
MAX_STAINS = 6     # same idea for water stains


# ---------------------------------------------------------------- helpers

def fbm(width, height, rng, grid=4, octaves=5, persistence=0.55):
    """Fractal Brownian Motion — layered noise at increasing frequencies.
    Each octave has half the impact of the previous one, which gives variation
    at multiple scales simultaneously (think fluffy clouds, not flat fog)."""
    total = np.zeros((height, width), dtype=np.float32)
    amp = 1.0

    for octave in range(octaves):
        g = grid * (2 ** octave)
        gh = max(2, int(round(g * height / width)))

        layer = rng.normal(0, 1, (gh, g)).astype(np.float32)
        layer = cv2.resize(
            layer, (width, height), interpolation=cv2.INTER_CUBIC
        )

        total += amp * layer
        amp *= persistence

    total -= total.mean()
    total /= total.std() + 1e-6
    return total


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def multiply_toward(paper, alpha, tint):
    """Darken and tint the paper without hiding the texture underneath.
    Multiplying keeps the grain and stains visible through the effect —
    that's what makes it feel like the paper changed rather than a layer on top."""
    tint = np.asarray(tint, dtype=np.float32)
    return paper * (1.0 - alpha[:, :, None] * (1.0 - tint[None, None, :]))


def validate_paper(paper):
    paper = np.asarray(paper, dtype=np.float32)

    if paper.ndim != 3 or paper.shape[2] != 3:
        raise ValueError("Expected an RGB image.")

    return paper.copy()


# ---------------------------------------------------------------- effects

def add_yellowing(paper, rng, strength):
    """Simulates cellulose oxidation — the paper slowly turns yellow-brown as it ages.
    Blue fades the most, green a bit, red stays. The effect is patchy rather than
    uniform, because real paper ages unevenly depending on humidity and exposure."""
    height, width = paper.shape[:2]

    broad = fbm(width, height, rng, grid=3, octaves=3)
    mid = fbm(width, height, rng, grid=14, octaves=4)

    field = 0.8 * broad + 0.45 * mid
    amount = strength * np.clip(1.0 + 0.55 * field, 0.15, 1.7)

    tint = [0.85, 0.60, 0.20]   # multiplier at amount == 1
    return multiply_toward(paper, amount, tint)


def add_water_damage(paper, rng, count, strength):
    """Creates tide-mark water stains — the kind you get when a page gets damp
    and dries out, leaving a dark ring at the boundary. We build all stains
    into an accumulated alpha map first, then apply once, so overlapping stains
    blend naturally instead of leaving weird gaps or doubling up."""
    height, width = paper.shape[:2]

    pool = []
    for _ in range(MAX_STAINS):
        pool.append((
            rng.uniform(-0.1 * width, 1.1 * width),
            rng.uniform(-0.1 * height, 1.1 * height),
            rng.uniform(110, 260),
            rng.uniform(130, 300),
            rng.uniform(0.7, 1.0),
            int(rng.integers(0, 2 ** 32)),
        ))

    wash_acc = np.zeros((height, width), dtype=np.float32)
    rim_acc = np.zeros((height, width), dtype=np.float32)

    for cx, cy, rx, ry, s, seed in pool[:count]:
        local = np.random.default_rng(seed)

        m = 2.0
        x1 = max(0, int(cx - rx * m))
        x2 = min(width, int(cx + rx * m))
        y1 = max(0, int(cy - ry * m))
        y2 = min(height, int(cy + ry * m))

        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            continue

        yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)

        warp = np.clip(fbm(bw, bh, local, grid=3, octaves=4), -2.5, 2.5)
        detail = fbm(bw, bh, local, grid=24, octaves=3)
        # Slow variation of rim width/darkness along the boundary
        along = fbm(bw, bh, local, grid=5, octaves=2)

        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        d = d * (1.0 + 0.16 * warp)

        inside = 1.0 - smoothstep(0.80, 1.0, d)
        wash = inside * (0.75 + 0.25 * detail)

        # The rim is asymmetric — softer on the inside, sharper on the outside,
        # and its width varies around the edge (some parts more concentrated).
        width_var = np.clip(1.0 + 0.35 * along, 0.5, 1.6)
        inner = np.exp(-((d - 0.97) / (0.07 * width_var)) ** 2)
        outer = np.exp(-((d - 0.97) / (0.03 * width_var)) ** 2)
        rim = np.where(d < 0.97, inner * 0.9, outer * 1.35).astype(np.float32)
        rim *= np.clip(0.85 + 0.30 * detail, 0.35, 1.3)
        rim *= np.clip(0.8 + 0.3 * along, 0.4, 1.3)
        rim = cv2.GaussianBlur(rim, (0, 0), 1.4)

        wash_acc[y1:y2, x1:x2] = 1.0 - (1.0 - wash_acc[y1:y2, x1:x2]) * (
            1.0 - np.clip(0.30 * s * strength * wash, 0, 1)
        )
        rim_acc[y1:y2, x1:x2] = np.maximum(
            rim_acc[y1:y2, x1:x2],
            np.clip(0.60 * s * strength * rim, 0, 1),
        )

    paper = multiply_toward(paper, wash_acc, [0.84, 0.70, 0.48])
    paper = multiply_toward(paper, rim_acc, [0.66, 0.46, 0.22])
    return paper


def add_foxing(paper, rng, count):
    """Foxing — those rusty-brown spots that appear on old paper from mold or
    oxidation. Small spots are darker and more saturated; large spots are paler.
    Each spot has a soft halo around it, because the discoloration spreads slightly
    into the surrounding paper."""
    height, width = paper.shape[:2]

    # Spread spots across the page using a density field — slightly more common
    # in areas where the field is higher, but not so extreme that they all cluster.
    density = fbm(width, height, rng, grid=3, octaves=3)
    prob = np.exp(0.6 * density).astype(np.float64).ravel()
    prob /= prob.sum()

    # Always generate the full MAX_SPOTS pool, then only use the first `count`.
    # This ensures heavy aging has all the spots that moderate does, plus more.
    idx = rng.choice(height * width, size=MAX_SPOTS, p=prob)
    ys, xs = np.divmod(idx, width)
    radius = np.clip(rng.lognormal(np.log(2.2), 0.6, MAX_SPOTS), 1.0, 11.0)
    power = rng.uniform(0.45, 1.0, MAX_SPOTS)

    core = np.zeros((height, width), dtype=np.float32)
    halo_src = np.zeros((height, width), dtype=np.float32)
    for i in range(count):
        c = (int(xs[i]), int(ys[i]))
        r = int(round(radius[i]))
        cv2.circle(core, c, r, float(power[i]), -1, lineType=cv2.LINE_AA)
        # Each spot gets a halo — a bigger, fainter disc representing
        # the discoloration that spreads into the paper around the spot.
        cv2.circle(
            halo_src, c, r * 2 + 2, float(power[i]) * 0.5, -1,
            lineType=cv2.LINE_AA,
        )

    # Ragged boundary: the edges of a foxing spot aren't clean circles
    n1 = cv2.GaussianBlur(
        rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 1.0
    )
    n2 = cv2.GaussianBlur(
        rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 2.5
    )
    ragged = n1 / (n1.std() + 1e-6) * 0.6 + n2 / (n2.std() + 1e-6) * 0.8

    body = cv2.GaussianBlur(core, (0, 0), 1.3) * (1.0 + 0.55 * ragged)
    solid = np.clip((cv2.GaussianBlur(core, (0, 0), 1.3) - 0.45) * 3.0, 0, 1)
    spot = np.maximum(np.clip((body - 0.15) * 1.8, 0, 1), solid)

    halo = cv2.GaussianBlur(halo_src, (0, 0), 4.0)
    halo = np.clip(halo * 0.9, 0, 0.5)

    # Apply two tints: pale rusty halo first, then a darker (but not black) core on top
    out = multiply_toward(paper, halo, [0.86, 0.72, 0.52])
    out = multiply_toward(out, spot * 0.78, [0.66, 0.44, 0.24])
    return out


def add_edge_aging(paper, rng, strength):
    """The edges of old manuscripts always go brown and dark — from handling,
    exposure to air, and moisture wicking in from the outside. The effect is
    uneven: some stretches are darker, some barely affected."""
    height, width = paper.shape[:2]

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.minimum.reduce([x, width - 1 - x, y, height - 1 - y])

    wobble = fbm(width, height, rng, grid=10, octaves=4)
    patch = fbm(width, height, rng, grid=5, octaves=3)

    band = 0.12 * width
    d = dist + 0.02 * width * wobble

    edge = np.clip(1.0 - d / band, 0.0, 1.0) ** 2
    alpha = strength * edge * np.clip(0.8 + 0.4 * patch, 0.25, 1.5)
    alpha = np.clip(alpha, 0, 1)

    return multiply_toward(paper, alpha, [0.70, 0.50, 0.28])


# ---------------------------------------------------------------- entry

def apply_aging(paper, profile, rng):
    if profile not in AGING_PROFILES:
        raise ValueError(f"Unknown aging profile: {profile}")

    paper = validate_paper(paper)
    config = AGING_PROFILES[profile]

    # Each effect gets its own independent random stream.
    # This means you can change how many foxing spots appear without
    # accidentally shifting the randomness of the yellowing or edge effects.
    seeds = rng.integers(0, 2 ** 32, size=4)
    r_yellow, r_water, r_fox, r_edge = (
        np.random.default_rng(int(s)) for s in seeds
    )

    paper = add_yellowing(paper, r_yellow, config["yellowing"])
    paper = add_water_damage(
        paper, r_water, config["water"], config["water_strength"]
    )
    paper = add_foxing(paper, r_fox, config["foxing"])
    paper = add_edge_aging(paper, r_edge, config["edge"])

    return np.clip(paper, 0, 255)