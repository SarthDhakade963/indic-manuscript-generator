
import cv2
import numpy as np
from PIL import Image, ImageOps
from dataclasses import dataclass


MATERIALS = {
    "paper": {
        "base": (218, 204, 179),   # warm off-white, like old handmade paper
        "large": 3.5,              # how much the big slow tone shifts show up
        "medium": 2.0,             # mid-scale variation (think uneven pulp density)
        "grain": 2.4,              # fine surface texture — raised from 1.15 to look grittier
        "fiber": 16.0,             # how visible the individual fiber strands are (was 9.0)
        "absorbency": 0.62,        # how much the paper soaks up ink
        "roughness": 0.35,         # surface tooth — affects ink bleeding at stroke edges
    },
    "parchment": {
        "base": (211, 190, 157),   # slightly darker and more golden than paper
        "large": 4.5,              # parchment has stronger tonal variation (it's animal skin)
        "medium": 2.5,
        "grain": 2.8,              # was 1.4 — parchment is noticeably grainier up close
        "fiber": 4.0,              # lower than paper — parchment doesn't have loose fibers (was 0.75)
        "absorbency": 0.48,        # parchment is less absorbent, ink sits more on the surface
        "roughness": 0.48,         # slightly rougher due to scraped skin texture
    },
    "palmleaf": { 
        "base": (168, 140, 84),
        "large": 5.0,
        "medium": 3.0,
        "grain": 2.0,
        "fiber": 12.0,
        "absorbency": 0.30,
        "roughness": 0.55,
    },
}


@dataclass
class MaterialSurface:
    rgb: np.ndarray
    absorbency: np.ndarray
    fiber_map: np.ndarray
    roughness: np.ndarray


def spatial_noise(width, height, rng, grid, blur=0):
    grid_h = max(2, int(grid * height / width))

    noise = rng.normal(0, 1, (grid_h, grid)).astype(np.float32)

    noise = cv2.resize(
        noise, (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    if blur:
        noise = cv2.GaussianBlur(noise, (0, 0), blur)

    noise -= noise.mean()
    noise /= noise.std() + 1e-6

    return noise


def generate_fiber_map(width, height, rng):
    """Draws individual paper fiber strands as faint curved lines across the canvas.
    Most fibers are lighter than the background (they reflect more light), a few are
    darker. We leave the result un-normalised so it stays sparse — only the actual
    fiber locations have any value, the rest is zero."""
    canvas = np.zeros((height, width), dtype=np.float32)

    count = int(width * height / 450)   # roughly 4500 strands for a 1200×1600 page

    for _ in range(count):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        angle = rng.uniform(0, np.pi)
        length = rng.lognormal(np.log(35), 0.6)
        n = max(3, int(length / 6))
        step = length / n

        pts = [(x, y)]
        for _ in range(n):
            angle += rng.normal(0, 0.25)          # gentle curvature
            x += np.cos(angle) * step
            y += np.sin(angle) * step
            pts.append((x, y))

        pts = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
        amp = rng.uniform(0.35, 1.0) * (1.0 if rng.random() < 0.65 else -1.0)  # ~35% of fibers are dark

        cv2.polylines(
            canvas, [pts], False, float(amp),
            thickness=1, lineType=cv2.LINE_AA,
        )

    return cv2.GaussianBlur(canvas, (0, 0), 0.5)


def load_reference_texture(path, width, height):
    image = Image.open(path).convert("RGB")

    image = ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS
    )

    return np.asarray(image, dtype=np.float32)

def generate_skin_detail(width, height, rng):
    """Adds the three things that make parchment look like parchment and not paper:
    - cloudy translucency (where the skin was thinner or thicker)
    - faint vein lines running across the surface
    - tiny dark follicle dots where the hair once was
    Returns three separate float maps so the caller can tune each one."""
    # Mid-scale mottling from the uneven thickness of the hide, roughly 20-60px features
    mottle = spatial_noise(width, height, rng, 40, blur=3)
    mottle += 0.6 * spatial_noise(width, height, rng, 110, blur=1.5)
    mottle /= mottle.std() + 1e-6

    # Faint vein lines — long, wandering, very softly blurred so they barely show
    veins = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(width * height / 120000) + 3):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        angle = rng.uniform(0, np.pi)
        pts = [(x, y)]
        for _ in range(40):
            angle += rng.normal(0, 0.12)
            x += np.cos(angle) * 14
            y += np.sin(angle) * 14
            pts.append((x, y))
        pts = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(veins, [pts], False, float(rng.uniform(0.4, 1.0)),
                      thickness=2, lineType=cv2.LINE_AA)
    veins = cv2.GaussianBlur(veins, (0, 0), 2.5)

    # Follicle specks — tiny dark dots scattered across the surface, one per old hair follicle
    specks = np.zeros((height, width), dtype=np.float32)
    count = int(width * height / 2500)
    xs = rng.integers(0, width, count)
    ys = rng.integers(0, height, count)
    for x, y in zip(xs, ys):
        cv2.circle(specks, (int(x), int(y)), 1,
                   float(rng.uniform(0.3, 1.0)), -1, lineType=cv2.LINE_AA)
    specks = cv2.GaussianBlur(specks, (0, 0), 0.8)

    return mottle, veins, specks

def generate_palm_striations(width, height, rng):  
    out = np.zeros((height, width), dtype=np.float32)
    for rows, cols, amp in ((height // 3, 8, 1.0), (height // 2, 20, 0.7), (height, 50, 0.45)):
        n = rng.normal(0, 1, (max(2, rows), cols)).astype(np.float32)
        out += amp * cv2.resize(n, (width, height), interpolation=cv2.INTER_CUBIC)
    out /= out.std() + 1e-6
    cracks = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(width * height / 9000)):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        length = rng.uniform(80, 500)
        n = max(3, int(length / 12))
        pts = [(x, y)]
        for _ in range(n):
            x += length / n
            y += rng.normal(0, 0.6)
            pts.append((x, y))
        pts = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(cracks, [pts], False, -float(rng.uniform(0.5, 1.2)),
                      thickness=1, lineType=cv2.LINE_AA)
    return (0.6 * out + cracks).astype(np.float32)

def generate_material_surface(
    width,
    height,
    material="paper",
    seed=42,
    reference_path=None,
):
    if material not in MATERIALS:
        raise ValueError(f"Unknown material: {material}")

    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")

    rng = np.random.default_rng(seed)
    config = MATERIALS[material]

    if reference_path:
        rgb = load_reference_texture(reference_path, width, height)
    else:
        rgb = np.empty((height, width, 3), dtype=np.float32)
        rgb[:] = config["base"]

    # Lay down large-scale and medium-scale tonal variation.
    # The warm multiplier makes the variation yellower-brown rather than neutral grey.
    large = spatial_noise(width, height, rng, 9, blur=20)
    medium = spatial_noise(width, height, rng, 65, blur=2)

    # If we're starting from a reference photo, be more conservative so we
    # don't clobber the photo's own texture with too much extra noise.
    strength = 0.45 if reference_path else 1.0

    warm = np.array([1.0, 0.84, 0.62], dtype=np.float32)

    rgb += (
        large[:, :, None]
        * config["large"] * strength
        * warm[None, None, :]
    )

    rgb += (
        medium[:, :, None]
        * config["medium"] * strength
        * warm[None, None, :]
    )

    # Fine paper grain — slight blur so it's not pure pixel noise
    grain = rng.normal(0, 1, (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.6)
    grain /= grain.std() + 1e-6
    rgb += (grain * config["grain"])[:, :, None]

    if material == "parchment":
        # Parchment doesn't have paper fibers — instead it has skin detail
        fiber_map = np.zeros((height, width), dtype=np.float32)
        mottle, veins, specks = generate_skin_detail(width, height, rng)
        warm = np.array([1.0, 0.85, 0.65], dtype=np.float32)
        tone = np.array([0.45, 0.75, 1.0], dtype=np.float32)   # mostly blue-channel
        rgb += mottle[:, :, None] * 5.0 * tone[None, None, :]
        rgb -= veins[:, :, None] * 6.0 * tone[None, None, :]
        rgb -= specks[:, :, None] * 28.0 * np.array([1.0, 1.0, 0.8],
                                                    dtype=np.float32)
    elif material == "palmleaf":  
        fiber_map = generate_palm_striations(width, height, rng)
        rgb += fiber_map[:, :, None] * config["fiber"] * np.array([1.0, 0.9, 0.6], dtype=np.float32)
        rgb += spatial_noise(width, height, rng, 6, blur=15)[:, :, None] * np.array([-5.0, 4.0, -3.0], dtype=np.float32)
    else:
        # Paper gets actual elongated fiber strands
        fiber_map = generate_fiber_map(width, height, rng)

        rgb += (
            fiber_map[:, :, None]
            * config["fiber"]
            * np.array([1.0, 0.85, 0.65], dtype=np.float32)
        )

    # Store absorbency and roughness maps — the ink engine needs these later
    # to decide how much ink bleeds and how uneven stroke edges look.
    absorption_noise = spatial_noise(width, height, rng, 55, blur=2)
    roughness_noise = spatial_noise(width, height, rng, 75, blur=1)

    absorbency = np.clip(
        config["absorbency"] + absorption_noise * 0.025,
        0.05, 0.95
    ).astype(np.float32)

    roughness = np.clip(
        config["roughness"] + roughness_noise * 0.025,
        0.0, 1.0
    ).astype(np.float32)

    return MaterialSurface(
        rgb=np.clip(rgb, 0, 255).astype(np.float32),
        absorbency=absorbency,
        fiber_map=fiber_map,
        roughness=roughness,
    )


def generate_material(
    width,
    height,
    material="paper",
    seed=42,
    reference_path=None,
):
    return generate_material_surface(
        width, height, material, seed, reference_path
    ).rgb
