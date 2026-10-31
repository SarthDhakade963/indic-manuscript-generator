
from pathlib import Path

import numpy as np
from PIL import Image

from src.material_texture import generate_material_surface
from src.aging_engine import apply_aging
from src.damage_engine import apply_damage, apply_palm_damage


MATERIALS = ["paper", "parchment", "palmleaf"]
AGING_LEVELS = ["light", "moderate", "heavy"]
EDGE_DARKEN = {"light": 0.25, "moderate": 0.40, "heavy": 0.60}


def generate_manuscript_background(
    width=1200,
    height=1600,
    material="paper",
    aging="moderate",
    seed=42,
    reference_path=None,
    damage=True,
    return_material_maps=False,
):
    if material not in MATERIALS:
        raise ValueError(f"Unknown material: {material}")

    if aging != "mixed" and aging not in AGING_LEVELS:
        raise ValueError(f"Unknown aging profile: {aging}")

    rng = np.random.default_rng(seed)

    if aging == "mixed":
        aging = str(rng.choice(AGING_LEVELS))

    # Give each sub-module its own independent RNG stream.
    # This way tweaking one module's parameters won't accidentally change
    # the random choices made by all the others.
    material_seed = int(rng.integers(0, 2**32))
    aging_seed = int(rng.integers(0, 2**32))
    damage_seed = int(rng.integers(0, 2**32))

    material_rng = np.random.default_rng(material_seed)
    aging_rng = np.random.default_rng(aging_seed)
    damage_rng = np.random.default_rng(damage_seed)

    # 1. Generate the physical material surface (texture + fiber maps)
    surface = generate_material_surface(
        width=width,
        height=height,
        material=material,
        seed=material_seed,
        reference_path=reference_path,
    )

    paper = surface.rgb.copy()

    # 2. Age the paper (yellowing, foxing, water stains, edge darkening)
    paper = apply_aging(
        paper,
        profile=aging,
        rng=aging_rng,
        damp=0.5 if material == "palmleaf" else 1.0,
    )

    # 3. Add physical damage: creases and torn edges
    crease_count = 0
    if material == "palmleaf":
        paper, alpha = apply_palm_damage(paper, damage_rng, edge_strength=EDGE_DARKEN[aging] * 0.5)
    elif damage:
        crease_count = int(rng.integers(1, 4))

        paper, alpha = apply_damage(
            paper,
            damage_rng,
            crease_count=crease_count,
            torn_edges=True,
            edge_strength=EDGE_DARKEN[aging],
        )
    else:
        alpha = np.full(
            (height, width), 255, dtype=np.uint8
        )

    paper = np.clip(paper, 0, 255).astype(np.uint8)

    image = Image.fromarray(paper, "RGB")
    alpha_image = Image.fromarray(alpha, "L")

    metadata = {
        "material": material,
        "aging": aging,
        "seed": seed,
        "material_seed": material_seed,
        "aging_seed": aging_seed,
        "damage_seed": damage_seed,
        "width": width,
        "height": height,
        "damage": damage,
        "crease_count": crease_count,
    }

    result = (image, alpha_image, metadata)

    if return_material_maps:
        maps = {
            "absorbency": surface.absorbency,
            "fiber_map": surface.fiber_map,
            "roughness": surface.roughness,
        }
        return (*result, maps)

    return result


def save_page(image, alpha, filename, output_dir="output"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f"{filename}.png"
    mask_path = output_dir / f"{filename}_mask.png"
    rgba_path = output_dir / f"{filename}_rgba.png"

    image.save(image_path)
    alpha.save(mask_path)

    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(rgba_path)

    return image_path, mask_path, rgba_path


if __name__ == "__main__":
    import json
    from PIL import ImageDraw

    out_dir = Path("output") / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = 1200, 1600
    TW, TH = 300, 400
    BG = (128, 128, 128)            # neutral grey, only for the contact sheet
    seeds = [101, 202]              # two pages per (material, aging) cell

    columns = [(lvl, s) for lvl in AGING_LEVELS for s in seeds]
    sheet = Image.new(
        "RGB",
        (len(columns) * (TW + 10) + 10, len(MATERIALS) * (TH + 30) + 10),
        (30, 30, 30),
    )
    draw = ImageDraw.Draw(sheet)

    for row, material in enumerate(MATERIALS):
        for col, (aging, seed) in enumerate(columns):
            image, alpha, metadata = generate_manuscript_background(
                width=W, height=H, material=material,
                aging=aging, seed=seed, damage=True,
            )

            name = f"{material}_{aging}_s{seed}"
            save_page(image, alpha, name, out_dir)
            (out_dir / f"{name}.json").write_text(
                json.dumps(metadata, indent=2)
            )

            comp = Image.new("RGB", (W, H), BG)
            comp.paste(image.convert("RGB"), (0, 0), alpha)

            x = 10 + col * (TW + 10)
            y = 10 + row * (TH + 30)
            draw.text((x, y), f"{material} {aging} s{seed}",
                      fill=(255, 255, 255))
            sheet.paste(comp.resize((TW, TH), Image.LANCZOS), (x, y + 16))

            print("Generated:", name, metadata["crease_count"], "creases")

    sheet.save(out_dir / "batch_sheet.png")
    print("Saved to:", out_dir)