# Synthetic Manuscript Generator

An end-to-end Python pipeline that generates **realistic synthetic Indic manuscript folios** paired with structured ground-truth annotation files (`.md`), suitable for training OCR models.

Supports three Indic scripts: **Devanagari**, **Modi**, and **Sharada**.

---

## Sample Output

| Devanagari (paper) | Modi (parchment) | Sharada (paper) |
|---|---|---|
| _generated image_ | _generated image_ | _generated image_ |

---

## Dataset

The generated dataset is hosted on Hugging Face:

**[your-username/indic-manuscript-dataset](https://huggingface.co/datasets/your-username/indic-manuscript-dataset)**

Structure:
```
devanagari/
  train/        (85 images + .md files)
  validation/   (10 images + .md files)
  test/         (5 images + .md files)
modi/
  train/ validation/ test/
sharada/
  train/ validation/ test/
```

---

## Installation

```bash
git clone https://github.com/your-username/indic-manuscript-generator
cd indic-manuscript-generator

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Font Setup

Place font files in the `assets/fonts/` directory:

```
assets/fonts/
  devanagiri/   NIRMALA.TTC          (Windows: C:/Windows/Fonts/Nirmala.ttf)
  modi/         NotoSansModi-Regular.ttf
  sharada/      NotoSansSharada-Regular.ttf
```

Download Noto fonts: [fonts.google.com/noto](https://fonts.google.com/noto)

### Text Corpus Setup

Run once to populate `assets/text/` with classical Indic source text:

```bash
python setup_assets.py
```

---

## Usage

```bash
# Generate all 300 images (100 per script)
python generate.py

# Quick test — 3 images, Devanagari only
python generate.py --count 3 --scripts devanagari

# Generate + upload to Hugging Face
python generate.py --upload

# Custom config / output directory
python generate.py --config config.yaml --output output/my_run
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.yaml` | Path to YAML config file |
| `--scripts` | all three | Subset of scripts to generate |
| `--count` | 100 | Images per script |
| `--output` | `output/dataset` | Output directory |
| `--upload` | off | Upload to Hugging Face after generation |

---

## Configuration

All parameters are in [`config.yaml`](config.yaml):

```yaml
width:  1400
height: 1900
count_per_script: 100
splits:
  train: 85
  validation: 10
  test: 5
materials: [paper, parchment]
aging_levels: [light, moderate, heavy]
```

---

## Repository Structure

```
.
├── generate.py                  # Main CLI entry point
├── setup_assets.py              # One-time text corpus generator
├── config.yaml                  # All tunable parameters
├── requirements.txt
│
├── assets/
│   ├── fonts/
│   │   ├── devanagiri/          # Nirmala / Devanagari fonts
│   │   ├── modi/                # Noto Sans Modi
│   │   └── sharada/             # Noto Sans Sharada
│   └── text/
│       ├── devanagari.txt       # Classical Marathi source text
│       ├── modi.txt             # Modi script source text
│       └── sharada.txt          # Sharada script source text
│
├── src/
│   ├── material_texture.py      # Paper / parchment base texture generation
│   ├── aging_engine.py          # Yellowing, foxing, water damage, edge aging
│   ├── damage_engine.py         # Crease shadows + torn irregular edges
│   ├── background_generator.py  # Orchestrates background pipeline
│   ├── corpus.py                # Source text loading and paragraph generation
│   ├── translit.py              # Devanagari → Modi / Sharada Unicode conversion
│   ├── text_renderer.py         # Text → stroke mask with handwriting effects
│   ├── layout.py                # Page composition: text + margins + rubrication
│   └── ink_engine.py            # Ink compositing (iron-gall + vermilion)
│
└── output/
    └── dataset/
        ├── devanagari/
        │   ├── train/
        │   ├── validation/
        │   └── test/
        ├── modi/
        └── sharada/
```

---

## Pipeline Overview

```
config.yaml
    │
    ▼
generate.py
    │
    ├── background_generator ──► material_texture ──► aging_engine ──► damage_engine
    │       (aged paper RGB + alpha mask + material maps)
    │
    ├── corpus + translit
    │       (classical Indic text in target script Unicode)
    │
    ├── layout (compose)
    │       (stroke mask: main text + margin note + folio number + rubricated words)
    │
    ├── ink_engine (ink_page)
    │       (iron-gall ink + vermilion highlights composited onto background)
    │
    └── save Image_NNN.png + Image_NNN.md
```

---

## Annotation Format

Each image has a paired `.md` file:

```markdown
श्री गणेशाय नमः ।

ऐकोनि ऋषीचे वचन । राजा जाहला प्रसन्न । ...।३१।

तरीच सुख । पुढे झाले ऐका वृत्त । ...।३२।

हे पुन्हा पाहावे
।२७२।
```

The file contains:
- Full transcription of all rendered text (line by line, paragraph separated)
- Marginal annotation text
- Folio number

---

## Visual Features

| Feature | Implementation |
|---|---|
| Paper textures | Multi-scale Perlin-style noise + fiber simulation |
| Aging effects | Yellowing, foxing spots, water stains, edge darkening |
| Physical damage | Crease shadows, torn/irregular page boundaries |
| Handwriting variation | Per-word baseline wobble, slant, pressure drift, stroke warp |
| Ink simulation | Absorption-driven bleeding, fading on stained areas, density variation |
| Rubrication | Random verse markers + words highlighted in vermilion ink |
| Ruled lines | Shirorekha ruling (Devanagari) / baseline ruling (Modi) |
| Margin notes | Rotated side annotations |
| Folio numbers | Script-native numerals in page corners |
| Scripts | Devanagari, Modi, Sharada via Unicode + RAQM shaping |

---

## Requirements

```
Pillow>=10.0
numpy>=1.26
opencv-python>=4.9
PyYAML>=6.0
datasets>=3.0
huggingface_hub>=0.25
scikit-learn>=1.5
fonttools>=4.0
uharfbuzz>=0.40
```

> **Note:** RAQM must be available in your Pillow build for correct Indic text shaping.
> Run `python check_environment.py` to verify.

