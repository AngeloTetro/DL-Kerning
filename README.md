# Kern

Automatic kerning estimation for the 52 Latin letters (A–Z, a–z), replicating and extending
**"Learning to Kern"** (Nakatsuru & Uchida, 2024). The pipeline compares five architectures —
pairwise, setwise, ensemble, set→pair fusion, and cross-attention — plus explainability analysis
and an interactive demo.


## Results

| Model | Test MAE (px) | Architecture |
|---|---|---|
| **Ensemble (NB06)** | **3.8804** | (PW+SW)/2 |
| Pairwise (NB04) | 3.8949 | 3-layer DNN 1128→256→256→1 |
| Set→Pair Fusion (NB07) | 4.0446 | PW + frozen SW mean context |
| Cross-Attention (NB08) | 4.1320 | Pair query × SW hidden states |
| Setwise (NB05) | 4.6705 | 1-layer Transformer, d_model=32 |


## Repository layout

- `notebooks/` — the full pipeline, from font download to XAI analysis.
- `demo/` — DLKern interactive GUI (`gui.py`) and inference functions (`predict.py`).
- `data/processed/models/` — trained checkpoints.
- `data/processed/figures/` — extracted pipeline figures.
- `data/processed/xai/` — explainability figures and demo screenshots.

Raw fonts, rasterized glyphs, and extracted features are regenerable and intentionally not
committed; see the notebooks for how they are produced.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS
uv run jupyter lab
```

Copy `.env.example` to `.env` and set `GOOGLE_FONTS_API_KEY`.

### Regenerating data

Large and regenerable data files are intentionally **not** committed. After cloning,
run the following notebooks in order to restore them:

1. `notebooks/01_download.ipynb` → downloads the raw fonts to `data/raw/fonts/*.ttf`
   (requires `GOOGLE_FONTS_API_KEY`).
2. `notebooks/02_rasterize.ipynb` → rasterizes glyphs to `data/processed/glyphs/*.npz`.
3. `notebooks/03_resnet_features.ipynb` → trains the ResNet18 classifier and extracts
   the 512-d glyph features to `data/processed/features/*.npz`.

The trained model checkpoints, the small metadata CSVs, and the figures are already
committed, so notebooks 04–09 and the demo can run without re-training once the files
above are regenerated.

## Demo

```bash
uv run python demo/gui.py
```

The GUI supports font autocomplete, pair mode (`A+B`), word mode (`Kerning`), per-model
rendering, and overlay comparison.

## Pipeline

| Notebook | Description |
|---|---|
| `01_download.ipynb` | Download Google Fonts + extract GPOS/kern pairs |
| `02_rasterize.ipynb` | Rasterize all 52 glyphs per font, compute N² pair distances |
| `03_resnet_features.ipynb` | Train ResNet18 classifier, extract 512-d features |
| `04_pairwise_baseline.ipynb` | Pairwise DNN on all N² pairs (MAE 3.8949) |
| `05_setwise_baseline.ipynb` | Setwise Transformer (MAE 4.6705) |
| `06_ensemble.ipynb` | Ensemble PW+SW (MAE 3.8804) |
| `07_set_pair_fusion.ipynb` | Set→Pair fusion (MAE 4.0446) |
| `08_cross_attention.ipynb` | Cross-attention (MAE 4.1320) |
| `09_xai.ipynb` | SHAP, Grad-CAM, attention, qualitative analysis |

## Dataset

- 7,263 Google Fonts after filtering, 52 glyphs per font.
- 256×256 binary glyph images, baseline at y=160.
- Split by font family: 5,758 train / 719 val / 786 test.
- Total all-pairs targets: 19,639,152.
