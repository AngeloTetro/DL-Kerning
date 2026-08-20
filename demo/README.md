# DLKern — Demo

Interactive desktop application to predict and visualize letter spacing (kerning) in fonts.

## What it does

- Pick a font (autocomplete across the 7,263 available fonts)
- Choose two letters (Pair) or a whole word (Word)
- Select which models to use: Ground Truth, Pairwise, Setwise, Ensemble, Set->Pair, Cross-Attention
- See each model in a separate cell, with its predicted spacing in pixels
- Compare all models in a single overlay using distinct colors

## How to run it

From the project root:

```bash
uv run python demo/gui.py
```

Requires Python 3.13 and the project dependencies (`uv sync`).

## Project files

- `gui.py` — tkinter graphical interface
- `predict.py` — inference functions (model loading and prediction)
- `icon.png` — application icon
