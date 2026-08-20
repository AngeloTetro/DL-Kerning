"""Core inference functions for DLKern. Loads trained models and predicts pair spacing.

Models:
  - Pairwise DNN (NB04)       — 3-layer MLP, 1128 -> 256 -> 256 -> 1
  - Setwise Transformer (NB05) — 1-layer, 2-head Transformer on N² token pairs
  - Set->Pair Fusion (NB07)    — Pairwise + global context from frozen setwise
  - Cross-Attention (NB08)     — Pairwise queries setwise hidden states via cross-attention
  - Ensemble                    — (pairwise + setwise) / 2
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (must match training) ──
LATIN_52 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
FEATURE_DIM = 512
NUM_CLASSES = 52
INPUT_DIM = FEATURE_DIM * 2 + NUM_CLASSES * 2  # 1128

# Setwise constants
D_MODEL = 32
NHEAD = 2
NUM_LAYERS = 1

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Models ──

class PairwiseDNN(nn.Module):
    """3-layer fully-connected DNN (NB04)."""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SetwiseModel(nn.Module):
    """Transformer-based set-wise model (NB05)."""
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(FEATURE_DIM * 2, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, batch_first=True,
            dim_feedforward=D_MODEL * 4, activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.output_proj = nn.Linear(D_MODEL, 1)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        x = self.output_proj(x).squeeze(-1)
        return x

    def get_hidden(self, x, mask=None):
        """Forward pass stopping after LayerNorm. Returns (B, N², 32) hidden states."""
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x


class FusionPairwiseDNN(nn.Module):
    """Pairwise DNN with global setwise context (NB07): 1160 -> 256 -> 256 -> 1."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM + D_MODEL, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PairwiseEncoderCA(nn.Module):
    """Encodes a single pair into a 64-d query vector (NB08 — matches checkpoint)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionModel(nn.Module):
    """Cross-attention: pairwise query attends over setwise hidden states (NB08)."""
    def __init__(self):
        super().__init__()
        self.encoder = PairwiseEncoderCA()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(D_MODEL, 64)
        self.v_proj = nn.Linear(D_MODEL, 64)
        self.head = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1))
        self.scale = 64 ** -0.5

    def forward(self, x, sw_hidden, mask):
        query = self.encoder(x)                     # (B, 64)
        Q = self.q_proj(query).unsqueeze(1)          # (B, 1, 64)
        K = self.k_proj(sw_hidden)                   # (B, N², 64)
        V = self.v_proj(sw_hidden)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, V).squeeze(1)      # (B, 64)
        fused = torch.cat([query, context], dim=1)    # (B, 128)
        return self.head(fused).squeeze(-1)


# ── Project root discovery ──

def _find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        if (p / "pyproject.toml").exists():
            return p
    return cwd


PROJECT_ROOT = _find_project_root()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = DATA_PROCESSED / "features"
GLYPH_DIR = DATA_PROCESSED / "glyphs"
PAIRWISE_MODEL = DATA_PROCESSED / "models" / "pairwise" / "best_model.pt"
SETWISE_MODEL = DATA_PROCESSED / "models" / "setwise" / "best_model.pt"
FUSION_MODEL = DATA_PROCESSED / "models" / "set_pair_fusion" / "best_model.pt"
CROSS_ATTN_MODEL = DATA_PROCESSED / "models" / "cross_attention" / "best_model.pt"

# ── Cache ──
_pairwise_model = None
_setwise_model = None
_fusion_model = None
_cross_attn_model = None
_font_features = {}       # stem -> features ndarray
_font_glyph_names = {}    # stem -> list of glyph name strings
_font_pair_dists = {}     # stem -> (pair_left, pair_right, pair_distances) for ground truth
_setwise_hidden_cache = {}  # stem -> (N², 32) hidden states (for Fusion & Cross-Attn)
_setwise_pred_cache = {}    # stem -> (N²,) predictions


def load_pairwise_model():
    """Load the pairwise model (lazy, cached)."""
    global _pairwise_model
    if _pairwise_model is None:
        if not PAIRWISE_MODEL.exists():
            raise FileNotFoundError(f"Pairwise model not found at {PAIRWISE_MODEL}")
        _pairwise_model = PairwiseDNN()
        _pairwise_model.load_state_dict(torch.load(PAIRWISE_MODEL, weights_only=True, map_location=DEVICE))
        _pairwise_model.to(DEVICE)
        _pairwise_model.eval()
    return _pairwise_model


def load_setwise_model():
    """Load the setwise model (lazy, cached)."""
    global _setwise_model
    if _setwise_model is None:
        if not SETWISE_MODEL.exists():
            raise FileNotFoundError(f"Setwise model not found at {SETWISE_MODEL}")
        _setwise_model = SetwiseModel()
        _setwise_model.load_state_dict(torch.load(SETWISE_MODEL, weights_only=True, map_location=DEVICE))
        _setwise_model.to(DEVICE)
        _setwise_model.eval()
    return _setwise_model


def load_fusion_model():
    """Load the Set->Pair Fusion model (NB07, lazy, cached)."""
    global _fusion_model
    if _fusion_model is None:
        if not FUSION_MODEL.exists():
            raise FileNotFoundError(f"Fusion model not found at {FUSION_MODEL}")
        _fusion_model = FusionPairwiseDNN()
        _fusion_model.load_state_dict(torch.load(FUSION_MODEL, weights_only=True, map_location=DEVICE))
        _fusion_model.to(DEVICE)
        _fusion_model.eval()
    return _fusion_model


def load_cross_attn_model():
    """Load the Cross-Attention model (NB08, lazy, cached)."""
    global _cross_attn_model
    if _cross_attn_model is None:
        if not CROSS_ATTN_MODEL.exists():
            raise FileNotFoundError(f"Cross-Attention model not found at {CROSS_ATTN_MODEL}")
        _cross_attn_model = CrossAttentionModel()
        _cross_attn_model.load_state_dict(torch.load(CROSS_ATTN_MODEL, weights_only=True, map_location=DEVICE))
        _cross_attn_model.to(DEVICE)
        _cross_attn_model.eval()
    return _cross_attn_model


def load_font(stem: str):
    """Load features, glyph names, and ground truth for a font (lazy, cached)."""
    if stem in _font_features:
        return _font_features[stem], _font_glyph_names[stem]

    feat_path = FEATURE_DIR / f"{stem}.npz"
    glyph_path = GLYPH_DIR / f"{stem}.npz"

    if not feat_path.exists():
        raise FileNotFoundError(f"Font '{stem}' not found (expected data/processed/features/{stem}.npz)")
    if not glyph_path.exists():
        raise FileNotFoundError(f"Font '{stem}' not found (expected data/processed/glyphs/{stem}.npz)")

    fz = np.load(feat_path)
    gz = np.load(glyph_path)

    features = fz["features"]  # (N, 512)
    glyph_names = list(gz["glyph_names"])

    # Store ground truth data
    _font_pair_dists[stem] = {
        "pair_left": gz["pair_left"],
        "pair_right": gz["pair_right"],
        "pair_distances": gz["pair_distances"],
        "glyph_indices": gz["glyph_indices"],
        "glyph_images": gz["glyph_images"],
        "centroids_canvas": gz["centroids_canvas"],
        "baseline_y": gz["baseline_y"],
    }

    _font_features[stem] = features
    _font_glyph_names[stem] = glyph_names
    return features, glyph_names


def _build_pairwise_input(features, glyph_names, letter_a: str, letter_b: str):
    """Build the 1128-d input tensor for a given pair."""
    if letter_a not in glyph_names:
        raise ValueError(f"Letter '{letter_a}' not found in font glyphs ({glyph_names})")
    if letter_b not in glyph_names:
        raise ValueError(f"Letter '{letter_b}' not found in font glyphs ({glyph_names})")

    i_a = glyph_names.index(letter_a)
    i_b = glyph_names.index(letter_b)

    feat_a = torch.from_numpy(features[i_a]).float()
    feat_b = torch.from_numpy(features[i_b]).float()

    ci_a = LATIN_52.index(letter_a)
    ci_b = LATIN_52.index(letter_b)

    oh_a = torch.zeros(NUM_CLASSES)
    oh_b = torch.zeros(NUM_CLASSES)
    oh_a[ci_a] = 1.0
    oh_b[ci_b] = 1.0

    x = torch.cat([feat_a, feat_b, oh_a, oh_b])  # (1128,)
    return x


def predict_pairwise(stem: str, letter_a: str, letter_b: str) -> float:
    """Predict the pair distance (in pixels) with the pairwise model.

    Args:
        stem: Font stem (e.g., 'Roboto_regular')
        letter_a: First letter (e.g., 'A')
        letter_b: Second letter (e.g., 'V')

    Returns:
        Predicted centroid-to-centroid distance in pixels.
    """
    model = load_pairwise_model()
    features, glyph_names = load_font(stem)
    x = _build_pairwise_input(features, glyph_names, letter_a, letter_b)
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x).item()
    return pred


def predict_setwise(stem: str, letter_a: str, letter_b: str) -> float:
    """Predict the pair distance (in pixels) with the setwise model.

    The setwise model predicts all N² pairs at once. We run the full
    forward pass and index the specific pair.

    Args:
        stem: Font stem (e.g., 'Roboto_regular')
        letter_a: First letter (e.g., 'A')
        letter_b: Second letter (e.g., 'V')

    Returns:
        Predicted centroid-to-centroid distance in pixels.
    """
    model = load_setwise_model()
    features, glyph_names = load_font(stem)

    N = len(glyph_names)
    i_a = glyph_names.index(letter_a)
    i_b = glyph_names.index(letter_b)

    # Build all N² pair tokens
    features_t = torch.from_numpy(features).float().to(DEVICE)
    tokens_list = []
    for i in range(N):
        fi = features_t[i]
        for j in range(N):
            tokens_list.append(torch.cat([fi, features_t[j]]))
    tokens = torch.stack(tokens_list)  # (N², 1024)
    tokens = tokens.unsqueeze(0)  # (1, N², 1024)

    # No padding mask (single font, all valid)
    mask = None

    with torch.no_grad():
        out = model(tokens, mask)  # (1, N²)

    pair_idx = i_a * N + i_b
    return out[0, pair_idx].item()


def _precompute_setwise_cache(stem: str):
    """Run setwise forward once and cache predictions + hidden states."""
    if stem in _setwise_pred_cache:
        return
    model = load_setwise_model()
    features, glyph_names = load_font(stem)
    N = len(glyph_names)
    features_t = torch.from_numpy(features).float().to(DEVICE)
    tokens_list = []
    for i in range(N):
        fi = features_t[i]
        for j in range(N):
            tokens_list.append(torch.cat([fi, features_t[j]]))
    tokens = torch.stack(tokens_list).unsqueeze(0)  # (1, N², 1024)
    with torch.no_grad():
        pred = model(tokens, mask=None).squeeze(0).cpu().numpy()
        hidden = model.get_hidden(tokens, mask=None).squeeze(0).cpu().numpy()
    _setwise_pred_cache[stem] = pred.astype(np.float32)
    _setwise_hidden_cache[stem] = hidden.astype(np.float32)


def predict_fusion(stem: str, letter_a: str, letter_b: str) -> float:
    """Predict with Set->Pair Fusion (NB07): pairwise input + global setwise context.

    Uses average-pooled setwise hidden states (32-d) as additional input.
    """
    model = load_fusion_model()
    features, glyph_names = load_font(stem)
    _precompute_setwise_cache(stem)

    N = len(glyph_names)
    i_a = glyph_names.index(letter_a)
    i_b = glyph_names.index(letter_b)

    feat_a = torch.from_numpy(features[i_a]).float().to(DEVICE)
    feat_b = torch.from_numpy(features[i_b]).float().to(DEVICE)
    ci_a = LATIN_52.index(letter_a)
    ci_b = LATIN_52.index(letter_b)
    oh_a = torch.zeros(NUM_CLASSES, device=DEVICE)
    oh_b = torch.zeros(NUM_CLASSES, device=DEVICE)
    oh_a[ci_a] = 1.0
    oh_b[ci_b] = 1.0

    x_pw = torch.cat([feat_a, feat_b, oh_a, oh_b])  # (1128,)
    ctx = torch.from_numpy(_setwise_hidden_cache[stem].mean(axis=0)).float().to(DEVICE)  # (32,)
    x = torch.cat([x_pw, ctx]).unsqueeze(0)  # (1, 1160)

    with torch.no_grad():
        pred = model(x).item()
    return pred


def predict_cross_attn(stem: str, letter_a: str, letter_b: str) -> float:
    """Predict with Cross-Attention model (NB08): pairwise encoder queries setwise hidden states."""
    model = load_cross_attn_model()
    features, glyph_names = load_font(stem)
    _precompute_setwise_cache(stem)

    N = len(glyph_names)
    i_a = glyph_names.index(letter_a)
    i_b = glyph_names.index(letter_b)

    feat_a = torch.from_numpy(features[i_a]).float().to(DEVICE)
    feat_b = torch.from_numpy(features[i_b]).float().to(DEVICE)
    ci_a = LATIN_52.index(letter_a)
    ci_b = LATIN_52.index(letter_b)
    oh_a = torch.zeros(NUM_CLASSES, device=DEVICE)
    oh_b = torch.zeros(NUM_CLASSES, device=DEVICE)
    oh_a[ci_a] = 1.0
    oh_b[ci_b] = 1.0

    x_pw = torch.cat([feat_a, feat_b, oh_a, oh_b]).unsqueeze(0)  # (1, 1128)
    sw_h = torch.from_numpy(_setwise_hidden_cache[stem]).unsqueeze(0).to(DEVICE)  # (1, N², 32)
    mask = torch.zeros(1, N * N, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        pred = model(x_pw, sw_h, mask).item()
    return pred


def predict_ensemble(stem: str, letter_a: str, letter_b: str) -> float:
    """Simple average ensemble: (pairwise + setwise) / 2."""
    pw = predict_pairwise(stem, letter_a, letter_b)
    sw = predict_setwise(stem, letter_a, letter_b)
    return (pw + sw) / 2.0


def get_ground_truth(stem: str, letter_a: str, letter_b: str) -> float | None:
    """Get the ground truth pair distance for a kerned pair, or None if not kerned.

    Returns None if the pair has no explicit kerning in the original font.
    """
    _, glyph_names = load_font(stem)
    pair_data = _font_pair_dists[stem]

    # Build LATIN_52 index -> local index mapping
    glyph_indices = pair_data["glyph_indices"]
    latin_to_local = {int(idx): i for i, idx in enumerate(glyph_indices)}

    ci_a = LATIN_52.index(letter_a)
    ci_b = LATIN_52.index(letter_b)

    # Search in kerned pairs
    left = pair_data["pair_left"]
    right = pair_data["pair_right"]
    dists = pair_data["pair_distances"]

    for k in range(len(left)):
        if int(left[k]) == ci_a and int(right[k]) == ci_b:
            return float(dists[k])

    return None


def list_available_fonts():
    """List all font stems that have features available."""
    stems = sorted(p.stem for p in FEATURE_DIR.glob("*.npz"))
    return stems


def list_font_glyphs(stem: str):
    """List all glyphs available in a given font."""
    _, glyph_names = load_font(stem)
    return glyph_names


__all__ = [
    "load_pairwise_model", "load_setwise_model", "load_fusion_model", "load_cross_attn_model",
    "predict_pairwise", "predict_setwise", "predict_fusion", "predict_cross_attn", "predict_ensemble",
    "get_ground_truth", "list_available_fonts", "list_font_glyphs",
    "LATIN_52",
]
