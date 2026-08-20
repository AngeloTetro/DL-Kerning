#!/usr/bin/env python
"""Kerning Demo — Interactive letter-spacing prediction GUI."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np

from predict import (
    predict_pairwise, predict_setwise, predict_ensemble,
    predict_fusion, predict_cross_attn, get_ground_truth,
    list_available_fonts, load_font, _font_pair_dists,
    load_pairwise_model, load_setwise_model,
    load_fusion_model, load_cross_attn_model,
)

MODEL_LIST = [
    ("GT",  "Ground Truth",      (0.00, 0.00, 0.00)),
    ("PW",  "Pairwise (NB04)",   (0.20, 0.47, 0.94)),
    ("SW",  "Setwise  (NB05)",   (0.88, 0.20, 0.20)),
    ("ENS", "Ensemble (NB06)",   (0.94, 0.55, 0.05)),
    ("SP",  "Set->Pair (NB07)",  (0.05, 0.70, 0.65)),
    ("CA",  "CrossAttn (NB08)",  (0.60, 0.25, 0.90)),
]

MODEL_FUNCTIONS = {
    "PW": predict_pairwise, "SW": predict_setwise,
    "ENS": predict_ensemble, "SP": predict_fusion,
    "CA": predict_cross_attn,
}

def _h(rgb):
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

# ── Rendering ───────────────────────────────────────────────────────────────

def _trim(canvas, pad=14):
    gray = canvas.mean(axis=2) if canvas.ndim == 3 else canvas
    rows = np.any(gray < 0.98, axis=1)
    cols = np.any(gray < 0.98, axis=0)
    if not rows.any() or not cols.any():
        return canvas
    rmin = max(0, np.where(rows)[0][0] - pad)
    rmax = min(canvas.shape[0], np.where(rows)[0][-1] + pad + 1)
    cmin = max(0, np.where(cols)[0][0] - pad)
    cmax = min(canvas.shape[1], np.where(cols)[0][-1] + pad + 1)
    return canvas[rmin:rmax, cmin:cmax]

def render_pair(imgs, cents, iL, iR, lx, rx, lrgb, rrgb, gh=256):
    cL, cR = cents[iL], cents[iR]; pad = 15; H = gh + 30
    lx0 = int(lx - cL[0]); rx0 = int(rx - cR[0])
    mn = min(lx0, rx0, 0) - pad; mx = max(lx0 + gh, rx0 + gh) + pad
    sh = -mn; tw = mx + sh
    img = np.ones((H, tw, 3), dtype=np.float32)
    for x0, i, rgb in [(lx0, iL, lrgb), (rx0, iR, rrgb)]:
        c = x0 + sh; w = min(gh, tw - c); h = min(gh, H)
        if w > 0 and h > 0:
            m = imgs[i][:h, :w] > 0
            for ch in range(3): img[:h, c:c+w, ch][m] = rgb[ch]
    return _trim(img, pad=4)

def render_overlay(imgs, cents, iL, iR, dists, colors, order, gh=256):
    """Overlay: left black, GT right black, each model's right glyph full-filled solid color."""
    cL, cR = cents[iL], cents[iR]; pad = 15; H = gh + 30; lcx = gh // 2 + pad
    lx0 = int(lcx - cL[0])
    rx0s = [int(lcx + d - cR[0]) for d in dists.values()]
    mn = min(lx0, *rx0s, 0) - pad; mx = max(lx0 + gh, *(r + gh for r in rx0s)) + pad
    sh = -mn; tw = mx + sh
    img = np.ones((H, tw, 3), dtype=np.float32)
    # Left glyph solid black
    lc = lx0 + sh; w = min(gh, tw - lc); h = min(gh, H)
    if w > 0 and h > 0:
        m = imgs[iL][:h, :w] > 0; img[:h, lc:lc+w, :][m] = 0
    # GT right glyph solid black
    if "GT" in dists:
        rc = int(lcx + dists["GT"] - cR[0]) + sh; w = min(gh, tw - rc); h = min(gh, H)
        if w > 0 and h > 0:
            m = imgs[iR][:h, :w] > 0; img[:h, rc:rc+w, :][m] = 0
    # Models: full-filled solid color (drawn over GT, order preserved)
    for label in order:
        if label == "GT" or label not in dists: continue
        rc = int(lcx + dists[label] - cR[0]) + sh; w = min(gh, tw - rc); h = min(gh, H)
        if w > 0 and h > 0:
            m = imgs[iR][:h, :w] > 0
            r, g, b = colors[label]
            for ch, val in enumerate([r, g, b]):
                img[:h, rc:rc+w, ch][m] = val
    return _trim(img, pad=4)

def render_word_single(imgs, cents, idxs, spacings, rgb, gh=256):
    n = len(idxs); pad = 15; H = gh + 30; lcx = gh // 2 + pad
    cx = [lcx]
    for p in range(n - 1): cx.append(cx[-1] + spacings[p])
    alx = [int(cx[i] - cents[idxs[i]][0]) for i in range(n)]
    arx = [x + gh for x in alx]; mn = min(alx) - pad; mx = max(arx) + pad
    sh = -mn; tw = mx + sh; img = np.ones((H, tw, 3), dtype=np.float32)
    lc = int(cx[0] - cents[idxs[0]][0]) + sh; w = min(gh, tw - lc); h = min(gh, H)
    if w > 0 and h > 0: m = imgs[idxs[0]][:h, :w] > 0; img[:h, lc:lc+w, :][m] = 0
    for i in range(1, n):
        rc = int(cx[i] - cents[idxs[i]][0]) + sh; w = min(gh, tw - rc); h = min(gh, H)
        if w > 0 and h > 0:
            m = imgs[idxs[i]][:h, :w] > 0
            for ch in range(3): img[:h, rc:rc+w, ch][m] = rgb[ch]
    return _trim(img, pad=4)

def render_word_overlay(imgs, cents, idxs, gt_sp, msp, colors, order, gh=256):
    """Word overlay: GT solid black, each model's letters full-filled solid color."""
    n = len(idxs); pad = 15; H = gh + 30; lcx = gh // 2 + pad
    gt_cx = [lcx]
    for p in range(n - 1): s = gt_sp[p] if gt_sp[p] is not None else 120; gt_cx.append(gt_cx[-1] + s)
    mcx = {}
    for k in msp:
        mc = [lcx]
        for p in range(n - 1): mc.append(mc[-1] + msp[k][p])
        mcx[k] = mc
    alx = [int(gt_cx[i] - cents[idxs[i]][0]) for i in range(n)]
    for mc in mcx.values():
        for i in range(n): alx.append(int(mc[i] - cents[idxs[i]][0]))
    arx = [x + gh for x in alx]; mn = min(alx) - pad; mx = max(arx) + pad
    sh = -mn; tw = mx + sh; img = np.ones((H, tw, 3), dtype=np.float32)
    # GT letters solid black
    for i in range(n):
        lc = int(gt_cx[i] - cents[idxs[i]][0]) + sh; w = min(gh, tw - lc); h = min(gh, H)
        if w > 0 and h > 0: m = imgs[idxs[i]][:h, :w] > 0; img[:h, lc:lc+w, :][m] = 0
    # Models: full-filled solid color
    for k in order:
        if k not in mcx: continue
        mc = mcx[k]; rgb = colors[k]
        for i in range(1, n):
            rc = int(mc[i] - cents[idxs[i]][0]) + sh; w = min(gh, tw - rc); h = min(gh, H)
            if w > 0 and h > 0:
                m = imgs[idxs[i]][:h, :w] > 0
                for ch, val in enumerate(rgb):
                    img[:h, rc:rc+w, ch][m] = val
    return _trim(img, pad=4)


# ── Autocomplete ────────────────────────────────────────────────────────────

class AC(ttk.Entry):
    def __init__(self, p, w=30, **kw):
        super().__init__(p, width=w, **kw)
        self.ch = []; self.lb = None; self.tv = self.winfo_toplevel()
        self.bind("<KeyRelease>", self._ak)
        self.bind("<FocusOut>", lambda e: self.after(150, self._h))
        self.bind("<Down>", self._f); self.bind("<Return>", self._s); self.bind("<Escape>", self._h)

    def set_choices(self, c): self.ch = sorted(c)

    def _ak(self, e):
        if e.keysym in ("Down", "Up", "Return", "Escape", "Tab"): return
        self.after_idle(self._up)

    def _up(self):
        t = self.get().lower()
        if not t: self._h(); return
        m = [c for c in self.ch if c.lower().startswith(t)][:12]
        if not m: self._h(); return
        if self.lb is None:
            self.lb = tk.Listbox(self.tv, height=min(len(m), 10), exportselection=False,
                relief=tk.FLAT, bd=1, bg="white", fg="#202124", font=("Segoe UI", 10))
            self.lb.bind("<ButtonRelease-1>", self._oc)
        self.lb.delete(0, tk.END)
        for x in m: self.lb.insert(tk.END, x)
        self.lb.config(height=min(len(m), 10))
        x = self.winfo_rootx() - self.tv.winfo_rootx()
        y = self.winfo_rooty() - self.tv.winfo_rooty() + self.winfo_height()
        self.lb.place(x=x, y=y, width=self.winfo_width()); self.lb.lift()

    def _f(self, e):
        if self.lb and self.lb.winfo_ismapped():
            self.lb.focus_set(); self.lb.selection_set(0); return "break"

    def _s(self, e=None):
        if self.lb and self.lb.winfo_ismapped():
            s = self.lb.curselection()
            if s: self.delete(0, tk.END); self.insert(0, self.lb.get(s[0])); self._h(); self.event_generate("<<AC>>")
            return "break"

    def _oc(self, e): self._s()
    def _h(self, e=None):
        if self.lb: self.lb.place_forget()


# ── Main app ────────────────────────────────────────────────────────────────

class App:
    BG = "#f5f5f7"      # Apple-like light gray
    C = "#ffffff"
    T = "#1d1d1f"
    M = "#86868b"
    ACCENT = "#0071e3"
    ACCENT_HOVER = "#0077ed"

    def __init__(self, r):
        self.r = r; r.title("DLKern"); r.minsize(960, 640)
        r.configure(bg=self.BG)
        s = ttk.Style(); s.theme_use("clam")

        # Base style
        s.configure(".", background=self.BG, foreground=self.T, font=("Segoe UI", 10))

        # Rounded, flat scrollbar (Apple-like)
        s.configure("Vertical.TScrollbar", background="#d1d1d6", troughcolor="#f5f5f7",
                    bordercolor="#f5f5f7", arrowcolor="#86868b", relief="flat",
                    gripcount=0, width=10)
        s.map("Vertical.TScrollbar", background=[("active", "#b0b0b6")])

        # Radio / check indicators
        s.configure("TRadiobutton", background=self.C, foreground=self.T, font=("Segoe UI", 10))
        s.map("TRadiobutton", background=[("active", self.C)],
              indicatorcolor=[("selected", self.ACCENT), ("!selected", "#c7c7cc")])
        s.configure("TCheckbutton", background=self.C, foreground=self.T, font=("Segoe UI", 10))
        s.map("TCheckbutton", background=[("active", self.C)],
              indicatorcolor=[("selected", self.ACCENT), ("!selected", "#c7c7cc")])

        # Entry styling
        s.configure("TEntry", fieldbackground="#ffffff", bordercolor="#d1d1d6",
                    lightcolor="#d1d1d6", darkcolor="#d1d1d6", padding=3)

        # Size to fit the usable screen area (accounting for taskbar/title bar)
        r.update_idletasks()
        sw = r.winfo_screenwidth()
        work_h = self._work_area_height()
        w = min(1280, sw - 40)
        h = min(860, work_h - 40)
        x = (sw - w) // 2; y = 10
        r.geometry(f"{w}x{h}+{x}+{y}")

        self._set_icon()

        self.cf = None; self.gn = []; self.rw = []; self.er = []
        load_pairwise_model(); load_setwise_model(); load_fusion_model(); load_cross_attn_model()
        self._ui()

    def _resource_path(self, filename):
        """Return the path to a bundled resource, works both frozen and unfrozen."""
        base = getattr(sys, "_MEIPASS", Path(__file__).parent)
        return Path(base) / filename

    def _set_icon(self):
        """Load the app icon from icon.png, fallback to programmatic icon."""
        try:
            icon_path = self._resource_path("icon.png")
            if icon_path.exists():
                from PIL import Image, ImageTk
                img = Image.open(icon_path).resize((64, 64), Image.LANCZOS)
                icon = ImageTk.PhotoImage(img)
                self.r.iconphoto(True, icon)
                self._icon_ref = icon  # keep reference
                return
        except Exception:
            pass
        try:
            icon = tk.PhotoImage(width=64, height=64)
            icon.put("#0071e3", to=(0, 0, 64, 64))
            for y in range(18, 46):
                icon.put("#ffffff", to=(18, y, 22, y + 1))
                icon.put("#ffffff", to=(30, y, 34, y + 1))
            icon.put("#ffffff", to=(22, 18, 42, 22))
            icon.put("#ffffff", to=(26, 32, 38, 36))
            self.r.iconphoto(True, icon)
            self._icon_ref = icon
        except Exception:
            pass

    def _work_area_height(self):
        """Return the usable height of the working area (screen minus taskbar)."""
        try:
            import ctypes
            from ctypes import wintypes
            rect = wintypes.RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)  # SPI_GETWORKAREA
            return rect.bottom - rect.top
        except Exception:
            return self.r.winfo_screenheight() - 80

    def _ui(self):
        h = ttk.Frame(self.r, padding=(20, 14, 20, 4)); h.pack(fill=tk.X)
        # Header with small icon + title
        hrow = tk.Frame(h, bg=self.BG); hrow.pack(anchor=tk.W)
        self._header_icon = tk.PhotoImage()
        try:
            from PIL import Image, ImageTk
            icon_path = Path(__file__).parent / "icon.png"
            if icon_path.exists():
                img = Image.open(icon_path).resize((28, 28), Image.LANCZOS)
                self._header_icon = ImageTk.PhotoImage(img)
        except Exception:
            self._header_icon = None
        if self._header_icon is not None:
            tk.Label(hrow, image=self._header_icon, bg=self.BG).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(hrow, text="DLKern", font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)

        c = tk.Frame(self.r, bg=self.C, bd=0, highlightthickness=0, padx=16, pady=10)
        c.pack(fill=tk.X, padx=20, pady=(4, 0))

        # Row 1: Font + status (status label on the right of the font box)
        fr = tk.Frame(c, bg=self.C); fr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(fr, text="Font", font=("Segoe UI", 9, "bold"), bg=self.C).pack(side=tk.LEFT)
        self.fe = AC(fr, 32); self.fe.set_choices(list_available_fonts())
        self.fe.pack(side=tk.LEFT, padx=(8, 0))
        self.fe.bind("<<AC>>", lambda e: self._load_font())
        self.fe.bind("<Return>", lambda e: self._load_font())
        self.st = tk.Label(fr, text="", bg=self.C, fg=self.M, font=("Segoe UI", 8, "italic"))
        self.st.pack(side=tk.LEFT, padx=(10, 0))

        # Row 2: Mode + input
        row2 = tk.Frame(c, bg=self.C); row2.pack(fill=tk.X, pady=(0, 6))
        self.md = tk.StringVar(value="pair")
        tk.Radiobutton(row2, text="Pair", variable=self.md, value="pair", command=self._sw,
                       bg=self.C, font=("Segoe UI", 9),
                       activebackground=self.C, selectcolor=self.C).pack(side=tk.LEFT, padx=(0, 4))
        tk.Radiobutton(row2, text="Word", variable=self.md, value="word", command=self._sw,
                       bg=self.C, font=("Segoe UI", 9),
                       activebackground=self.C, selectcolor=self.C).pack(side=tk.LEFT, padx=(0, 16))

        self.pf = tk.Frame(row2, bg=self.C); self.pf.pack(side=tk.LEFT)
        tk.Label(self.pf, text="A", font=("Segoe UI", 9, "bold"), bg=self.C).pack(side=tk.LEFT)
        self.ea = AC(self.pf, 4); self.ea.pack(side=tk.LEFT, padx=(4, 2))
        tk.Label(self.pf, text="+", font=("Segoe UI", 11), bg=self.C).pack(side=tk.LEFT, padx=4)
        self.eb = AC(self.pf, 4); self.eb.pack(side=tk.LEFT, padx=2)

        self.wf = tk.Frame(row2, bg=self.C)
        tk.Label(self.wf, text="Text", font=("Segoe UI", 9, "bold"), bg=self.C).pack(side=tk.LEFT)
        self.we = tk.Entry(self.wf, width=24, font=("Segoe UI", 11), relief=tk.FLAT, bd=1)
        self.we.pack(side=tk.LEFT, padx=4)

        # Row 3: Models
        modr = tk.Frame(c, bg=self.C); modr.pack(fill=tk.X, pady=(0, 8))
        tk.Label(modr, text="Models", font=("Segoe UI", 9, "bold"), bg=self.C).pack(side=tk.LEFT)
        self.gt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(modr, text="GT", variable=self.gt_var, font=("Segoe UI", 9),
                       bg=self.C, activebackground=self.C, selectcolor=self.C, fg="#000").pack(side=tk.LEFT, padx=3)
        self.mv = {}
        for key, label, rgb in MODEL_LIST:
            if key == "GT": continue
            v = tk.BooleanVar(value=(key in ("PW", "SW")))
            cb = tk.Checkbutton(modr, text=label, variable=v, font=("Segoe UI", 9),
                                bg=self.C, activebackground=self.C, selectcolor=self.C, fg=_h(rgb))
            cb.pack(side=tk.LEFT, padx=3); self.mv[key] = v

        # Row 4: Predict button (below all sections)
        self.predict_btn = tk.Button(c, text="PREDICT", font=("Segoe UI", 11, "bold"),
                                     bg=self.ACCENT, fg="white", activebackground=self.ACCENT_HOVER,
                                     activeforeground="white", relief=tk.FLAT, padx=28, pady=6,
                                     cursor="hand2", command=self._predict, bd=0)
        self.predict_btn.pack(anchor=tk.W, pady=(0, 4))
        self.predict_btn.bind("<Enter>", lambda e: self.predict_btn.config(bg=self.ACCENT_HOVER))
        self.predict_btn.bind("<Leave>", lambda e: self.predict_btn.config(bg=self.ACCENT))

        self.efr = tk.Frame(c, bg=self.C); self.efr.pack(fill=tk.X)

        rc = ttk.Frame(self.r); rc.pack(fill=tk.BOTH, expand=True, padx=20, pady=(4, 12))
        self.cv = tk.Canvas(rc, bd=0, highlightthickness=0, bg=self.BG)
        self.sb = ttk.Scrollbar(rc, orient=tk.VERTICAL, command=self.cv.yview, style="Vertical.TScrollbar")
        self.inn = tk.Frame(self.cv, bg=self.BG)
        self.inn.bind("<Configure>", self._on_inner_configure)
        self.cv.create_window((0, 0), window=self.inn, anchor=tk.NW, tags=("w",))
        self.cv.configure(yscrollcommand=self.sb.set)
        self.cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cv.bind("<Configure>", self._on_canvas_configure)
        self.cv.bind("<Enter>", lambda e: self.cv.bind_all("<MouseWheel>",
            lambda ev: self.cv.yview_scroll(int(-ev.delta / 120), "units")))
        self.cv.bind("<Leave>", lambda e: self.cv.unbind_all("<MouseWheel>"))

    def _on_inner_configure(self, e):
        self.cv.configure(scrollregion=self.cv.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, e):
        self.cv.itemconfig("w", width=e.width)
        self._toggle_scrollbar()

    def _toggle_scrollbar(self):
        # Only show scrollbar when content is taller than the visible canvas
        if self.inn.winfo_reqheight() > self.cv.winfo_height():
            self.sb.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            self.sb.pack_forget()

    def _sw(self):
        self.pf.pack_forget(); self.wf.pack_forget()
        if self.md.get() == "pair":
            self.pf.pack(side=tk.LEFT)
        else:
            self.wf.pack(side=tk.LEFT)

    def _err(self, msg):
        l = tk.Label(self.efr, text="  ⚠  " + msg, fg="#d93025", font=("Segoe UI", 9), bg=self.C, anchor=tk.W)
        l.pack(fill=tk.X); self.er.append(l)

    def _clear_err(self):
        for w in self.er: w.destroy()
        self.er.clear()

    def _load_font(self):
        s = self.fe.get().strip()
        if not s: self.st.config(text="Type a font name to search", fg="#d93025"); return
        try:
            _, gn = load_font(s); self.cf = s; self.gn = gn
            self.ea.set_choices(gn); self.eb.set_choices(gn)
            self._clear_err(); self.st.config(text=f"✓  {s}  ({len(gn)} glyphs)", fg="#1e8e3e")
        except FileNotFoundError: self.cf = None; self._err(f"Font '{s}' not found")
        except Exception as e: self.cf = None; self._err(str(e))

    def _active(self):
        out = [(k, l, rgb) for k, l, rgb in MODEL_LIST if k == "GT" or self.mv[k].get()]
        if not self.gt_var.get(): out = [x for x in out if x[0] != "GT"]
        return out

    def _clr(self):
        for w in self.rw: w.destroy()
        self.rw.clear()

    def _predict(self):
        self._clr(); self._clear_err()
        stem = self.fe.get().strip()
        if not stem or not self.cf: self._load_font(); return
        active = self._active()
        if not active: self._err("Select at least one model"); return
        try:
            _, gn = load_font(stem); pd = _font_pair_dists[stem]
        except KeyError: self._err(f"Load '{stem}' first"); return
        except FileNotFoundError as e: self._err(str(e)); return
        imgs = pd["glyph_images"]; cents = pd["centroids_canvas"]

        if self.md.get() == "pair":
            a = self.ea.get().strip(); b = self.eb.get().strip()
            if not a or not b: self._err("Enter both letters"); return
            ms = [x for x in (a, b) if x not in gn]
            if ms: self._err(f"'{', '.join(ms)}' not in font glyphs"); return
            ia, ib = gn.index(a), gn.index(b)
            gt = get_ground_truth(stem, a, b)
            self._pair(ia, ib, a, b, gt, imgs, cents, active, stem)
        else:
            txt = self.we.get().strip()
            if not txt: self._err("Enter a word"); return
            chars = [c for c in txt]
            if len(chars) < 2: self._err("At least 2 chars"); return
            ms = sorted(set(c for c in chars if c not in gn))
            if ms: self._err(f"'{', '.join(ms)}' not in font"); return
            self._word(chars, imgs, cents, gn, active, stem)

    def _sh(self, title, sub=""):
        f = tk.Frame(self.inn, bg=self.BG); f.pack(fill=tk.X, padx=6, pady=(10, 2))
        tk.Label(f, text=title, font=("Segoe UI", 12, "bold"), bg=self.BG).pack(anchor=tk.W)
        if sub: tk.Label(f, text=sub, font=("Segoe UI", 9), fg=self.M, bg=self.BG).pack(anchor=tk.W)
        self.rw.append(f)

    def _ef(self, fig, parent):
        c = FigureCanvasTkAgg(fig, parent); c.draw()
        w = c.get_tk_widget(); w.pack(padx=0, pady=0); return w

    def _figsize(self, img, max_w=220, max_h=260):
        h, w = img.shape[:2]
        s = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        return (w * s / 100, h * s / 100)

    # ── Pair ─────────────────────────────────────────────────────────────────

    def _pair(self, ia, ib, a, b, gt, imgs, cents, active, stem):
        self._sh("Per-Model Results", f"{stem}  ::  {a} + {b}")
        wrap = tk.Frame(self.inn, bg=self.BG); wrap.pack(fill=tk.X, padx=4, pady=2)
        self.rw.append(wrap)
        gh = 256; lcx = gh // 2 + 30; ms = {}; cs = {}; ok = False

        # Build all cells first
        cells = []
        for key, label, rgb in active:
            if key == "GT":
                if gt is None: continue
                d = gt
            else:
                fn = MODEL_FUNCTIONS.get(key)
                try: d = fn(stem, a, b)
                except: continue
                if d is None: continue
            ms[key] = d; cs[key] = rgb; ok = True
            cn = render_pair(imgs, cents, ia, ib, lcx, lcx + d, (0, 0, 0), rgb)
            cells.append((label, d, rgb, cn))

        if not ok: self._err("No predictions produced"); return

        max_cols = 6
        ncols = min(max_cols, len(cells))
        for i, (label, d, rgb, cn) in enumerate(cells):
            r, ccol = divmod(i, ncols)
            cell = tk.Frame(wrap, bg=self.C, bd=0, highlightthickness=0)
            cell.grid(row=r, column=ccol, padx=5, pady=4, sticky="nsew")
            tk.Frame(cell, bg=_h(rgb), height=3).pack(fill=tk.X)
            fw, fh = self._figsize(cn, max_w=230, max_h=280)
            fig = Figure(figsize=(fw, fh + 0.5), facecolor=self.C)
            ax = fig.add_subplot(111); ax.imshow(cn)
            ax.set_title(f"{label}\n{d:.1f} px", fontsize=8, color=_h(rgb),
                         fontweight="bold", pad=4)
            ax.axis("off")
            fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.02)
            self._ef(fig, cell)

        # Make all columns uniform width
        for ccol in range(ncols):
            wrap.grid_columnconfigure(ccol, weight=1, uniform="cells")

        self._sh("Overlay Comparison", "Black = GT. Colored outlines = model predictions (opaque)")
        ovf = tk.Frame(self.inn, bg=self.BG); ovf.pack(fill=tk.X, padx=4, pady=2)
        self.rw.append(ovf)
        order = [k for k in ms if k != "GT"]
        cn = render_overlay(imgs, cents, ia, ib, ms, cs, order)
        fw, fh = self._figsize(cn, max_w=900, max_h=320)
        fig = Figure(figsize=(fw, fh), facecolor=self.C)
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(cn); ax.axis("off")
        self._ef(fig, ovf)

    # ── Word ─────────────────────────────────────────────────────────────────

    def _word(self, chars, imgs, cents, gn, active, stem):
        idxs = [gn.index(c) for c in chars]; n = len(idxs); npairs = n - 1
        gt_sp = [get_ground_truth(stem, chars[p], chars[p + 1]) for p in range(npairs)]
        gt_ok = any(g is not None for g in gt_sp)

        msp = {}
        for key, _, _ in active:
            if key == "GT": continue
            fn = MODEL_FUNCTIONS.get(key); msp[key] = []
            for p in range(npairs):
                a, b = chars[p], chars[p + 1]
                try: v = fn(stem, a, b); v = v if v is not None else 120
                except: v = 120
                msp[key].append(v)

        self._sh("Per-Model Word", f'{stem}  ::  "{chars}"')
        wrap = tk.Frame(self.inn, bg=self.BG); wrap.pack(fill=tk.X, padx=4, pady=2)
        self.rw.append(wrap)
        gh = 256; lcx = gh // 2 + 30

        # Build word cells
        word_cells = []
        if gt_ok and self.gt_var.get():
            gt_sp_valid = [s if s is not None else 120 for s in gt_sp]
            cn = render_word_single(imgs, cents, idxs, gt_sp_valid, (0, 0, 0))
            word_cells.append(("Ground Truth", (0, 0, 0), cn, gt_sp_valid))
        for key, _, rgb in active:
            if key == "GT" or key not in msp: continue
            cn = render_word_single(imgs, cents, idxs, msp[key], rgb)
            word_cells.append((next((l for k, l, _ in active if k == key), key), rgb, cn, msp[key]))

        max_cols = 3
        ncols = min(max_cols, len(word_cells))
        for i, (ttl, rgb, cn, sps) in enumerate(word_cells):
            r, ccol = divmod(i, ncols)
            cell = tk.Frame(wrap, bg=self.C, bd=0, highlightthickness=0)
            cell.grid(row=r, column=ccol, padx=3, pady=2, sticky="nsew")
            tk.Frame(cell, bg=_h(rgb), height=3).pack(fill=tk.X)
            # Pair spacing labels, wrapped into multiple lines
            pair_items = [f"{chars[p]}{chars[p+1]}: {sps[p]:.2f} px" for p in range(len(sps))]
            # Join with 3 spaces, then wrap at a reasonable width
            pair_txt = "   ".join(pair_items)
            lbl = tk.Label(cell, text=pair_txt, font=("Segoe UI", 7), fg=_h(rgb),
                           bg=self.C, justify="left", wraplength=300)
            lbl.pack(fill=tk.X, padx=2)
            fw, fh = self._figsize(cn, max_w=360, max_h=180)
            fig = Figure(figsize=(fw, fh), facecolor=self.C)
            ax = fig.add_subplot(111); ax.imshow(cn)
            ax.set_title(ttl, fontsize=7, color=_h(rgb), fontweight="bold", pad=2)
            ax.axis("off")
            fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02)
            self._ef(fig, cell)
        for ccol in range(ncols):
            wrap.grid_columnconfigure(ccol, weight=1, uniform="wordcells")

        # Overlay
        self._sh("Overlay Word", "Black = GT. Colored outlines = model predictions (opaque)")
        ovf = tk.Frame(self.inn, bg=self.BG); ovf.pack(fill=tk.X, padx=4, pady=2)
        self.rw.append(ovf)

        colors_ov = {k: rgb for k, _, rgb in active if k != "GT" and k in msp}
        order = [k for k in msp]
        cn = render_word_overlay(imgs, cents, idxs,
                                 [s if s is not None else 120 for s in gt_sp],
                                 msp, colors_ov, order)
        fw, fh = self._figsize(cn, max_w=900, max_h=250)
        fig = Figure(figsize=(fw, fh), facecolor=self.C)
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(cn); ax.axis("off")
        self._ef(fig, ovf)


if __name__ == "__main__":
    root = tk.Tk(); App(root); root.mainloop()
