#!/usr/bin/env python3
"""Common helpers for interactive figure column scripts.

This module centralizes repeated UI/settings/export logic used by
figure{1..5}_col.py scripts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox


# ═══════════════════════════════════════════════════════════════════════════
#  IUCr journal dimensions and standard defaults
# ═══════════════════════════════════════════════════════════════════════════

IUCR_COL_W_MM = 88.5   # single-column width (mm) — IUCr J Appl Cryst
IUCR_COL_W   = IUCR_COL_W_MM / 25.4   # inches (≈ 3.48 in)
IUCR_FULL_W_MM = 171.0  # full-page width (mm) — IUCr two-column spread
IUCR_FULL_W  = IUCR_FULL_W_MM / 25.4  # inches (≈ 6.75 in)
IUCR_TARGET_TEXT_MM = 2.5  # target printed label height at native placement width
IUCR_MAX_H   = 9.50    # maximum figure height (inches)  — 241 mm
IUCR_PANEL_H = 2.75    # typical height for one split panel (inches)
IUCR_DPI     = 600     # monochrome line art (guide: ≥600 d.p.i.)
IUCR_DPI_COLOR = 400   # colour raster (guide: ≥400 d.p.i.)

# IUCr artwork guide: line weights 0.35–1.5 pt at final size (EPS/vector).
IUCR_LINE_LW_MIN = 0.35
IUCR_LINE_LW_MAX = 1.5
IUCR_LINE_LW_STD = 1.0
IUCR_MARKER_SIZE_STD = 4.0
IUCR_MARKER_EDGE_LW = IUCR_LINE_LW_MIN
IUCR_GRID_LW = IUCR_LINE_LW_MIN

# Preferred submission formats (guide §2); PDF kept for local proofing.
IUCR_EXPORT_FORMATS = ("png", "eps", "pdf")

FIBONACCI_SERIES_KEYS = frozenset({"fib_rej", "fib_all"})

# IUCr requires text height 1.5–3 mm when the figure is placed at column width.
# Vector figures scale uniformly, so printed text height (mm) = fontsize (pt) × MM_PER_PT.
MM_PER_PT = 25.4 / 72.0
IUCR_TEXT_MM_MIN = 1.5
IUCR_TEXT_MM_MAX = 3.0
IUCR_TEXT_PT_MIN = IUCR_TEXT_MM_MIN / MM_PER_PT   # ≈ 4.25 pt
IUCR_TEXT_PT_MAX = IUCR_TEXT_MM_MAX / MM_PER_PT   # ≈ 8.47 pt


def pt_to_mm(pt: float) -> float:
    """Printed text height (mm) when the figure is placed at its design width."""
    return float(pt) * MM_PER_PT


def mm_to_pt(mm: float) -> float:
    """Font size (pt) for a target printed height (mm) at column width."""
    return float(mm) / MM_PER_PT


def clamp_iucr_font_pt(pt: float, *, name: str = "font") -> float:
    """Clamp a font size to the IUCr 1.5–3 mm range at column width."""
    pt = float(pt)
    lo, hi = IUCR_TEXT_PT_MIN, IUCR_TEXT_PT_MAX
    if pt < lo:
        import warnings
        warnings.warn(f"{name}: {pt:.2f} pt ({pt_to_mm(pt):.2f} mm) below IUCr minimum; using {lo:.2f} pt")
        return lo
    if pt > hi:
        import warnings
        warnings.warn(f"{name}: {pt:.2f} pt ({pt_to_mm(pt):.2f} mm) above IUCr maximum; using {hi:.2f} pt")
        return hi
    return pt


def iucr_font_sizes() -> dict[str, float]:
    """Default font sizes (pt) within the 1.5–3 mm band at column width."""
    return dict(
        label=8.0,
        tick=7.0,
        legend=7.0,
        legend_title=7.5,
        annotation=7.5,
        minor_tick=6.0,
    )


def iucr_font_sizes_full_page(target_mm: float = IUCR_TARGET_TEXT_MM) -> dict[str, float]:
    """Font sizes (pt) for figures designed at full-page width (native ~2.5 mm labels)."""
    label = clamp_iucr_font_pt(mm_to_pt(target_mm), name="label")
    tick = clamp_iucr_font_pt(mm_to_pt(target_mm * 0.94), name="tick")
    legend = clamp_iucr_font_pt(mm_to_pt(target_mm * 0.90), name="legend")
    legend_title = clamp_iucr_font_pt(mm_to_pt(target_mm * 0.96), name="legend_title")
    annotation = clamp_iucr_font_pt(mm_to_pt(target_mm * 0.94), name="annotation")
    minor_tick = clamp_iucr_font_pt(mm_to_pt(target_mm * 0.82), name="minor_tick")
    return dict(
        label=label,
        tick=tick,
        legend=legend,
        legend_title=legend_title,
        annotation=annotation,
        minor_tick=minor_tick,
    )


def scale_scatter_point_size(
    point_size: float,
    *,
    design_width_in: float,
    target_width_in: float,
    reference_point_size: float | None = None,
    reference_width_in: float | None = None,
) -> float:
    """Scale 3D scatter ``s`` when changing figure width, preserving relative tuning."""
    ps = float(point_size)
    if reference_point_size is not None and reference_width_in is not None:
        ps = reference_point_size * (ps / float(reference_point_size))
    if design_width_in <= 0:
        return ps
    return max(0.5, ps * (float(target_width_in) / float(design_width_in)))


def clamp_iucr_line_lw(lw: float, *, name: str = "line") -> float:
    """Clamp line width to the IUCr 0.35–1.5 pt band at final size."""
    lw = float(lw)
    lo, hi = IUCR_LINE_LW_MIN, IUCR_LINE_LW_MAX
    if lw < lo:
        import warnings
        warnings.warn(f"{name}: {lw:.2f} pt below IUCr minimum; using {lo:.2f} pt")
        return lo
    if lw > hi:
        import warnings
        warnings.warn(f"{name}: {lw:.2f} pt above IUCr maximum; using {hi:.2f} pt")
        return hi
    return lw


def iucr_artwork_style() -> dict[str, float]:
    """Standard line/marker sizes for IUCr submission exports."""
    return dict(
        line_lw=IUCR_LINE_LW_STD,
        marker_size=IUCR_MARKER_SIZE_STD,
        marker_edgewidth=IUCR_MARKER_EDGE_LW,
        grid_lw=IUCR_GRID_LW,
    )


def _pdftops_executable() -> str | None:
    """Return path to Poppler's ``pdftops``, if available."""
    return shutil.which("pdftops")


def pdf_to_eps(pdf_path: str | Path, eps_path: str | Path) -> None:
    """Convert a vector PDF to EPS using Poppler ``pdftops -eps``."""
    pdftops = _pdftops_executable()
    if not pdftops:
        raise RuntimeError(
            "pdftops not found on PATH. Install Poppler, e.g. "
            "'conda install -c conda-forge poppler' or "
            "'sudo apt install poppler-utils'."
        )
    pdf_path = Path(pdf_path)
    eps_path = Path(eps_path)
    subprocess.run(
        [pdftops, "-eps", str(pdf_path), str(eps_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def apply_iucr_artwork_rcparams() -> None:
    """Matplotlib rcParams aligned with IUCr artwork guide (fonts, editable EPS)."""
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def markevery_stride(n_points: int, target_markers: int) -> int:
    """Return matplotlib ``markevery`` stride to show ~*target_markers* symbols."""
    n = int(n_points)
    t = max(1, int(target_markers))
    if n <= 1:
        return 1
    if t >= n:
        return 1
    return max(1, int(np.ceil(n / t)))


def _figure4_panel_xy(
    results: dict,
    key: str,
    metric: str,
    *,
    x_min: float,
    x_max: float,
    use_xlog: bool,
    use_ylog: bool,
):
    """Return (x, y) arrays for points drawn in a figure-4 panel."""
    import numpy as np

    r = results.get(key, {})
    x = np.asarray(r.get("n_eff", []))
    yy = np.asarray(r.get(metric, []))
    if x.size == 0 or yy.size == 0:
        return x[:0], yy[:0]
    idx_sort = np.argsort(x)
    x, yy = x[idx_sort], yy[idx_sort]
    ok = np.isfinite(yy)
    if use_ylog:
        ok &= yy > 0
    if use_xlog:
        ok &= x > 0
    if not ok.any():
        return x[:0], yy[:0]
    x, yy = x[ok], yy[ok]
    ok2 = (x >= x_min) & (x <= x_max)
    return x[ok2], yy[ok2]


def figure4_mean_non_fib_marker_count(
    visible: list[str],
    results: dict,
    metric: str,
    *,
    x_min: float,
    x_max: float,
    xlog: dict,
    ylog: dict,
    pk: str,
) -> int:
    """Average number of plotted points on non-Fibonacci curves (for subsampling)."""
    use_xlog = xlog.get(pk, True)
    use_ylog = ylog.get(pk, False)
    counts: list[int] = []
    for key in visible:
        if key in FIBONACCI_SERIES_KEYS:
            continue
        x, _ = _figure4_panel_xy(
            results, key, metric,
            x_min=x_min, x_max=x_max, use_xlog=use_xlog, use_ylog=use_ylog,
        )
        if x.size > 0:
            counts.append(int(x.size))
    if not counts:
        return 1
    return max(1, int(round(sum(counts) / len(counts))))


def validate_iucr_fontsizes(**named_pt: float) -> None:
    """Raise ValueError if any named size falls outside 1.5–3 mm at column width."""
    bad = []
    for name, pt in named_pt.items():
        mm = pt_to_mm(pt)
        if mm < IUCR_TEXT_MM_MIN or mm > IUCR_TEXT_MM_MAX:
            bad.append(f"{name}={pt:.2f} pt ({mm:.2f} mm)")
    if bad:
        raise ValueError(
            "Font sizes outside IUCr 1.5–3 mm at column width: " + ", ".join(bad)
        )


def apply_iucr_typography_full_page(
    settings_obj,
    field_map: dict[str, str] | None = None,
    *,
    target_mm: float = IUCR_TARGET_TEXT_MM,
) -> dict[str, float]:
    """Override typography for figures placed at full-page width (~2.5 mm labels)."""
    fonts = iucr_font_sizes_full_page(target_mm=target_mm)
    if field_map is None:
        field_map = {}
    defaults = {
        "font_size": "label",
        "text_size": "label",
        "label_size": "label",
        "tick_size": "tick",
        "minor_tick_size": "minor_tick",
        "legend_size": "legend",
        "title_size": "legend_title",
        "subtitle_size": "annotation",
        "legend_title_size": "legend_title",
        "panel_label_size": "legend_title",
        "cbar_font_size": "tick",
    }
    merged = {**defaults, **field_map}
    for attr, key in merged.items():
        if hasattr(settings_obj, attr) and key in fonts:
            setattr(settings_obj, attr, fonts[key])
    return fonts


def apply_iucr_typography(
    settings_obj,
    field_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Override typography on a settings object; return the font size table used.

    *field_map* maps attribute names on *settings_obj* to keys in
    :func:`iucr_font_sizes` (for example ``font_size`` → ``label``).
    """
    fonts = iucr_font_sizes()
    if field_map is None:
        field_map = {}
    defaults = {
        "font_size": "label",
        "label_size": "label",
        "tick_size": "tick",
        "minor_tick_size": "minor_tick",
        "legend_size": "legend",
        "title_size": "legend_title",
        "subtitle_size": "legend_title",
        "legend_title_size": "legend_title",
        "panel_label_size": "legend_title",
        "arrow_a_text_size": "annotation",
        "arrow_b_text_size": "annotation",
    }
    merged = {**defaults, **field_map}
    for attr, key in merged.items():
        if hasattr(settings_obj, attr) and key in fonts:
            setattr(settings_obj, attr, fonts[key])
    return fonts


def export_iucr_figure(
    fig,
    stem: str,
    *,
    dpi: int = IUCR_DPI,
    formats: tuple[str, ...] = IUCR_EXPORT_FORMATS,
) -> None:
    """Save figure assets for IUCR submission (PNG + EPS + PDF by default).

    EPS is produced from the usetex vector PDF via Poppler ``pdftops -eps``.
    Matplotlib's direct EPS path with ``text.usetex=True`` rasterizes through
    Ghostscript; do not pass a high *dpi* to EPS.
    """
    save_kw = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    want_eps = "eps" in formats
    want_pdf = "pdf" in formats
    pdf_path = Path(f"{stem}.pdf")
    temp_pdf: tempfile.NamedTemporaryFile[str] | None = None

    if want_eps and not want_pdf:
        temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf_for_eps = Path(temp_pdf.name)
    else:
        pdf_for_eps = pdf_path

    for ext in formats:
        path = f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, format="png", dpi=dpi, **save_kw)
            print(f"[iucr export] {path}")
        elif ext == "pdf":
            fig.savefig(path, format="pdf", **save_kw)
            print(f"[iucr export] {path}")
        elif ext == "eps":
            continue
        else:
            fig.savefig(path, format=ext, **({**save_kw, "dpi": dpi} if ext != "svg" else save_kw))
            print(f"[iucr export] {path}")

    if want_eps:
        if not want_pdf:
            fig.savefig(pdf_for_eps, format="pdf", **save_kw)
        eps_path = Path(f"{stem}.eps")
        pdf_to_eps(pdf_for_eps, eps_path)
        print(f"[iucr export] {eps_path}")
        if temp_pdf is not None:
            pdf_for_eps.unlink(missing_ok=True)
            temp_pdf.close()


# Standard font sizes for IUCr single-column figures (pt).
# These stay legible after scaling to column width.
_F = iucr_font_sizes()
IUCR_FONT = dict(
    label       = _F["label"],
    tick        = _F["tick"],
    title       = clamp_iucr_font_pt(8.5, name="title"),
    subtitle    = _F["legend_title"],
    legend      = _F["legend"],
    legend_title= _F["legend_title"],
    panel_label = clamp_iucr_font_pt(8.5, name="panel_label"),
    annotation  = _F["annotation"],
    cbar        = _F["tick"],
)

# Standard layout margins for IUCr single-column stacked panels.
IUCR_MARGINS = dict(
    left   = 0.18,
    right  = 0.97,
    top    = 0.97,
    bottom = 0.09,
    hspace = 0.30,
    wspace = 0.10,
)

# Minimum line weight (pt) at final size (IUCr artwork guide).
IUCR_MIN_LW = IUCR_LINE_LW_MIN


def apply_plot_rcparams(
    *,
    font_family: str = "sans-serif",
    sans_serif: Optional[list[str]] = None,
) -> None:
    """Apply a shared plotting style for interactive column figures."""
    if sans_serif is None:
        sans_serif = ["Helvetica Neue", "Helvetica", "DejaVu Sans"]
    matplotlib.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": sans_serif,
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def add_panel_label(
    ax,
    label: str,
    *,
    x: float,
    y: float,
    fontsize: float,
    use_tex: bool = False,
    fontweight: str = "bold",
) -> None:
    """Place a standard panel label (for example (a), (b)).
    
    Updates in-place if a label with the same tag already exists on *ax*.
    """
    tag = f"_panel_label_{label}"
    text = label
    if use_tex:
        text = rf"$\mathbf{{{label}}}$"
        fontweight = "normal"

    # Try to find existing artist with this tag and update it
    for child in ax.get_children():
        if hasattr(child, '_panel_tag') and child._panel_tag == tag:
            child.set_position((x, y))
            child.set_text(text)
            child.set_fontsize(fontsize)
            child.set_fontweight(fontweight)
            return

    # Create new text artist
    if hasattr(ax, "text2D"):
        t = ax.text2D(
            x, y, text,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va="top", ha="left",
        )
    else:
        t = ax.text(
            x, y, text,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va="top", ha="left",
        )
    t._panel_tag = tag


def save_dataclass_settings(path: str, settings_obj, *, wrapper_key: str = "") -> None:
    """Save dataclass settings to JSON."""
    if not is_dataclass(settings_obj):
        raise TypeError("settings_obj must be a dataclass instance")
    payload = asdict(settings_obj)
    if wrapper_key:
        payload = {wrapper_key: payload}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_dataclass_settings(path: str, settings_obj, *, wrapper_key: str = "") -> bool:
    """Load JSON into dataclass settings object.  Returns True on success."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except FileNotFoundError:
        return False
    if wrapper_key:
        if isinstance(payload, dict):
            payload = payload.get(wrapper_key, {})
        else:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    for key, val in payload.items():
        if hasattr(settings_obj, key):
            setattr(settings_obj, key, val)
    return True


def export_figure_with_tex(
    fig,
    out_path: str,
    redraw_callback: Callable[[], None],
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str = "white",
) -> None:
    """Export one figure with temporary text.usetex enabled."""
    prev_usetex = matplotlib.rcParams.get("text.usetex", False)
    matplotlib.rcParams["text.usetex"] = True
    try:
        redraw_callback()
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)
    finally:
        matplotlib.rcParams["text.usetex"] = prev_usetex
        redraw_callback()


# ═══════════════════════════════════════════════════════════════════════════
#  ControlWindow — two-column panel builder with scrollbar
# ═══════════════════════════════════════════════════════════════════════════

class ControlWindow:
    """Two-column control panel builder with a visible scrollbar.

    Widget types
    -------------
    * ``number_row``  — compact labelled text entries (for font sizes, positions, …).
      Fires on Enter only → no continuous redraws while typing.
    * ``slider``      — continuous drag (zoom, alpha, elevation, …).
    * ``textbox``     — free-form text (file paths, annotation text, …).
    * ``button_row``  — action buttons.
    * ``radio_group`` — exclusive choice.
    * ``checkbox_grid`` — toggles arranged in a grid.
    """

    def __init__(
        self,
        fig,
        *,
        left: float = 0.05,
        right: float = 0.92,
        top: float = 0.97,
        col_gap: float = 0.04,
        col1_frac: float = 0.42,
        section_gap: float = 0.014,
        row_gap: float = 0.005,
    ):
        self.fig = fig
        self.left = float(left)
        self.right = float(right)
        self.top = float(top)
        self.col_gap = float(col_gap)
        self.col1_frac = float(col1_frac)
        self.section_gap = float(section_gap)
        self.row_gap = float(row_gap)

        total_w = self.right - self.left
        self.col1_w = total_w * self.col1_frac - self.col_gap * 0.5
        self.col2_w = total_w * (1.0 - self.col1_frac) - self.col_gap * 0.5
        self.col1_x = self.left
        self.col2_x = self.col1_x + self.col1_w + self.col_gap

        self.y = {1: self.top, 2: self.top}
        self.widgets: dict = {}
        self._all_axes: list[tuple] = []          # [(ax, (l,b,w,h)), ...]

        # ── scrollbar (vertical Slider on right edge) ──
        sb_x = self.right + 0.015
        sb_w = 0.022
        sb_bottom = 0.02
        sb_h = self.top - sb_bottom
        self._sb_ax = fig.add_axes([sb_x, sb_bottom, sb_w, sb_h])
        self._sb_ax.set_facecolor("0.94")
        self._scrollbar = Slider(
            self._sb_ax, "", 0, 1, valinit=1.0,
            orientation="vertical", color="0.72",
        )
        self._scrollbar.valtext.set_visible(False)
        self._scrollbar.label.set_visible(False)
        self._scroll_max = 0.0

    # ── internal helpers ──────────────────────────────────────────

    def _col(self, col: int):
        if col == 1:
            return self.col1_x, self.col1_w
        return self.col2_x, self.col2_w

    def _register(self, ax, rect) -> None:
        self._all_axes.append((ax, tuple(rect)))

    # ── scrollbar wiring ──────────────────────────────────────────

    def connect_scroll(self) -> None:
        """Compute scroll range and wire scrollbar + mouse wheel."""
        if not self._all_axes:
            self._sb_ax.set_visible(False)
            return
        min_bottom = min(r[1] for _, r in self._all_axes)
        visible_bottom = 0.02
        overflow = visible_bottom - min_bottom
        if overflow <= 0:
            self._sb_ax.set_visible(False)
            self._scroll_max = 0.0
            return
        self._scroll_max = overflow
        self._scrollbar.on_changed(self._on_sb_changed)
        self.fig.canvas.mpl_connect("scroll_event", self._on_wheel)
        self._on_sb_changed(1.0)

    def _on_sb_changed(self, val) -> None:
        offset = (1.0 - float(val)) * self._scroll_max
        for ax, (l, b, w, h) in self._all_axes:
            nb = b + offset
            ax.set_position([l, nb, w, h])
            ax.set_visible(nb + h > -0.01 and nb < self.top + 0.02)
        self.fig.canvas.draw_idle()

    def _on_wheel(self, evt) -> None:
        if self._scroll_max <= 0:
            return
        step = 0.06
        if evt.button == "up":
            nv = min(1.0, self._scrollbar.val + step)
        elif evt.button == "down":
            nv = max(0.0, self._scrollbar.val - step)
        else:
            return
        self._scrollbar.set_val(nv)

    # ── section header ────────────────────────────────────────────

    def section(self, col: int, title: str, *, header_h: float = 0.014) -> None:
        x, w = self._col(col)
        self.y[col] -= self.section_gap
        rect = (x, self.y[col] - header_h, w, header_h)
        ax = self.fig.add_axes(list(rect))
        ax.set_axis_off()
        ax.text(
            0.0, 0.1, title.upper(),
            fontsize=7, fontweight="bold", color="0.42",
            transform=ax.transAxes, va="bottom",
        )
        self._register(ax, rect)
        self.y[col] -= header_h + 0.004

    # ── number_row (compact text entries) ─────────────────────────

    def number_row(
        self,
        col: int,
        entries,
        *,
        h: float = 0.030,
        label_size: float = 7.0,
    ):
        """Compact labelled number entries in one row (fires on Enter).

        Parameters
        ----------
        entries : list of (name, label, value) or (name, label, value, fmt)
            *fmt* is a ``str.format``-style pattern, default ``"{:.1f}"``.
        """
        x, w = self._col(col)
        n = len(entries)
        gap = 0.004
        entry_w = (w - gap * max(0, n - 1)) / n
        label_frac = 0.42 if n <= 2 else 0.36

        for i, entry in enumerate(entries):
            name, label, value = entry[0], entry[1], entry[2]
            fmt = entry[3] if len(entry) > 3 else "{:.1f}"

            ex = x + i * (entry_w + gap)
            lw = entry_w * label_frac - 0.002
            ew = entry_w * (1.0 - label_frac)

            # label
            rect_l = (ex, self.y[col], lw, h)
            ax_l = self.fig.add_axes(list(rect_l))
            ax_l.set_axis_off()
            ax_l.text(
                1.0, 0.5, label,
                fontsize=label_size, va="center", ha="right",
                transform=ax_l.transAxes, color="0.35",
            )
            self._register(ax_l, rect_l)

            # text entry
            rect_tb = (ex + lw + 0.002, self.y[col], ew, h)
            ax_tb = self.fig.add_axes(list(rect_tb))
            try:
                initial = fmt.format(value)
            except (ValueError, TypeError):
                initial = str(value)
            tb = TextBox(ax_tb, "", initial=initial)
            tb.label.set_fontsize(label_size)
            self.widgets[name] = tb
            self._register(ax_tb, rect_tb)

        self.y[col] -= h + self.row_gap

    # ── slider (continuous drag) ──────────────────────────────────

    def slider(
        self,
        col: int,
        name: str,
        label: str,
        lo: float,
        hi: float,
        val: float,
        *,
        step=None,
        h: float = 0.040,
        fontsize: float = 7.0,
    ):
        x, w = self._col(col)
        label_h = min(0.012, h * 0.35)
        track_h = max(0.014, h - label_h - 0.001)

        rect_label = (x, self.y[col] + track_h + 0.001, w, label_h)
        ax_label = self.fig.add_axes(list(rect_label))
        ax_label.set_axis_off()
        ax_label.text(
            0.0, 0.0, label,
            fontsize=fontsize, color="0.30",
            va="bottom", ha="left", transform=ax_label.transAxes,
        )

        rect_track = (x, self.y[col], w, track_h)
        ax = self.fig.add_axes(list(rect_track))
        ax.set_facecolor("0.97")
        sl = Slider(ax, "", lo, hi, valinit=val, valstep=step)
        sl.label.set_visible(False)
        sl.valtext.set_fontsize(max(6.0, fontsize - 0.5))
        sl.valtext.set_color("0.30")
        self.widgets[name] = sl
        self._register(ax_label, rect_label)
        self._register(ax, rect_track)
        self.y[col] -= h + self.row_gap
        return sl

    # ── textbox ───────────────────────────────────────────────────

    def textbox(self, col: int, name: str, label: str, initial: str, *, h: float = 0.026):
        x, w = self._col(col)
        rect = (x, self.y[col], w, h)
        ax = self.fig.add_axes(list(rect))
        tb = TextBox(ax, label, initial=initial)
        tb.label.set_fontsize(7)
        self.widgets[name] = tb
        self._register(ax, rect)
        self.y[col] -= h + self.row_gap
        return tb

    # ── button row ────────────────────────────────────────────────

    def button_row(self, col: int, pairs, *, h: float = 0.026, pad: float = 0.004):
        x, w = self._col(col)
        n = max(1, len(pairs))
        bw = (w - pad * (n - 1)) / n
        for i, (name, label) in enumerate(pairs):
            bx = x + i * (bw + pad)
            rect = (bx, self.y[col], bw, h)
            ax = self.fig.add_axes(list(rect))
            b = Button(ax, label, color="0.93", hovercolor="0.86")
            b.label.set_fontsize(7)
            self.widgets[name] = b
            self._register(ax, rect)
        self.y[col] -= h + self.row_gap

    # ── radio group ───────────────────────────────────────────────

    def radio_group(
        self, col: int, name: str, labels, *,
        active: int = 0, h: float = 0.060, label_size: float = 7.5,
    ):
        x, w = self._col(col)
        rect = (x, self.y[col] - h, w, h)
        ax = self.fig.add_axes(list(rect))
        rb = RadioButtons(ax, labels, active=active)
        for t in rb.labels:
            t.set_fontsize(label_size)
        self.widgets[name] = rb
        self._register(ax, rect)
        self.y[col] -= h + self.row_gap
        return rb

    # ── checkbox grid ─────────────────────────────────────────────

    def checkbox_grid(
        self,
        col: int,
        name: str,
        items,
        *,
        n_cols: int,
        row_h: float = 0.034,
        label_size: float = 6.5,
        color_map: Optional[dict[str, str]] = None,
    ):
        """items: list[(key, label, initial)].  Returns list of CheckButtons."""
        x, w = self._col(col)
        n = len(items)
        n_cols = max(1, int(n_cols))
        n_rows = (n + n_cols - 1) // n_cols
        cw = w / n_cols
        y0 = self.y[col] - n_rows * row_h
        out = []
        for idx, (key, label, initial) in enumerate(items):
            r, c = idx // n_cols, idx % n_cols
            bx = x + c * cw
            by = y0 + (n_rows - 1 - r) * row_h
            rect = (bx, by, cw - 0.003, row_h - 0.004)
            ax = self.fig.add_axes(list(rect))
            chk = CheckButtons(ax, [label], [initial])
            for txt in chk.labels:
                txt.set_fontsize(label_size)
                if color_map is not None and key in color_map:
                    txt.set_color(color_map[key])
            out.append(chk)
            self._register(ax, rect)
        self.widgets[name] = out
        self.y[col] = y0 - self.row_gap
        return out

    # ── value helpers (work with both Slider and TextBox) ─────────

    def get_val(self, name: str, default: float = 0.0) -> float:
        """Get numeric value from a Slider or TextBox widget."""
        w = self.widgets.get(name)
        if w is None:
            return default
        if isinstance(w, Slider):
            return float(w.val)
        if isinstance(w, TextBox):
            try:
                return float(w.text.strip())
            except (ValueError, TypeError):
                return default
        return default

    def set_val(self, name: str, value, fmt: str | None = None) -> None:
        """Set value on a Slider or TextBox widget."""
        w = self.widgets.get(name)
        if w is None:
            return
        if isinstance(w, Slider):
            w.set_val(float(value))
        elif isinstance(w, TextBox):
            if fmt is not None:
                w.set_val(fmt.format(value))
            else:
                w.set_val(str(value))


# ═══════════════════════════════════════════════════════════════════════════
#  Typography / restyle helpers — shared across all figure*_col.py
# ═══════════════════════════════════════════════════════════════════════════
#
# A "typography rows" spec is a list of rows, where each row is a list of
# tuples:  (widget_name, label, settings_attr)  or
#          (widget_name, label, settings_attr, fmt)
# Default fmt is "{:.1f}".
#
# Example:
#   _TYPO_ROWS = [
#       [("font_size", "Font", "font_size"), ("title_size", "Title", "title_size")],
#       [("panel_label_x", "Lbl x", "panel_label_x", "{:.3f}"), ...],
#   ]

def _flat_typo(rows):
    """Flatten rows into (widget_name, settings_attr, fmt) triples."""
    out = []
    for row in rows:
        for item in row:
            wname, _, attr = item[0], item[1], item[2]
            fmt = item[3] if len(item) > 3 else "{:.1f}"
            out.append((wname, attr, fmt))
    return out


def build_typography_section(cw, col: int, settings_obj, rows) -> None:
    """Add a "Typography" section header + number_row entries."""
    cw.section(col, "Typography")
    for row in rows:
        entries = []
        for item in row:
            wname, label, attr = item[0], item[1], item[2]
            fmt = item[3] if len(item) > 3 else "{:.1f}"
            entries.append((wname, label, getattr(settings_obj, attr), fmt))
        cw.number_row(col, entries)


def sync_typography(cw, settings_obj, rows) -> None:
    """Read typography widget values into *settings_obj*."""
    for wname, attr, _fmt in _flat_typo(rows):
        setattr(settings_obj, attr, cw.get_val(wname, getattr(settings_obj, attr)))


def push_typography(cw, settings_obj, rows) -> None:
    """Push *settings_obj* values into typography widgets."""
    for wname, attr, fmt in _flat_typo(rows):
        cw.set_val(wname, getattr(settings_obj, attr), fmt)


def bind_typography(cw, rows, callback: Callable) -> None:
    """Wire on_submit for every typography text-entry to *callback*."""
    for wname, _, _ in _flat_typo(rows):
        cw.widgets[wname].on_submit(lambda _, _k=wname: callback())


def restyle_axes(
    fig,
    axes_tags,
    s,
    *,
    use_tex: bool = False,
    font_attr: str = "font_size",
    label_offset: float = 0,
    tick_offset: float = -1,
    min_tick: float = 6,
    minor_tick_attr: Optional[str] = None,
    minor_tick_length: float = 2,
    legend_attr: str = "legend_size",
    min_legend: float = 5,
) -> None:
    """Lightweight restyle: update label / tick / panel-label / legend fonts.

    Parameters
    ----------
    axes_tags : iterable of (ax, tag) — e.g. ``[(axL, "(a)"), (axR, "(b)")]``
    font_attr : settings attribute for axis-label font size.
    label_offset : added to *font_attr* for axis labels (e.g. +1 for fig3).
    tick_offset : added to *font_attr* for tick labels (ignored when *s* has
                  a ``tick_size`` attribute).
    minor_tick_attr : if set, also updates ``which='minor'`` tick labels.
    """
    fs = getattr(s, font_attr)
    ts = getattr(s, "tick_size", fs + tick_offset)
    for ax, tag in axes_tags:
        ax.xaxis.label.set_fontsize(fs + label_offset)
        ax.yaxis.label.set_fontsize(fs + label_offset)
        ax.tick_params(which="major", labelsize=max(min_tick, ts))
        if minor_tick_attr is not None:
            ax.tick_params(
                which="minor",
                labelsize=getattr(s, minor_tick_attr),
                length=minor_tick_length,
            )
        add_panel_label(
            ax, tag,
            x=s.panel_label_x, y=s.panel_label_y,
            fontsize=s.panel_label_size, use_tex=use_tex,
        )
        leg = ax.get_legend()
        if leg is not None:
            ls = getattr(s, legend_attr, 6.0)
            for t in leg.get_texts():
                t.set_fontsize(max(min_legend, ls))
    fig.canvas.draw_idle()
