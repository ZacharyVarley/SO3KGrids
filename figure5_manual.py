#!/usr/bin/env python
"""Figure 5 — Thomson relaxation E₃ + covering radius vs iteration (manual / non-interactive)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_ui_common import add_panel_label


def _covering_radius_star_deg(n_eff: int) -> float:
    """Asymptotic θ*(N) in degrees (matches ``covering_radius.covering_radius_star_deg``)."""
    n = float(n_eff)
    rad = (
        1.889832101454757e+00 * n ** (-1.0 / 3)
        + 5.739307635124498e-01 * n ** -1.0
        + 5.810692175392054e-01 * n ** (-5.0 / 3)
        + 8.597894141163933e-01 * n ** (-7.0 / 3)
        + 1.490131881944242e+00 * n ** -3.0
        + 2.791922498626377e+00 * n ** (-11.0 / 3)
        + 5.481645765756285e+00 * n ** (-13.0 / 3)
        + 1.112163594341394e+01 * n ** -5.0
    )
    return rad * (180.0 / np.pi)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PARAMETERS — edit these directly                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Output
STEM = "figure5_col"
DPI = 600

# Data
DATA_FILE = "figure5_data.npz"
SETTINGS_JSON = "figure5_col_settings.json"


def load_col_settings(path: str = SETTINGS_JSON) -> None:
    """Apply interactive column settings from JSON to module globals."""
    if not os.path.exists(path):
        print(f"[settings] {path} not found — using script defaults")
        return
    import json
    with open(path) as f:
        d = json.load(f)
    g = globals()
    for key, gname in (
        ("label_size", "LABEL_SIZE"),
        ("tick_size", "TICK_SIZE"),
        ("legend_size", "LEGEND_SIZE"),
        ("line_lw", "LINE_LW"),
        ("show_legend_b", "SHOW_LEGEND_B"),
        ("plot_cr_ratio", "PLOT_CR_RATIO"),
        ("show_grid", "SHOW_GRID"),
        ("grid_alpha", "GRID_ALPHA"),
        ("show_theta_star", "SHOW_THETA_STAR"),
    ):
        if key in d:
            g[gname] = d[key]
    if "xlog" in d:
        XLOG.update(d["xlog"])
    if "ylog" in d:
        YLOG.update(d["ylog"])
    if "show" in d:
        SHOW.update(d["show"])
    print(f"[settings] Loaded {path}")

# Figure size (IUCr single column)
FIG_W = 3.25
FIG_H = 5.5

# Typography
LABEL_SIZE = 8.0
TICK_SIZE = 7.0
PANEL_LABEL_SIZE = 9.0
PANEL_LABEL_X = 0.02
PANEL_LABEL_Y = 0.98
LEGEND_SIZE = 7.0

# Lines
LINE_LW = 1.0

# E3 y-axis limits (None = auto)
E3_YMIN = 0.925
E3_YMAX = 0.96

# Options
SHOW_LEGEND_B = True
PLOT_CR_RATIO = False          # True → (θ/θ* − 1), False → θ(°)
SHOW_THETA_STAR = True
SHOW_GRID = True
GRID_ALPHA = 0.25

# Per-panel log scale
XLOG = {"E3": False, "CR": False}
YLOG = {"E3": False, "CR": False}

# Series visibility
SHOW = {"O_KR_FCC": True, "O_KR_PC": True, "O_rej": True}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  THEME                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PALETTE = {
    "O_KR_FCC": "#000000",
    "O_KR_PC":  "#757575",
    "O_rej":    "#BBBBBB",
}
LINESTYLES = {"O_KR_FCC": "-", "O_KR_PC": "-", "O_rej": "-"}
BASE_KEYS = ["O_rej", "O_KR_PC", "O_KR_FCC"]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _series_keys_from_data(data):
    keys = list(data.keys())
    def key_order(k):
        n_eff = int(data.get(k, {}).get("n_eff") or 0)
        order = BASE_KEYS.index(k) if k in BASE_KEYS else 99
        return (n_eff, order)
    return sorted(keys, key=key_order)


def _format_series_label(base, d, use_tex=False):
    n_fz = d.get("n_fz")
    if n_fz is not None:
        n_fz = int(n_fz.item()) if hasattr(n_fz, "item") else int(n_fz)

    if use_tex:
        if base == "O_KR_FCC":
            meth = "\\textrm{KR\\ FCC}"; pad = ""
        elif base == "O_KR_PC":
            meth = "\\textrm{KR}"; pad = "\u00a0" * 7
        elif base == "O_rej":
            meth = "\\textrm{REJ.}"; pad = "\u00a0" * 4
        else:
            meth = "\\textrm{RAND.}"; pad = ""
    else:
        if base == "O_KR_FCC":
            meth = r"\mathrm{KR\;FCC}"; pad = ""
        elif base == "O_KR_PC":
            meth = r"\mathrm{KR}"; pad = "\u00a0" * 7
        elif base == "O_rej":
            meth = r"\mathrm{REJ.}"; pad = "\u00a0" * 4
        else:
            meth = r"\mathrm{RAND.}"; pad = ""

    parts = []
    if n_fz is not None:
        n_str = f"{n_fz:,}".replace(",", r"{,}")
        if use_tex:
            parts.append("$N_{\\mathrm{FZ}}=" + n_str + "$")
        else:
            parts.append(rf"$N_{{\mathrm{{FZ}}}}={n_str}$")
    if parts:
        if use_tex:
            return "$O$ $" + meth + "$" + pad + " (" + ", ".join(parts) + ")"
        return rf"$O$ ${meth}${pad} ({', '.join(parts)})"
    if use_tex:
        return "$O$ $" + meth + "$"
    return rf"$O$ ${meth}$"


def _theta_star_per_group(data, visible):
    def _to_int(v):
        return int(v.item()) if v is not None and hasattr(v, "item") else (int(v) if v is not None else None)
    group_n_eff = {}
    if not any(k.startswith("O_") for k in visible):
        return group_n_eff
    n_eff_vals = [_to_int(data.get(k, {}).get("n_eff")) for k in BASE_KEYS]
    n_eff = max(v or 0 for v in n_eff_vals)
    if n_eff > 0:
        group_n_eff["O"] = n_eff
    return group_n_eff


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found. Generate data first.")
        sys.exit(1)
    raw = dict(np.load(DATA_FILE, allow_pickle=True))
    keys = set()
    for k in raw:
        if k.endswith("_iters"):
            keys.add(k.replace("_iters", ""))
    results = {}
    for key in keys:
        if key.endswith("_rand"):
            continue
        def _scalar(v):
            return int(v.item()) if v is not None and hasattr(v, "item") else v
        n_fz = _scalar(raw.get(f"{key}_n_fz", None))
        n_eff = _scalar(raw.get(f"{key}_n_eff", None))
        n_eff_run = _scalar(raw.get(f"{key}_n_eff_run", None))
        results[key] = {
            "E3_ratio": raw.get(f"{key}_E3_ratio", np.array([])),
            "cr_deg":   raw.get(f"{key}_cr_deg", np.array([])),
            "iters":    raw.get(f"{key}_iters", np.array([])),
            "n_fz": n_fz,
            "n_eff": n_eff,
            "n_eff_run": n_eff_run,
        }
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _axis_labels(use_tex=True):
    x_label = "$\\textrm{Iteration}$" if use_tex else r"$\mathrm{Iteration}$"
    e3_label = "$E_3\\,/\\,E_3^{\\,*}$" if use_tex else r"$E_3\,/\,E_3^{\,*}$"
    if PLOT_CR_RATIO:
        cr_label = "$(\\theta_{\\mathrm{S^3}}/\\theta_{\\mathrm{S^3}}^{*}) - 1$"
        cr_ref_label = "$\\theta_{\\mathrm{S^3}}/\\theta_{\\mathrm{S^3}}^* - 1 = 0$"
    else:
        cr_label = "$\\theta_{\\mathrm{S^3}}(^\\circ)$"
        cr_ref_label = "$\\theta_{\\mathrm{S^3}}^*$"
    return x_label, e3_label, cr_label, cr_ref_label


def _max_iteration(data, visible):
    max_x = 1
    for key in visible:
        d = data[key]
        arr = d.get("E3_ratio", d.get("cr_deg", []))
        n = len(arr) if hasattr(arr, "__len__") else 0
        if n > 0:
            iters = np.asarray(d.get("iters", np.arange(n)))
            if len(iters) > 0:
                max_x = max(max_x, float(np.max(iters)) + 1)
    return max_x


def plot_e3_panel(ax, data, visible, *, max_x, use_tex=True):
    """E₃/E₃* vs iteration (no panel letter)."""
    x_label, e3_label, _, _ = _axis_labels(use_tex)
    for key in visible:
        d = data[key]
        iters = d.get("iters", np.arange(len(d["E3_ratio"])))
        if len(iters) == 0:
            iters = np.arange(len(d["E3_ratio"]))
        x_vals = np.asarray(iters, dtype=float) + 1
        yy = np.asarray(d["E3_ratio"])
        ok = np.isfinite(yy)
        if YLOG.get("E3", False):
            ok &= yy > 0
        if not ok.any():
            continue
        ax.plot(x_vals[ok], yy[ok],
                color=PALETTE.get(key, "#666"),
                linestyle=LINESTYLES.get(key, "-"),
                label=_format_series_label(key, d, use_tex=use_tex),
                lw=LINE_LW)

    ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
    ax.set_ylabel(e3_label, fontsize=LABEL_SIZE)
    ax.set_xscale("log" if XLOG.get("E3", False) else "linear")
    ax.set_yscale("log" if YLOG.get("E3", False) else "linear")
    ax.set_xlim(1, max_x)
    if E3_YMIN is not None and E3_YMAX is not None:
        ax.set_ylim(E3_YMIN, E3_YMAX)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)


def plot_cr_panel(ax, data, visible, *, max_x, use_tex=True):
    """Covering radius vs iteration (no panel letter)."""
    x_label, _, cr_label, cr_ref_label = _axis_labels(use_tex)
    for key in visible:
        d = data[key]
        iters = d.get("iters", np.arange(len(d["cr_deg"])))
        if len(iters) == 0:
            iters = np.arange(len(d["cr_deg"]))
        x_vals = np.asarray(iters, dtype=float) + 1
        yy = np.asarray(d["cr_deg"], dtype=float)
        if PLOT_CR_RATIO:
            n_eff_i = d.get("n_eff")
            n_eff_i = int(n_eff_i.item()) if hasattr(n_eff_i, "item") else n_eff_i
            theta_star_i = (_covering_radius_star_deg(int(n_eff_i))
                            if n_eff_i is not None and int(n_eff_i) > 0
                            else float("nan"))
            if np.isfinite(theta_star_i) and theta_star_i > 0:
                yy = yy / theta_star_i - 1.0
            else:
                yy = np.full_like(yy, np.nan, dtype=float)
        ok = np.isfinite(yy)
        if YLOG.get("CR", False):
            ok &= yy > 0
        if not ok.any():
            continue
        ax.plot(x_vals[ok], yy[ok],
                color=PALETTE.get(key, "#666"),
                linestyle=LINESTYLES.get(key, "-"),
                label=_format_series_label(key, d, use_tex=use_tex),
                lw=LINE_LW)

    if SHOW_THETA_STAR and visible:
        if PLOT_CR_RATIO:
            ax.axhline(0.0, color="#000", linestyle=":", lw=LINE_LW,
                       alpha=0.8, label=cr_ref_label)
        else:
            group_n_eff = _theta_star_per_group(data, visible)
            for _, n_eff in group_n_eff.items():
                theta_star = _covering_radius_star_deg(n_eff)
                ax.axhline(theta_star, color="#000", linestyle=":", lw=LINE_LW,
                           alpha=0.8, label=cr_ref_label)

    ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
    ax.set_ylabel(cr_label, fontsize=LABEL_SIZE)
    ax.set_xscale("log" if XLOG.get("CR", False) else "linear")
    ax.set_yscale("log" if YLOG.get("CR", False) else "linear")
    ax.set_xlim(1, max_x)
    if SHOW_LEGEND_B:
        ax.legend(loc="upper right", fontsize=LEGEND_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA)


def main():
    load_col_settings()
    data = load_data()
    series_keys = _series_keys_from_data(data)
    visible = [k for k in series_keys if SHOW.get(k, True)]
    max_x = _max_iteration(data, visible)

    with plt.rc_context({"text.usetex": True}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H))
        plot_e3_panel(ax1, data, visible, max_x=max_x)
        plot_cr_panel(ax2, data, visible, max_x=max_x)
        add_panel_label(ax1, "(a)", x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=True)
        add_panel_label(ax2, "(b)", x=PANEL_LABEL_X, y=PANEL_LABEL_Y,
                        fontsize=PANEL_LABEL_SIZE, use_tex=True)
        plt.tight_layout()

        for ext in ("png", "pdf"):
            path = f"{STEM}.{ext}"
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"[export] {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
