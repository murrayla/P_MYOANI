"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _angHist.py
        Create publicaiton figure 13 plots
"""

# . imports
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# . fig preset
plt.rcParams.update({
    'font.size': 10,          # Base font size for all text
    'axes.titlesize': 10,     # Title size
    'axes.labelsize': 10,     # Axis label size
    'xtick.labelsize': 10,    # X tick label size
    'ytick.labelsize': 10,    # Y tick label size
    'legend.fontsize': 10,    # Legend font size
    'legend.title_fontsize': 10  # Legend title size
})

# . Constants
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]
PIXX, PIXY, PIXZ = 11, 11, 50
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
EXCLUSION = [335, 653, 775, 1108, 1406, 42, 185, 191, 335, 653, 775, 1108, 1406, 1674, 44, 136, 1652, 1732, 1744]
CUBE = {"x": 1000, "y": 1000, "z": 100}
regs = pd.read_csv("_csv/reg_.csv")
whole = pd.read_csv("_csv/rot_norm_w.csv")

mm_to_inch = 1 / 25.4

# ── Experimental data ─────────────────────────────────────────────────────────
EXP = {
    "sig_yy": {
        "compression": [0.13, 0.23, 0.29, 0.30],
        "tension":     [0.29, 0.27, 0.23, 0.19, 0.14, 0.06, 0.03],
    },
    "sig_zz": {
        "compression": [0.13, 0.23, 0.29, 0.30],   # ← replace with real zz values
        "tension":     [0.29, 0.27, 0.23, 0.19, 0.14, 0.06, 0.03],
    },
}

# ── Shared helpers ────────────────────────────────────────────────────────────
STRESS_FILTER = (-50, 50)


def _load_and_filter(path: str) -> pd.DataFrame | None:
    """Load a CSV and apply the interior-volume spatial filter."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None

    return df[
        df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0]) &
        df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1]) &
        df["Z"].between(0.05 * EDGE[2], 0.95 * EDGE[2])
    ]


def _scaled_mean(df: pd.DataFrame, component: str, area: float) -> float | None:
    """Return the scaled mean stress for one component, or None if no data."""
    if df is None:
        return None
    mask = df[component].between(*STRESS_FILTER)
    vals = df.loc[mask, component]
    return vals.mean() * 1000 * area if not vals.empty else None


def _collect_means(disp_pct: int, component: str, area: float,
                   sim_ids: list) -> list[float]:
    """Collect per-simulation mean stress values for a given displacement %."""
    means = []
    for i in sim_ids:
        for resolution in (200, 300):
            df = _load_and_filter(f"_csv/sim/sim_{i}_{disp_pct}_{resolution}.csv")
            if df is not None:
                break
        val = _scaled_mean(df, component, area)
        if val is not None:
            means.append(val)
    return means


# ── Line plot: single test specimen across displacements ──────────────────────
def _newton():
    disps = np.arange(0, 22, 2)
    A_y = (EDGE[1] * 0.90e-6) * (EDGE[2] * 0.90e-6)

    components = ["sig_yy", "sig_zz"]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(160 * mm_to_inch, 80 * mm_to_inch),
        dpi=300,
        sharey=False,
    )

    colors = sns.color_palette("tab20", 4)

    for ax, comp in zip(axes, components):
        means = []
        for disp in disps:
            df = _load_and_filter(f"_csv/sim_test_{disp}_200.csv")
            means.append(_scaled_mean(df, comp, A_y))

        sns.lineplot(x=disps, y=means, ax=ax, color="black",
                     linewidth=1.5, linestyle="-")

        exp = EXP[comp]
        ax.axhline(max(exp["compression"]), color=colors[2],
                   linestyle="--", linewidth=1.2, label="Peak [Compression]")
        ax.axhline(max(exp["tension"]), color=colors[3],
                   linestyle="--", linewidth=1.2, label="Peak [Tension]")

        ax.set_title(comp.replace("sig_", "σ_"), fontsize=9)
        ax.set_xlabel("Displacement [mm]", fontsize=8)
        ax.set_ylabel("Force [N]" if comp == "sig_yy" else "", fontsize=8)
        ax.set_xticks(np.arange(0, 24, 4))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.8, zorder=0)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, framealpha=0.7)

    plt.tight_layout()
    plt.savefig("_png/SUP_NEWTON.png", bbox_inches="tight", dpi=500)
    plt.close()


# ── Box plot: distribution across simulations at fixed displacements ──────────
def _newton_boxplot():
    A_y = (EDGE[1] * 0.90e-6) * (EDGE[2] * 0.90e-6)
    sim_ids  = ["test"] + list(range(18))
    disp_pcts = [20, 18, 16, 14]
    components = ["sig_yy", "sig_zz"]

    # Collect simulation means: {disp_pct: {component: [means]}}
    sim_data = {
        d: {c: _collect_means(d, c, A_y, sim_ids) for c in components}
        for d in disp_pcts
    }

    fig, axes = plt.subplots(
        1, 2,
        figsize=(140 * mm_to_inch, 90 * mm_to_inch),
        dpi=300,
        sharey=False,
    )

    palette = sns.color_palette("muted", 2 + len(disp_pcts))

    for ax, comp in zip(axes, components):
        exp  = EXP[comp]
        groups = [exp["compression"], exp["tension"]] + [
            sim_data[d][comp] for d in disp_pcts
        ]
        labels = ["Exp [Comp]", "Exp [Ten]"] + [f"Sim [{d}%]" for d in disp_pcts]

        sns.boxplot(
            data=groups, ax=ax,
            palette=palette,
            width=0.6,
            showfliers=False,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(comp.replace("sig_", "σ_"), fontsize=9)
        ax.set_ylabel("Force [N]" if comp == "sig_yy" else "", fontsize=8)
        ax.set_xlabel("")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, zorder=0)
        ax.tick_params(labelsize=8)

    plt.tight_layout(pad=1.2)
    plt.savefig("_png/SUP_BOX.png", bbox_inches="tight", dpi=500)
    plt.close()

# . Main
def main():

    # _newton("sig")
    _newton_boxplot()

# . Inititate
if __name__ == "__main__":
    main()