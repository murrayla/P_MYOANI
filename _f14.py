"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _angHist.py
        Create publicaiton figure 11 plots
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

def _actStress():

    # . setup
    disps = np.arange(0, 20, 2)
    colors = sns.color_palette("rocket")
    act_ = {"all": [], "non": [], "mid": []}
    comps = ["sig_xy", "sig_xz", "sig_yz"]

    # . fig
    mm_to_inch = 1 / 25.4
    fig_w, fig_h = 160 * mm_to_inch, 50 * mm_to_inch 
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=300)
    axes = axes.flatten()

    # . iterate axes
    for jj, ax in enumerate(axes):

        # . store data
        pas_x = np.zeros(len(disps))

        # . passive
        for ii, disp in enumerate(disps):

            # . load
            df = pd.read_csv(f"_csv/sim/sim_test_{disp}_200.csv") 

            # . logical conditions
            df_r = df[
                (df["X"].between(0.05*EDGE[0], 0.95*EDGE[0])) & 
                (df["Y"].between(0.05*EDGE[1], 0.95*EDGE[1])) & 
                (df["Z"].between(0.05*EDGE[2], 0.95*EDGE[2]))
            ]

            # . store passive
            pas_x[ii] = df_r.loc[df_r[comps[jj]].between(-50, 50), comps[jj]].mean()

        ax.plot(
            disps/100, abs(pas_x),
            linestyle='-',
            lw=2,
            color="blue",
            label="Passive"
        )

        # . iterate keys
        for i, a in enumerate(act_.keys()):

            # . load
            _t = np.load(f"/Users/murrayla/Documents/main_PhD/P_MYOPOR/_npy/{a}_TIME.npy")
            _c = np.load(f"/Users/murrayla/Documents/main_PhD/P_MYOPOR/_npy/{a}_XYZ.npy")
            _s = np.load(f"/Users/murrayla/Documents/main_PhD/P_MYOPOR/_npy/{a}_sig.npy")
            _u = np.load(f"/Users/murrayla/Documents/main_PhD/P_MYOPOR/_npy/{a}_SHORT.npy") / (EDGE[0]//2) 
            _i = 6 * np.arange(len(_t))

            # . logical conditions
            mask = (
                (_c[:, 0] > 0.05 * EDGE[0]) & (_c[:, 0] < 0.95 * EDGE[0]) &
                (_c[:, 1] > 0.05 * EDGE[1]) & (_c[:, 1] < 0.95 * EDGE[1]) &
                (_c[:, 2] > 0.05 * EDGE[2]) & (_c[:, 2] < 0.95 * EDGE[2])
            )

            # . find maximum stress point
            # _idx = np.argmin(abs(_u - 0.18))

            # . plot all
            vals = list(abs(_s[mask][:, _i + (3 + jj)].mean(axis=0)))
            # idx = vals.index(max(vals))
            if a == "all":
                idx = 5
                ls = "--"
            elif a == "non":
                idx = 4
                ls = "-."
            elif a == "mid":
                idx = 5
                ls = ":"
            ax.plot(
                _u[:idx + 1], vals[:idx+ 1],
                linestyle=ls,
                lw=2,
                color=colors[i*2 + 1],
                label=f"Active: {a}"
            )

        if jj == 0:
            ax.set_ylim(-0.1, 0.9)
            ax.set_yticks(np.arange(0, 1, 0.2))
        if jj == 1:
            ax.set_ylim(-0.01, 0.11)
            ax.set_yticks(np.arange(0, 0.12, 0.02))
        if jj == 2:
            ax.set_ylim(-0.001, 0.031)
            ax.set_yticks(np.arange(0, 0.035, 0.007))

        # . plot space formatting
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(-0.01, 0.21)
        ax.set_xticks(np.arange(0, 0.3, 0.1))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)


    # . save
    plt.tight_layout(pad=1.0, w_pad=1.0)
    plt.savefig(f"_png/PUB_ACTNORM.png", bbox_inches="tight")
    plt.close()

# . Main
def main():

    # . stress curve
    _actStress()
    # _actStressBar()

# . Inititate
if __name__ == "__main__":
    main()