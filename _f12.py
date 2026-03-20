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

def _tension():

    datasets = {
        "Fish et al. (1984)": {
            "eps": [2, 2.2, 2.4],
            "sig": [0.3, 0.6, 1.8],
        },
        "Granzier et al. (1995)": {
            "eps": [1.8, 2, 2.2],
            "sig": [0, 2, 5],
        },
        "Harris et al. (2002)": {
            "eps": [1, 1.02, 1.04, 1.06, 1.08],
            "sig": [0, 3, 6, 12, 21],
        },
        "Caporizzo et al. (2020)": {
            "eps": [1, 1.02, 1.04, 1.06, 1.08, 1.10, 1.11],
            "sig": [0, 2.5, 4.8, 7.0, 9.50, 13.00, 19.00],
        },
        "Li et al. (2015)": {
            "eps": [1.0, 1.05, 1.10, 1.15, 1.20],
            "sig": [0, 0.3857, 1.1048, 1.8023, 2.6942],
        }, 
    }

    # ∆ Define figure size in mm
    mm_to_inch = 1 / 25.4
    fig_w, fig_h = 80 * mm_to_inch, 80 * mm_to_inch  

    # ∆ Plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=300)

    for i, (label, data) in enumerate(datasets.items()):

        sig = np.array(data["sig"]) / np.max(np.array(data["sig"]))
        eps = (np.array(data["eps"]) - np.min(np.array(data["eps"]))) / np.min(np.array(data["eps"]))
        plt.plot(
            eps,
            sig,
            marker="x",
            color="black" if label == "Simulation" else sns.color_palette("Paired")[i],
            alpha=1 if label == "Simulation" else 0.5,
            linestyle= '-' if label == "Simulation" else '--', 
            linewidth=2,
            markersize=4,
            label=label,
        )

    disps = np.arange(0, 22, 2)
    comps = ["yy", "zz"]
    tests = ["test"]
    colors = sns.color_palette("tab20", 4)

    A_y = (EDGE[1] * 0.90 * 10**-6) * (EDGE[2] * 0.90 * 10**-6)

    # Store means for both line and bar plots
    xx = []

    for ii, disp in enumerate(disps):

        df = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")   

        # Internal volume filter
        df_r = df[
            (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df["Z"].between(0.05 * EDGE[2], 0.95 * EDGE[2]))
        ]

        xx.append(
            abs(df_r.loc[df_r[f"sig_xx"].between(-50, 50), f"sig_yy"].mean())
        )

    plt.plot(
        disps/100,
        np.array(xx) / max(xx),
        marker="x",
        color="black",
        alpha=1,
        linestyle= '-',
        linewidth=2,
        markersize=4,
        label="Simulation",
    )

        # plt.plot(
        #     eps,
        #     sig,
        #     marker="x",
        #     color="black",
        #     alpha=1 if label == "Simulation" else 0.5,
        #     linestyle= '-' if label == "Simulation" else '--', 
        #     linewidth=2,
        #     markersize=4,
        #     label=label,
        # )

    # plt.xlabel("Strain, ε")
    # plt.ylabel("Stress, σ")

    # . strip text
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    # plt.legend(frameon=False, fontsize=9)
    # plt.legend(frameon=True, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)

    # . adjust layout for maximum space between plots
    plt.tight_layout(pad=1.0, w_pad=1.0)
    plt.savefig("_png/SUP_NORTEN.png", bbox_inches='tight')


# . Main
def main():

    _tension()

# . Inititate
if __name__ == "__main__":
    main()