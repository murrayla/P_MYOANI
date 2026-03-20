"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _angHist.py
        Create publicaiton figure 10 plots
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

def _hist():

    angle_key = "Sph_[DEG]"
    color = "#7EDAFF"
    label ="Spherical"

    norm_df = pd.read_csv(f"_csv/rot_norm_w.csv")
    all_angles = norm_df[angle_key].dropna()
    all_angles = all_angles[all_angles.abs() <= 30].tolist()

    # ∆ Define bin size and edges
    bin_size = 1.5  # degrees per bin
    bin_edges = np.arange(-30 - bin_size/2, 30 + bin_size/2, bin_size)

    # ∆ Define figure size in mm
    mm_to_inch = 1 / 25.4
    fig_w, fig_h = 80 * mm_to_inch, 80 * mm_to_inch  

    # ∆ Plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=300)

    # ∆ Plot histogram
    sns.histplot(
        all_angles,
        bins=bin_edges,
        stat='count',
        color=color,
        edgecolor='black',
        ax=ax
    )  

    print(len(all_angles))

    norm_df = pd.read_csv(f"_csv/OUT_ns.csv")

    lower = []
    upper = []

    for i, row in norm_df.iterrows():

        if row["Sph_[DEG]"] > 30: continue

        x, y, _ = ast.literal_eval(row["Centroid"])

        if x <= 2500 and y <= 2500:
            lower.append(row["Sph_[DEG]"])

        elif x >= 1500 and y >= 1500:
            upper.append(row["Sph_[DEG]"])

    # ∆ Plot histogram
    sns.histplot(
        lower,
        bins=bin_edges,
        stat='count',
        color="#FAA52B",
        edgecolor='black',
        ax=ax
    )  

    print(np.mean(lower))
    print(np.std(lower))

    print(len(lower))

    # sns.histplot(
    #     upper,
    #     bins=bin_edges,
    #     stat='count',
    #     color="#A98BFF",
    #     edgecolor='black',
    #     ax=ax
    # )  

    # . strip text
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)

    # . adjust layout for maximum space between plots
    plt.tight_layout(pad=1.0, w_pad=1.0)
    plt.savefig("_png/SUP_DISTVAL.png", bbox_inches='tight')

def _scat():

    data = pd.read_csv("_csv/OUT_ns.csv")
    
    cnts = data["Centroid"]

    # . figure setup
    fig, ax = plt.subplots(1, 1, figsize=(6.9, 6.9), dpi=300)
    
    # 2. Plot Scatters (Larger markers)
    for c in cnts:
        x, y, z = ast.literal_eval(c)

        if x <= 2500 and y <= 2500:
            ax.plot(x, y, 'x', c="#FAA52B")

        elif x >= 1500 and y >= 1500: continue
            # ax.plot(x, y, 'x', c="#A98BFF")

    data = pd.read_csv(f"_csv/norm_stats_whole.csv")
    cnts = data["Centroid"]

    for c in cnts:
        x, y, z = ast.literal_eval(c)

        ax.plot(x, y, 'x', c="#7EDAFF")

    # . strip text
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)

    # . adjust layout for maximum space between plots
    plt.tight_layout(pad=1.0, w_pad=1.0)
    plt.savefig("_png/SUP_SCAT.png", bbox_inches='tight')

# . Main
def main():

    _scat()
    _hist()

# . Inititate
if __name__ == "__main__":
    main()