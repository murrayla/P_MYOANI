"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _angHist.py
        Create histogram data of angle values
"""

# ∆ Raw
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ∆ Constants
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]
EXCLUSION = [335, 653, 775, 1108, 1406] 
CUBE = {"x": 1000, "y": 1000, "z": 100}

def plot_angle_distributions(angle_key, color, label, prefix):

    # ∆ Settings
    sns.set_style("whitegrid")

    all_angles = []
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), dpi=300)
    axes = axes.flatten()

    for i in range(18):
        df = pd.read_csv(f"_csv/tile_{i}.csv")
        angles = df[angle_key].dropna()
        all_angles.extend(angles)

        sns.histplot(angles, bins=30, binrange=(-30, 30), stat='count',
                     color=color, edgecolor='black', ax=axes[i])
        axes[i].set_xlim(-30, 30)
        axes[i].set_title(f"Region {i}")
        axes[i].set_xlabel("Angle [Degree]")
        axes[i].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(f"_png/{prefix}_Grid.png")
    plt.close()

    # ∆ Final cumulative histogram
    plt.figure(figsize=(6, 6), dpi=300)
    sns.histplot(all_angles, bins=30, binrange=(-30, 30), stat='count',
                 color=color, edgecolor='black')

    plt.xlim(-30, 30)
    plt.xlabel("Angle [Degree]")
    plt.ylabel("Count")
    plt.title(f"Total Z-Disc {label} Angle Distribution")
    plt.tight_layout()
    plt.savefig(f"_png/{prefix}_Total.png")
    plt.close()

def plot_violin_significance(angle_key, color, label, prefix, reference_group=0):
    # ∆ Collect all data
    combined_data = []
    for i in range(18):
        df = pd.read_csv(f"_csv/tile_{i}.csv")
        angles = df[angle_key].dropna()
        combined_data.extend(zip([f"Region {i}"] * len(angles), angles))

    df_all = pd.DataFrame(combined_data, columns=["Region", "Angle"])

    # ∆ Create violin plot
    plt.figure(figsize=(16, 6), dpi=300)
    sns.violinplot(data=df_all, x="Region", y="Angle", palette=[color]*18, inner="box", linewidth=1.2)

    plt.xticks(rotation=90)
    plt.ylabel("Angle [Degree]")
    plt.title(f"{label} Angle Distributions with Significance Marking")
    plt.tight_layout()

    # ∆ Statistical comparison vs reference group
    reference_name = f"Region {reference_group}"
    y_max = df_all["Angle"].max()
    height_offset = 2  # spacing for sig lines

    for i in range(18):
        if i == reference_group:
            continue
        current_name = f"Region {i}"
        group1 = df_all[df_all["Region"] == reference_name]["Angle"]
        group2 = df_all[df_all["Region"] == current_name]["Angle"]

        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')

        if p < 0.05:
            # ∆ Draw significance bar
            x1, x2 = reference_group, i
            y = y_max + height_offset
            plt.plot([x1, x1, x2, x2], [y, y + 0.5, y + 0.5, y], lw=1.2, c='k')
            plt.text((x1 + x2) / 2, y + 0.7, f"* (p={p:.3f})", ha='center', va='bottom', fontsize=9)
            y_max += height_offset + 1  # bump for next line

    # ∆ Save
    plt.savefig(f"_png/{prefix}_Violin_Significance.png")
    plt.close()

def pairwise_significance_heatmap(df_all, prefix, label):
    regions = sorted(df_all["Region"].unique(), key=lambda x: int(re.search(r'\d+', x).group()))
    p_matrix = np.ones((len(regions), len(regions)))

    # ∆ Calculate pairwise p-values
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if i < j:
                group1 = df_all[df_all["Region"] == r1]["Angle"]
                group2 = df_all[df_all["Region"] == r2]["Angle"]
                stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
                p_matrix[i, j] = p

    # ∆ FDR correction
    triu_indices = np.triu_indices(len(regions), k=1)
    p_vals = p_matrix[triu_indices]
    _, p_corrected, _, _ = multipletests(p_vals, method='fdr_bh')
    p_matrix[triu_indices] = p_corrected

    # ∆ Symmetrize
    p_matrix = np.minimum(p_matrix, p_matrix.T)

    # ∆ Plot heatmap
    plt.figure(figsize=(12, 10), dpi=300)
    mask = np.tril(np.ones_like(p_matrix, dtype=bool))
    sns.heatmap(p_matrix, xticklabels=regions, yticklabels=regions,
                cmap="magma", mask=mask, square=True,
                cbar_kws={'label': 'Corrected p-value'}, annot=True, fmt=".3f",
                vmin=0.0, vmax=0.1) # Set data range from 0.0 to 0.1

    plt.title(f"{label} Pairwise Significance Heatmap (FDR-corrected)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"_png/{prefix}_Significance_Heatmap.png")
    plt.close()

# ∆ Main
def main():

    # # ∆ Plot lateral (XY) angle data
    # plot_angle_distributions(
    #     angle_key="Azi_[DEG]",
    #     color="#7EDAFF",
    #     label="Lateral (XY)",
    #     prefix="Lateral"
    # )

    # # ∆ Plot elevation (XZ) angle data
    # plot_angle_distributions(
    #     angle_key="Ele_[DEG]",
    #     color="#FAA52B",
    #     label="Elevation (XZ)",
    #     prefix="Elevation"
    # )

    plot_angle_distributions(
        angle_key="Sph_[DEG]",
        color="#7EDAFF",
        label="Spherical Angle [DEG]",
        prefix="Spherical"
    )

    # # ∆ Violin + significance plot for lateral (XY)
    # plot_violin_significance(
    #     angle_key="Azi_[DEG]",
    #     color="#7EDAFF",
    #     label="Lateral (XY)",
    #     prefix="Lateral",
    #     reference_group=0  # Compare all other regions against Region 0
    # )

    # # ∆ Violin + significance plot for elevation (XZ)
    # plot_violin_significance(
    #     angle_key="Ele_[DEG]",
    #     color="#FAA52B",
    #     label="Elevation (XZ)",
    #     prefix="Elevation",
    #     reference_group=0
    # )

    # combined_data = []
    # for i in range(18):
    #     df = pd.read_csv(f"_csv/tile_{i}.csv")
    #     angles = df["Ele_[DEG]"].dropna()
    #     combined_data.extend(zip([f"Region {i}"] * len(angles), angles))

    # # ∆ Create full DataFrame for analysis
    # df_all = pd.DataFrame(combined_data, columns=["Region", "Angle"])

    # pairwise_significance_heatmap(df_all, prefix="Elevation", label="Elevation (XZ)")

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Open main
    main()
