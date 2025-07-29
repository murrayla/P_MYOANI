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
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp
from statsmodels.stats.multitest import multipletests

# ∆ Constants
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]
EXCLUSION = [335, 653, 775, 1108, 1406, 42, 185, 191, 335, 653, 775, 1108, 1406, 1674, 44, 136, 1652, 1732, 1744]
CUBE = {"x": 1000, "y": 1000, "z": 100}

plt.rcParams.update({
    'font.size': 10,          # Base font size for all text
    'axes.titlesize': 10,     # Title size
    'axes.labelsize': 10,     # Axis label size
    'xtick.labelsize': 10,    # X tick label size
    'ytick.labelsize': 10,    # Y tick label size
    'legend.fontsize': 8,    # Legend font size
    'legend.title_fontsize': 8  # Legend title size
})

def arrow_plot():

    # ∆ Settings
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), dpi=300)
    axes = axes.flatten()

    # ∆ Colormap settings
    all_sph = []

    # First pass: collect all Sph_[DEG] values
    for i in range(18):
        df = pd.read_csv(f"_csv/tile_{i}_w.csv")
        sph_values = df["Sph_[DEG]"].dropna().astype(float)
        all_sph.extend(sph_values)

    norm = mcolors.Normalize(vmin=min(all_sph), vmax=max(all_sph))
    cmap = plt.cm.viridis

    # Second pass: plot each region
    for i in range(18):
        df = pd.read_csv(f"_csv/tile_{i}_w.csv")
        ax = axes[i]

        for _, row in df.iterrows():
            cx, cy, cz = np.array(ast.literal_eval(row["Centroid"]))
            pc3 = row["PC3_ROT"]
            px, py, pz = np.array(ast.literal_eval(re.sub(r'(?<=[0-9])\s+(?=[\-0-9])', ', ', pc3.strip())))
            sph = float(row["Sph_[DEG]"])

            color = cmap(norm(sph))

            # Scale vector length (you can adjust scale factor as needed)
            scale = 40
            ax.quiver(cx, cy, px, py, angles='xy',
                    scale_units='xy', scale=1/scale, color=color, width=0.003)

        ax.set_title(f"Region {i}")
        ax.set_aspect('equal')
        ax.set_xlim(0, 4096)    
        ax.set_ylim(0, 4096)     
        ax.set_xticks([])
        ax.set_yticks([])

    # ∆ Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label("Spherical Angle [DEG]")

    # ∆ Finalize
    plt.tight_layout()
    fig.savefig(f"_png/Arrow_Grid.png")
    plt.close()


def reg_hists():

    # ∆ Settings
    sns.set_style("whitegrid")

    # ∆ Read all angle data in
    norm_df = pd.read_csv(f"_csv/rot_norm_w.csv")
    norm_df[['x', 'y', 'z']] = norm_df['Centroid'].apply(lambda s: pd.Series(ast.literal_eval(s)))

    # ∆ Extract coordinate data
    x_min, x_max = norm_df['x'].min(), norm_df['x'].max()
    y_min, y_max = norm_df['y'].min(), norm_df['y'].max()
    z_min, z_max = norm_df['z'].min(), norm_df['z'].max()

    # ∆ Extract regional data
    # µ Nucleus set
    nuc_df = (
        (norm_df['z'] <= 160) &
        (norm_df['x'] >= 2300) & (norm_df['x'] <= 3100) &
        (norm_df['y'] <= 2000)
    )
    # µ Periphery set
    per_df = (
        ((norm_df['z'] <= z_min + 200) | (norm_df['z'] >= z_max - 500)) &
        ((norm_df['x'] <= x_min + 200) | (norm_df['x'] >= x_max - 200))
    )
    # µ Central set
    cen_df = ~(nuc_df | per_df)

    lab_df = (((norm_df['x'] <= 1900)))

    # ∆ Store angle data
    sph_nuc = norm_df.loc[nuc_df, "Sph_[DEG]"].tolist()
    # sph_per = norm_df.loc[per_df & ~nuc_df, "Sph_[DEG]"].tolist()  
    sph_per = norm_df.loc[per_df, "Sph_[DEG]"].tolist()  
    sph_cen = norm_df.loc[cen_df, "Sph_[DEG]"].tolist()
    sph_data = [sph_nuc, sph_per, sph_cen]

    # ∆ Allocate figure size
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), dpi=300)
    colors = ["#7EDAFF", "#FAA52B", "#A98BFF"]
    titles = ["Nucleus", "Periphery", "Core"]
    axes = axes.flatten()

    # ∆ Iterate through dataframes
    for i, (data, color, title) in enumerate(zip(sph_data, colors, titles)):

        # ∆ Plot 
        sns.histplot(
            data, bins=30, binrange=(-30, 30), stat='count',
            color=color, edgecolor='black', ax=axes[i]
        )
        axes[i].set_xlim(-30, 30)
        axes[i].set_title(title)
        axes[i].set_xlabel("Angle [Degree]")
        axes[i].set_ylabel("Count")

    # ∆ Save
    plt.tight_layout()
    fig.savefig(f"_png/NucPerCen_Hist.png")
    plt.close()

    # ∆ Alpha for significance indicator
    alpha = 0.05

    # ∆ Store p0values
    p_mwu = []
    p_welch = []
    p_ks = []

    # ∆ Iterate groups
    for i, (data, sub_df) in enumerate(zip(sph_data[:-1], [nuc_df, per_df])):
        
        # ∆ Extract all data
        all_data = norm_df.loc[cen_df, "Sph_[DEG]"].tolist()

        # ∆ Test all cases
        _, p1 = mannwhitneyu(data, all_data, alternative='two-sided')
        _, p2 = ttest_ind(data, all_data, equal_var=False)
        _, p3 = ks_2samp(data, all_data)

        # ∆ Store data
        p_mwu.append(p1)
        p_welch.append(p2)
        p_ks.append(p3)

    x = np.arange(2)
    width = 0.25

    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_style("whitegrid")

    bars1 = plt.bar(x - width, p_mwu, width=width, label='Mann–Whitney U', color="#7EDAFF",edgecolor="black")
    bars2 = plt.bar(x, p_welch, width=width, label="Welch's t-test", color="#FAA52B", edgecolor="black")
    bars3 = plt.bar(x + width, p_ks, width=width, label='K–S test', color="#A98BFF",edgecolor="black")

    print(f"p_mwu: {p_mwu}")
    print(f"p_welch: {p_welch}")
    print(f"p_ks: {p_ks}")

    # exit()

    # Annotate with stars if significant
    for i, (data, color, title) in enumerate(zip(sph_data[:-1], colors[:-1], titles[:-1])):
        if p_mwu[i] < alpha:
            plt.text(x[i] - width, 0.06, "*", ha='center', va='bottom', fontsize=20, color='black')
        if p_welch[i] < alpha:
            plt.text(x[i], 0.06, "*", ha='center', va='bottom', fontsize=20, color='black')
        if p_ks[i] < alpha:
            plt.text(x[i] + width, 0.06, "*", ha='center', va='bottom', fontsize=20, color='black')

    plt.axhline(alpha, color='black', linestyle='--', label=f'p = {alpha}')
    plt.xticks(x, [t for t in titles[:-1]], rotation=45)
    plt.ylabel("p-value")
    plt.title("Region vs Global Statistical Comparison")
    plt.legend()
    plt.ylim(0, 1.05)  # extend y-axis to fit asterisks if needed
    plt.tight_layout()
    plt.savefig(f"_png/NucPerCen_compare.png")
    plt.close()

    # ∆ 3D Scatterplot of coordinates by inclusion region
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Plot each region with a label and color
    ax.scatter(norm_df.loc[nuc_df, 'x'], norm_df.loc[nuc_df, 'y'], norm_df.loc[nuc_df, 'z'],
               color=colors[0], label="Nucleus", alpha=1)
    ax.scatter(norm_df.loc[per_df, 'x'], norm_df.loc[per_df, 'y'], norm_df.loc[per_df, 'z'],
               color=colors[1], label="Periphery", alpha=1)
    # ax.scatter(norm_df.loc[cen_df, 'x'], norm_df.loc[cen_df, 'y'], norm_df.loc[cen_df, 'z'],
    #            color=colors[2], label="Core", alpha=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Region Classification by Coordinate")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"_png/NucPerCen_Scatter3D.png")
    plt.close()

    # ∆ 2D Top-down Scatterplot (X vs Z)
    plt.figure(figsize=(6, 8), dpi=300)

    # Plot each region
    plt.scatter(norm_df.loc[nuc_df, 'x'], norm_df.loc[nuc_df, 'y'],
                color=colors[0], label="Nucleus", alpha=0.6, edgecolors='black', s=40)
    plt.scatter(norm_df.loc[per_df, 'x'], norm_df.loc[per_df, 'y'],
                color=colors[1], label="Periphery", alpha=0.6, edgecolors='black', s=40)
    plt.scatter(norm_df.loc[cen_df, 'x'], norm_df.loc[cen_df, 'y'],
                color=colors[2], label="Core", alpha=0.6, edgecolors='black', s=40)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Top-Down Region Classification (X vs Y)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"_png/NucPerCen_Scatter2D_Whole.png")
    plt.close()

def heat_bar_(angle_key, label):
    
    alpha = 0.05
    angle_data = []
    all_angles = []

    for i in range(18):
        df = pd.read_csv(f"_csv/tile_{i}_w.csv")
        angles = df[angle_key].dropna()
        angles = angles[angles.abs() <= 35].values
        # angles = df[angle_key].dropna().values
        angle_data.append(angles)
        all_angles.extend(angles)

    all_angles = np.array(all_angles)

    p_mwu = []
    p_welch = []
    p_ks = []

    for i in range(18):
        region = angle_data[i]
        global_other = np.hstack([angle_data[j] for j in range(18) if j != i])

        _, p1 = mannwhitneyu(region, global_other, alternative='two-sided')
        _, p2 = ttest_ind(region, global_other, equal_var=False)
        _, p3 = ks_2samp(region, global_other)

        p_mwu.append(p1)
        p_welch.append(p2)
        p_ks.append(p3)

    test_ids = [str(x) for x in range(0, 18, 1)]
    df_pvals = pd.DataFrame({
        "Test_ID": test_ids,
        "Mann_Whitney_U_p_value": p_mwu,
        "KS_p_value": p_ks
    })
    df_pvals.to_csv(f"_csv/ori_p_values.csv", index=False)
    print(f"P-values saved to _csv/ori_p_values.csv")

    x = np.arange(18)
    width = 0.25

    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_style("whitegrid")

    bars1 = plt.bar(x - width, p_mwu, width=width, label='Mann–Whitney U', color="#7EDAFF",edgecolor="black")
    bars2 = plt.bar(x, p_welch, width=width, label="Welch's t-test", color="#FAA52B", edgecolor="black")
    bars3 = plt.bar(x + width, p_ks, width=width, label='K–S test', color="#A98BFF",edgecolor="black")

    # Annotate with stars if significant
    for i in range(18):
        if p_mwu[i] < alpha:
            plt.text(x[i] - width, 0.06, "*", ha='center', va='bottom', fontsize=20, color='black')
        if p_welch[i] < alpha:
            plt.text(x[i], 0.06, "*", ha='center', va='bottom', fontsize=20, color='black')
        if p_ks[i] < alpha:
            plt.text(x[i] + width, 0.06, "*", ha='center', va='bottom', fontsize=20, color='black')

    plt.axhline(alpha, color='black', linestyle='--', label=f'p = {alpha}')
    plt.xticks(x, [f"R{i}" for i in range(18)], rotation=45)
    plt.ylabel("p-value")
    plt.title("Region vs Global Statistical Comparison")
    plt.legend()
    plt.ylim(0, 1.05)  # extend y-axis to fit asterisks if needed
    plt.tight_layout()
    plt.savefig(f"_png/{label}_pvalue_bar_compare.png")
    plt.close()



def hist_dist(angle_key, color, label):

    # ∆ Settings
    # sns.set_style("whitegrid")

    # all_angles = []
    # fig, axes = plt.subplots(3, 6, figsize=(18, 9), dpi=300)
    # axes = axes.flatten()

    # for i in range(0, 18, 1):
    #     df = pd.read_csv(f"_csv/tile_{i}_w.csv")
    #     angles = df[angle_key].dropna()
    #     angles = angles[angles.abs() <= 35]
    #     all_angles.extend(angles)

    #     sns.histplot(angles, bins=30, binrange=(-30, 30), stat='count',
    #                  color=color, edgecolor='black', ax=axes[i])
    #     axes[i].set_xlim(-30, 30)
    #     axes[i].set_title(f"Region {i}")
    #     axes[i].set_xlabel("Angle [Degree]")
    #     axes[i].set_ylabel("Count")

    # plt.tight_layout()
    # fig.savefig(f"_png/{label}_Grid.png")
    # plt.close()

    select = [0, 1, 5, 6, 8, 9, 12, 13, 17]

    # ∆ Gather angle data with significance labels
    angle_data = []

    for i in range(18):
        df = pd.read_csv(f"_csv/tile_{i}_w.csv")
        angles = df[angle_key].dropna()
        angles = angles[angles.abs() <= 30]
        
        group = "p < 0.05" if i in select else "p >= 0.05"
        
        for a in angles:
            angle_data.append({'Region': f'R{i}', 'Angle': a, 'Group': group})

    # ∆ Convert to DataFrame
    angle_df = pd.DataFrame(angle_data)

    # ∆ Plot violinplot with group colors
    plt.figure(figsize=(4.3, 5), dpi=500)
    sns.violinplot(
        x='Angle', y='Region', hue='Group', data=angle_df, inner="quart", inner_kws=dict(alpha=1),
        palette={"p < 0.05": "#A98BFF", "p >= 0.05": color},
        dodge=False  # ensure violins are not split side-by-side
    )

    plt.axvline(0.03, color='black', linestyle='--', linewidth=0.8, label="μ")
    plt.axvline(6.59, color='black', linestyle=':', linewidth=0.8, label="μ ± σ")
    plt.axvline(-6.53, color='black', linestyle=':', linewidth=0.8,)
    plt.xlim((-30, 30))
    plt.ylabel("")
    plt.xlabel("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"_png/{label}_Boxplot.png")
    plt.close()

    # norm_df = pd.read_csv(f"_csv/rot_norm_w.csv")
    # all_angles = norm_df[angle_key].dropna()
    # all_angles = all_angles[all_angles.abs() <= 30].tolist()

    # # ∆ Final cumulative histogram
    # # plt.figure(figsize=(6, 6), dpi=500)
    # sns.histplot(all_angles, bins=40, binrange=(-30, 30), stat='count',
    #              color=color, edgecolor='black')
                 

    # plt.xlim(-30, 30)
    # # plt.xlabel("Angle [DEG]")
    # plt.ylabel("Count [#]")
    # plt.title(f"Total Z-Disc {label} Angle Distribution")
    # plt.tight_layout()
    # plt.savefig(f"_png/{label}_Total.png")
    # plt.close()

    # print(np.mean(all_angles))
    # print(np.std(all_angles))

    # from scipy.stats import skew, kurtosis

    # # Calculate skewness
    # # bias=False corrects for statistical bias, providing the sample skewness.
    # # bias=True (default) calculates the population skewness.
    # skewness_value = skew(all_angles, bias=False)
    # print(f"Skewness: {skewness_value}")

    # # Calculate kurtosis
    # # Fisher's definition of kurtosis is typically used, where a normal distribution has a kurtosis of 0.
    # # bias=False corrects for statistical bias.
    # kurtosis_value = kurtosis(all_angles, fisher=True, bias=False)
    # print(f"Kurtosis: {kurtosis_value}")

    # # If you want Pearson's definition of kurtosis (where normal distribution has kurtosis of 3),
    # # set fisher=False.
    # pearson_kurtosis_value = kurtosis(all_angles, fisher=False, bias=False)
    # print(f"Pearson's Kurtosis: {pearson_kurtosis_value}")
    # exit()

# ∆ Main
def main():

    hist_dist("Sph_[DEG]", "#7EDAFF", "Spherical")
    # heat_bar_("Sph_[DEG]", "Spherical")
    # reg_hists()
    # arrow_plot()

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Open main
    main()
