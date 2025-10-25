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
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp
from statsmodels.stats.multitest import multipletests

plt.rcParams.update({
    'font.size': 12,          # Base font size for all text
    'axes.titlesize': 12,     # Title size
    'axes.labelsize': 12,     # Axis label size
    'xtick.labelsize': 12,    # X tick label size
    'ytick.labelsize': 12,    # Y tick label size
    'legend.fontsize': 12,    # Legend font size
    'legend.title_fontsize': 12  # Legend title size
})

# ∆ Constants
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

def load_and_filter(filename, mx, my, mz, wid, disp=True, bounds=None):
    """Load CSV and filter by either bounding box (wid) or fractional bounds."""
    df = pd.read_csv(filename)
    if bounds:  # fractional EDGE bounds
        df = df[
            (df["X"] >= bounds[0][0]) & (df["X"] <= bounds[0][1]) &
            (df["Y"] >= bounds[1][0]) & (df["Y"] <= bounds[1][1]) &
            (df["Z"] >= bounds[2][0]) & (df["Z"] <= bounds[2][1])
        ]
    else:  # width around mx,my,mz
        df = df[
            (df["X"] >= mx - wid) & (df["X"] <= mx + wid) &
            (df["Y"] >= my - wid) & (df["Y"] <= my + wid) &
            (df["Z"] >= mz - wid) & (df["Z"] <= mz + wid)
        ]
    df = df.copy()
    if disp:
        df["X_du"] = df["X"] + df["disp_x"]
        df["Y_du"] = df["Y"] + df["disp_y"]
        df["Z_du"] = df["Z"] + df["disp_z"]
    return df


def plot_convex_quiver(j, ax, control_df, test_df, color, linewidth=1.5, scatter_center=None):
    """Plot control/test convex hulls and quiver vectors."""
    control_pts = control_df[["Y_du", "Z_du"]].dropna().values
    hull = ConvexHull(control_pts)
    ax.fill(control_pts[hull.vertices, 0], control_pts[hull.vertices, 1],
            facecolor="none", edgecolor="black", linewidth=linewidth-1)
    if scatter_center:
        ax.scatter(*scatter_center, marker="x", color="black", s=10)

    YZ = test_df[["Y", "Z"]].dropna().values
    vYZ = test_df[["disp_y", "disp_z"]].dropna().values
    test_pts = test_df[["Y_du", "Z_du"]].dropna().values
    hull = ConvexHull(test_pts)
    ax.fill(test_pts[hull.vertices, 0], test_pts[hull.vertices, 1],
            facecolor="none", edgecolor=color, linewidth=linewidth)
    
    y_bar, z_bar = np.mean(test_pts[hull.vertices, 0]), np.mean(test_pts[hull.vertices, 1])
    print(j)
    print(y_bar, z_bar)
    # ax.fill(test_pts[hull.vertices, 0], test_pts[hull.vertices, 1],
    #         facecolor=color, edgecolor="none", alpha=0.5)
    ax.quiver(YZ[::20, 0], YZ[::20, 1], vYZ[::20, 0], vYZ[::20, 1], color=color, width=0.02, alpha=0.5)


def calc_rotation(df, my, mz, min_radius=100):
    """Calculate rotation about X-axis in degrees."""
    vy, vz = df["Y"].values - my, df["Z"].values - mz
    vy_d, vz_d = df["Y_du"].values - my, df["Z_du"].values - mz
    radius = np.sqrt(vy**2 + vz**2)
    mask = radius > min_radius
    angles_0 = np.rad2deg(np.arctan2(vz[mask], vy[mask]))
    angles_1 = np.rad2deg(np.arctan2(vz_d[mask], vy_d[mask]))
    return angles_1 - angles_0


def quiver_plot():
    select = [0, 4, 6, 14]
    colors = sns.color_palette("tab20", 18)

    # Common geometry setup
    xx = np.flip(np.linspace(0.05 * EDGE[0], EDGE[0] - 0.05 * EDGE[0], 5))
    yy = np.linspace(0.05 * EDGE[1], EDGE[1] - 0.05 * EDGE[1], 5)
    zz = np.linspace(0.05 * EDGE[2], EDGE[2] - 0.05 * EDGE[2], 5)
    mx, my, mz = xx[2], yy[2], zz[2]
    wid = 1000

    # # --- 1. Main 3x6 grid of quivers ---
    # fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    control_df = load_and_filter("_csv/sim_test_20_200.csv", mx, my, mz, wid)
    # for i, ax in enumerate(axes.flatten()):
    #     ax.set_title(f"R{i}", fontsize=10)
    #     test_file = f"_csv/sim_{i}_20_{300 if i == 11 else 200}.csv"
    #     test_df = load_and_filter(test_file, mx, my, mz, wid)
    #     plot_convex_quiver(j, ax, control_df, test_df, colors[i], scatter_center=(my, mz))
    #     ax.set_xlim(my - 1.75 * wid, my + 1.75 * wid)
    #     ax.set_ylim(mz - 1.75 * wid, mz + 1.75 * wid)
    #     ax.set_aspect('equal', adjustable='box')
    # fig.tight_layout(); fig.savefig("_png/quiver_shear.png", dpi=500); plt.close(fig)

    # # # --- 2. 1x4 grid for select regions ---
    fig, axes = plt.subplots(4, 1, figsize=(2, 6))
    for ax, j in zip(axes, select):
        ax.set_title(f"R{j}", fontsize=10)
        test_df = load_and_filter(f"_csv/sim_{j}_20_200.csv", mx, my, mz, wid)
        plot_convex_quiver(j, ax, control_df, test_df, colors[j], linewidth=3, scatter_center=(my, mz))
        ax.set_xlim(my - 1.75 * wid, my + 1.75 * wid)
        ax.set_ylim(mz - 1.75 * wid, mz + 1.75 * wid)
        ax.set_aspect('equal', adjustable='box')
        if j != select[-1]: ax.set_xticklabels([])
    fig.tight_layout(); fig.savefig("_png/PUB_shear_04614.png", dpi=500); plt.close(fig)

    # # --- 3. Radar rotation plots for select ---
    # fig, axes = plt.subplots(4, 1, figsize=(2, 4.2))
    # for ax, i in zip(axes, select):
    #     ax.set_title(f"R{j}", fontsize=10)
    #     test_file = f"_csv/sim_{i}_20_{300 if i == 11 else 200}.csv"
    #     test_df = load_and_filter(test_file, mx, my, mz, wid)
    #     rotation_angles = calc_rotation(test_df, my, mz)
    #     bins = np.linspace(-30, 30, 64)
    #     hist, _ = np.histogram(rotation_angles, bins=bins)
    #     theta = 0.5 * (bins[:-1] + bins[1:])
    #     ax.plot(theta, hist, color=colors[i])
    #     ax.fill(theta, hist, color=colors[i], alpha=0.4)
    #     if i != select[-1]: ax.set_xticklabels([])
    # fig.tight_layout(); fig.savefig("_png/PUB_rothist.png", dpi=500); plt.close(fig)

    # exit()

    # --- 4. Violin plot of all rotations ---
    all_rotations = []
    for j in range(18):
        test_file = f"_csv/sim_{j}_20_{300 if j == 11 else 200}.csv"
        test_df = load_and_filter(test_file, mx, my, mz, wid)
        for angle in calc_rotation(test_df, my, mz):
            if abs(angle) <= 45:
                all_rotations.append({"rotation": angle, "region": f"R{j}", "region_index": j})
    df_rot = pd.DataFrame(all_rotations)
    palette = {f"R{i}": colors[i] for i in range(18)}
    fig, ax = plt.subplots(figsize=(7.8, 2.1))
    sns.violinplot(data=df_rot, x="region", y="rotation", hue="region",
                   palette=palette, ax=ax, dodge=False, inner="quart")
    # sns.boxplot(data=df_rot, x="region", y="rotation", hue="region",
    #                palette=palette, ax=ax, showfliers=False)
    
    for angle in calc_rotation(control_df, my, mz):
            if abs(angle) <= 45:
                all_rotations.append({"rotation": angle, "region": f"test", "region_index": 0})
    df_rot = pd.DataFrame(all_rotations)

    q2 = np.quantile(df_rot["rotation"], 0.5)
    q1 = np.quantile(df_rot["rotation"], 0.25)
    q3 = np.quantile(df_rot["rotation"], 0.75)
    # std_control = np.std(control_disps)

    # Add reference lines from control data
    # ax.axhline(q2, color='black', linestyle='--', linewidth=1)
    # ax.axhline(q1, color='black', linestyle=':', linewidth=1)
    # ax.axhline(q3, color='black', linestyle=':', linewidth=1)
    
    ax.set_ylabel(""); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig("_png/vio_rotation.png", dpi=500); plt.close(fig)

def shear_plot():
    # ∆ Prepare figure for displacement visualization
    fig_disp, axes_disp = plt.subplots(3, 6, figsize=(14, 7))
    axes_disp = axes_disp.flatten()
    colors = sns.color_palette("tab20", len(axes_disp) + 1)

    # ∆ Prepare figure for radar plots
    fig_radar, axes_radar = plt.subplots(3, 6, subplot_kw={'projection': 'polar'}, figsize=(14, 7))
    axes_radar = axes_radar.flatten()

    # ∆ Determine points within the 3D space
    xx = np.flip(np.linspace(0.05 * EDGE[0], EDGE[0] - 0.05 * EDGE[0], 5))
    yy = np.linspace(0.05 * EDGE[1], EDGE[1] - 0.05 * EDGE[1], 5)
    zz = np.linspace(0.05 * EDGE[2], EDGE[2] - 0.05 * EDGE[2], 5)

    # ∆ Keep Control
    df = pd.read_csv(f"_csv/sim_test_20_200.csv")

    # ∆ Middle points
    mx, my, mz = xx[2], yy[2], zz[2]
    wid = 2000

    # ∆ Select for middle values
    df_r = df[
        (df["X"] >= mx - wid) & (df["X"] <= mx + wid) &
        (df["Y"] >= my - wid) & (df["Y"] <= my + wid) &
        (df["Z"] >= mz - wid) & (df["Z"] <= mz + wid)
    ].copy()

    # ∆ Add displacement terms
    df_r["X_du"] = df_r["X"] + df_r["disp_x"]
    df_r["Y_du"] = df_r["Y"] + df_r["disp_y"]
    df_r["Z_du"] = df_r["Z"] + df_r["disp_z"]

    # ∆ Loop through regions
    for i in range(18):
        ax_disp = axes_disp[i]
        ax_radar = axes_radar[i]

        ax_disp.set_title(f"R{i}", fontsize=10)

        # ∆ Convex hull of control
        control_pts = df_r[["Y_du", "Z_du"]].dropna().values
        hull = ConvexHull(control_pts)
        hull_pts = control_pts[hull.vertices]
        cy_bar, cz_bar = np.mean(hull_pts[:, 0]), np.mean(hull_pts[:, 1])
        ax_disp.fill(hull_pts[:, 0], hull_pts[:, 1], facecolor="none", edgecolor=colors[0], linewidth=1.5, linestyle="--")
        ax_disp.scatter(my, mz, marker="x", color=colors[0], s=10)

        # ∆ Load test
        if i == 11:
            df_i = pd.read_csv(f"_csv/sim_{i}_20_300.csv")
        else:
            df_i = pd.read_csv(f"_csv/sim_{i}_20_200.csv")

        df_i_r = df_i[
            (df_i["X"] >= mx - wid) & (df_i["X"] <= mx + wid) &
            (df_i["Y"] >= my - wid) & (df_i["Y"] <= my + wid) &
            (df_i["Z"] >= mz - wid) & (df_i["Z"] <= mz + wid)
        ].copy()

        df_i_r["X_du"] = df_i_r["X"] + df_i_r["disp_x"]
        df_i_r["Y_du"] = df_i_r["Y"] + df_i_r["disp_y"]
        df_i_r["Z_du"] = df_i_r["Z"] + df_i_r["disp_z"]

        # ∆ Hull of test
        test_pts = df_i_r[["Y_du", "Z_du"]].dropna().values
        hull = ConvexHull(test_pts)
        hull_pts = test_pts[hull.vertices]
        y_bar, z_bar = np.mean(hull_pts[:, 0]), np.mean(hull_pts[:, 1])

        ax_disp.fill(hull_pts[:, 0], hull_pts[:, 1], facecolor=colors[i], alpha=0.3)
        ax_disp.fill(hull_pts[:, 0], hull_pts[:, 1], facecolor="none", edgecolor=colors[i], linewidth=1.5)
        ax_disp.scatter(y_bar, z_bar, color=colors[i], s=10)

        # ∆ Rotation calculation (about X axis)
        vy = df_i_r["Y"].values - cy_bar
        vz = df_i_r["Z"].values - cz_bar
        vy_d = df_i_r["Y_du"].values - cy_bar
        vz_d = df_i_r["Z_du"].values - cz_bar

        radius = np.sqrt(vy**2 + vz**2)
        mask = radius > 100  # exclude near-axis noise

        angles_0 = np.arctan2(vz[mask], vy[mask])
        angles_1 = np.arctan2(vz_d[mask], vy_d[mask])
        rotation_angles = (angles_1 - angles_0 + np.pi) % (2 * np.pi) - np.pi

        mean_angle = np.degrees(np.mean(rotation_angles))
        std_angle = np.degrees(np.std(rotation_angles))

        ax_disp.text(y_bar, z_bar - 500,
                     f"$\\mu$: {mean_angle:.1f}°, $\\sigma$: {std_angle:.1f}",
                     fontsize=8, ha="center", va="center")

        # ∆ Plot radar histogram
        bins = np.linspace(-np.pi, np.pi, 36)
        hist, _ = np.histogram(rotation_angles, bins=bins)
        theta = 0.5 * (bins[:-1] + bins[1:])
        ax_radar.plot(theta, hist, color=colors[i])
        ax_radar.fill(theta, hist, color=colors[i], alpha=0.4)
        ax_radar.set_yticklabels([])
        ax_radar.set_title(f"R{i}", fontsize=9)

        # Format displacement plot
        ax_disp.set_aspect("equal")
        ax_disp.set_xticks([])
        ax_disp.set_yticks([])
        ax_disp.set_xlim(my - (1.75 * wid), my + (1.75 * wid))
        ax_disp.set_ylim(mz - (1.75 * wid), mz + (1.75 * wid))

        # Console print
        print(f"R{i} → Mean: {mean_angle:.2f}°, Std: {std_angle:.2f}°")

    # ∆ Finalize and save both plots
    fig_disp.tight_layout(rect=[0, 0, 1, 0.95])
    fig_disp.savefig(f"_png/shear_displacement.png", dpi=500)

    fig_radar.tight_layout(rect=[0, 0, 1, 0.95])
    fig_radar.savefig(f"_png/shear_rotation_radar.png", dpi=500)

    plt.close('all')

# ∆ Plot displacements
def displacement_plot():

    # ∆ Prepare figure
    # fig, axes = plt.subplots(3, 6, figsize=(10, 5))
    # axes = axes.flatten()
    colors = sns.color_palette("tab20", 18)

    # ∆ Keep Control
    df = pd.read_csv(f"_csv/sim_test_20_200.csv")

    # ∆ Adding the displacement terms to find true height
    df["X_du"] = df["X"] + df["disp_x"]
    df["Y_du"] = df["Y"] + df["disp_y"]
    df["Z_du"] = df["Z"] + df["disp_z"]

    # ∆ Select for the edge of Y and along X values
    f_df = df[
        (df["Y"] < 0.05*EDGE[1]) | (df["Y"] > 0.95*EDGE[1]) |
        (df["Z"] < 0.05*EDGE[2]) | (df["Z"] > 0.95*EDGE[2])
    ].copy()

    # ∆ Group X
    cy0z1 = f_df[
        (f_df["Y"] < 0.05*EDGE[1]) & (df["Z"] > 0.999*EDGE[2])
    ].groupby("X")
    cy0z0 = f_df[
        (f_df["Y"] < 0.05*EDGE[1]) & (df["Z"] < 0.001*EDGE[2])
    ].groupby("X")
    cz0y1 = f_df[
        (f_df["Z"] < 0.05*EDGE[2]) & (df["Y"] > 0.999*EDGE[1])
    ].groupby("X")
    cz0y0 = f_df[
        (f_df["Z"] < 0.05*EDGE[2]) & (df["Y"] < 0.001*EDGE[1])
    ].groupby("X")

    # ∆ Collect movement and displacement value
    cy0z1_i = cy0z1["Z_du"].idxmax()
    cy0z1_d = df.loc[cy0z1_i, "disp_z"]
    cy0z0_i = cy0z0["Z_du"].idxmin()
    cy0z0_d = df.loc[cy0z0_i, "disp_z"]
    cz0y1_i = cz0y1["Y_du"].idxmax()
    cz0y1_d = df.loc[cz0y1_i, "disp_y"]
    cz0y0_i = cz0y0["Y_du"].idxmin()
    cz0y0_d = df.loc[cz0y0_i, "disp_y"]

    g_0, g_1 = 0.001, 0.999

    # # ∆ Loop axes and display control data
    # for i, ax in enumerate(axes):

    #     ax.set_title(f"R{i}", fontsize=12)

    #     # ∆ Keep Control
    #     if i == 11:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_300.csv")
    #     else:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_200.csv")
            

    #     # ∆ Adding the displacement terms to find true height
    #     df_i["X_du"] = df_i["X"] + df_i["disp_x"]
    #     df_i["Y_du"] = df_i["Y"] + df_i["disp_y"]
    #     df_i["Z_du"] = df_i["Z"] + df_i["disp_z"]

    #     # ∆ Select for the edge of Y and along X values
    #     f_df_i = df_i[
    #         (df_i["Y"] < 0.05*EDGE[1]) | (df_i["Y"] > 0.95*EDGE[1]) |
    #         (df_i["Z"] < 0.05*EDGE[2]) | (df_i["Z"] > 0.95*EDGE[2])
    #     ].copy()

    #     # ∆ Group X
    #     y0z1 = f_df_i[
    #         (f_df_i["Y"] < 0.33*EDGE[1]) & (f_df_i["Z"] > g_1*EDGE[2])
    #     ].groupby("X")
    #     y0z0 = f_df_i[
    #         (f_df_i["Y"] < 0.33*EDGE[1]) & (f_df_i["Z"] < g_0*EDGE[2])
    #     ].groupby("X")

    #     # ∆ Collect movement and displacement value
    #     y0z1_i = y0z1["Z_du"].idxmax()
    #     y0z1_d = df_i.loc[y0z1_i, "disp_z"]
    #     y0z0_i = y0z0["Z_du"].idxmin()
    #     y0z0_d = df_i.loc[y0z0_i, "disp_z"]

    #     # ∆ Plot test data
    #     ax.plot(y0z1.groups.keys(), y0z1_d, color=colors[i], alpha=1)
    #     ax.plot(y0z0.groups.keys(), y0z0_d, color=colors[i], alpha=1)

    #     # ∆ Group X
    #     ymz1 = f_df_i[
    #         (f_df_i["Y"] > 0.34*EDGE[1]) & (f_df_i["Y"] < 0.66*EDGE[1]) & (f_df_i["Z"] > g_1*EDGE[2])
    #     ].groupby("X")
    #     ymz0 = f_df_i[
    #         (f_df_i["Y"] > 0.34*EDGE[1]) & (f_df_i["Y"] < 0.66*EDGE[1]) & (f_df_i["Z"] < g_0*EDGE[2])
    #     ].groupby("X")

    #     # ∆ Collect movement and displacement value
    #     ymz1_i = ymz1["Z_du"].idxmax()
    #     ymz1_d = df_i.loc[ymz1_i, "disp_z"]
    #     ymz0_i = ymz0["Z_du"].idxmin()
    #     ymz0_d = df_i.loc[ymz0_i, "disp_z"]

    #     # ∆ Plot test data
    #     ax.plot(ymz1.groups.keys(), ymz1_d, color=colors[i], alpha=0.5)
    #     ax.plot(ymz0.groups.keys(), ymz0_d, color=colors[i], alpha=0.5)

    #     # ∆ Group X
    #     y1z1 = f_df_i[
    #         (f_df_i["Y"] > 0.67*EDGE[1]) & (f_df_i["Z"] > g_1*EDGE[2])
    #     ].groupby("X")
    #     y1z0 = f_df_i[
    #         (f_df_i["Y"] > 0.67*EDGE[1]) & (f_df_i["Z"] < g_0*EDGE[2])
    #     ].groupby("X")

    #     # ∆ Collect movement and displacement value
    #     y1z1_i = y1z1["Z_du"].idxmax()
    #     y1z1_d = df_i.loc[y1z1_i, "disp_z"]
    #     y1z0_i = y1z0["Z_du"].idxmin()
    #     y1z0_d = df_i.loc[y1z0_i, "disp_z"]

    #     # ∆ Plot test data
    #     ax.plot(y1z1.groups.keys(), y1z1_d, color=colors[i], alpha=0.20)
    #     ax.plot(y1z0.groups.keys(), y1z0_d, color=colors[i], alpha=0.20)

    #     ax.set_ylim(-700, 700)

    #     # ∆ Hide x-axis 
    #     if i < 12:
    #         ax.set_xticks([])

    #     if not(i in [0, 6, 12]):
    #         ax.set_yticks([])


    #     # ∆ Plot control envelop
    #     ax.plot(cy0z1.groups.keys(), cy0z1_d, color="black")
    #     ax.plot(cy0z0.groups.keys(), cy0z0_d, color="black")

    #     ax.set_xlabel("")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(f"_png/z_disp.png", dpi=500)
    # plt.close()

    # # ∆ Prepare figure
    # fig, axes = plt.subplots(3, 6, figsize=(10, 5))
    # axes = axes.flatten()
    # colors = sns.color_palette("tab20", len(axes))

    # # ∆ Loop axes and display control data
    # for i, ax in enumerate(axes):

    #     ax.set_title(f"R{i}", fontsize=12)

    #     # ∆ Keep Control
    #     if i == 11:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_300.csv")
    #     else:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_200.csv")

    #     # ∆ Adding the displacement terms to find true height
    #     df_i["X_du"] = df_i["X"] + df_i["disp_x"]
    #     df_i["Y_du"] = df_i["Y"] + df_i["disp_y"]
    #     df_i["Z_du"] = df_i["Z"] + df_i["disp_z"]

    #     # ∆ Select for the edge of Y and along X values
    #     f_df_i = df_i[
    #         (df_i["Y"] < 0.05*EDGE[1]) | (df_i["Y"] > 0.95*EDGE[1]) |
    #         (df_i["Z"] < 0.05*EDGE[2]) | (df_i["Z"] > 0.95*EDGE[2])
    #     ].copy()

    #     # ∆ Group X
    #     z0y1 = f_df_i[
    #         (f_df_i["Z"] < 0.33*EDGE[2]) & (f_df_i["Y"] > 0.999*EDGE[1])
    #     ].groupby("X")
    #     z0y0 = f_df_i[
    #         (f_df_i["Z"] < 0.33*EDGE[2]) & (f_df_i["Y"] < 0.001*EDGE[1])
    #     ].groupby("X")

    #     # # ∆ Collect movement and displacement value
    #     z0y1_i = z0y1["Y_du"].idxmax()
    #     z0y1_d = df_i.loc[z0y1_i, "disp_y"]
    #     z0y0_i = z0y0["Y_du"].idxmin()
    #     z0y0_d = df_i.loc[z0y0_i, "disp_y"]

    #     # ∆ Plot test data
    #     ax.plot(z0y1.groups.keys(), z0y1_d, color=colors[i])
    #     ax.plot(z0y0.groups.keys(), z0y0_d, color=colors[i])

    #     # ∆ Group X
    #     zmy1 = f_df_i[
    #         (f_df_i["Z"] > 0.34*EDGE[2]) & (f_df_i["Z"] < 0.66*EDGE[2]) & (f_df_i["Y"] > 0.999*EDGE[1])
    #     ].groupby("X")
    #     zmy0 = f_df_i[
    #         (f_df_i["Z"] > 0.34*EDGE[2]) & (f_df_i["Z"] < 0.66*EDGE[2]) & (f_df_i["Y"] < 0.001*EDGE[1])
    #     ].groupby("X")

    #     # # ∆ Collect movement and displacement value
    #     zmy1_i = zmy1["Y_du"].idxmax()
    #     zmy1_d = df_i.loc[zmy1_i, "disp_y"]
    #     zmy0_i = zmy0["Y_du"].idxmin()
    #     zmy0_d = df_i.loc[zmy0_i, "disp_y"]

    #     # ∆ Plot test data
    #     ax.plot(zmy1.groups.keys(), zmy1_d, color=colors[i], alpha=0.5)
    #     ax.plot(zmy0.groups.keys(), zmy0_d, color=colors[i], alpha=0.5)

    #     # ∆ Group X
    #     z1y1 = f_df_i[
    #         (f_df_i["Z"] > 0.67*EDGE[2]) & (f_df_i["Y"] > 0.999*EDGE[1])
    #     ].groupby("X")
    #     z1y0 = f_df_i[
    #         (f_df_i["Z"] > 0.67*EDGE[2]) & (f_df_i["Y"] < 0.001*EDGE[1])
    #     ].groupby("X")

    #     # # ∆ Collect movement and displacement value
    #     z1y1_i = z1y1["Y_du"].idxmax()
    #     z1y1_d = df_i.loc[z1y1_i, "disp_y"]
    #     z1y0_i = z1y0["Y_du"].idxmin()
    #     z1y0_d = df_i.loc[z1y0_i, "disp_y"]

    #     # ∆ Plot test data
    #     ax.plot(z1y1.groups.keys(), z1y1_d, color=colors[i], alpha=0.2)
    #     ax.plot(z1y0.groups.keys(), z1y0_d, color=colors[i], alpha=0.2)

    #     ax.set_ylim(-1200, 1200)

    #     # ∆ Hide x-axis 
    #     ax.set_xlabel("")

    #     # ∆ Hide x-axis 
    #     if i < 12:
    #         ax.set_xticks([])

    #     if not(i in [0, 6, 12]):
    #         ax.set_yticks([])

    #     # ∆ Plot control envelop
    #     ax.plot(cz0y1.groups.keys(), cz0y1_d, color="black")
    #     ax.plot(cz0y0.groups.keys(), cz0y0_d, color="black")

    #     ax.set_xlabel("")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(f"_png/y_disp.png", dpi=500)
    # plt.close()

    # # ∆ Prepare figure
    # fig, axes = plt.subplots(2, 2, figsize=(4.56, 4.29))
    # axes = axes.flatten()
    # # colors = sns.color_palette("tab20", len(axes))

    # select = [0, 4, 6, 14]

    # # ∆ Loop axes and display control data
    # for j, ax in enumerate(axes):

    #     i = select[j]

    #     ax.set_title(f"R{i}", fontsize=12)

    #     # ∆ Keep Control
    #     if i == 11:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_300.csv")
    #     else:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_200.csv")

    #     # ∆ Adding the displacement terms to find true height
    #     df_i["X_du"] = df_i["X"] + df_i["disp_x"]
    #     df_i["Y_du"] = df_i["Y"] + df_i["disp_y"]
    #     df_i["Z_du"] = df_i["Z"] + df_i["disp_z"]

    #     # ∆ Select for the edge of Y and along X values
    #     f_df_i = df_i[
    #         (df_i["Y"] < 0.05*EDGE[1]) | (df_i["Y"] > 0.95*EDGE[1]) |
    #         (df_i["Z"] < 0.05*EDGE[2]) | (df_i["Z"] > 0.95*EDGE[2])
    #     ].copy()

    #     # ∆ Group X
    #     z0y1 = f_df_i[
    #         (f_df_i["Z"] < 0.33*EDGE[2]) & (f_df_i["Y"] > 0.999*EDGE[1])
    #     ].groupby("X")
    #     z0y0 = f_df_i[
    #         (f_df_i["Z"] < 0.33*EDGE[2]) & (f_df_i["Y"] < 0.001*EDGE[1])
    #     ].groupby("X")

    #     # # ∆ Collect movement and displacement value
    #     z0y1_i = z0y1["Y_du"].idxmax()
    #     z0y1_d = df_i.loc[z0y1_i, "disp_y"]
    #     z0y0_i = z0y0["Y_du"].idxmin()
    #     z0y0_d = df_i.loc[z0y0_i, "disp_y"]

    #     # ∆ Plot test data
    #     ax.plot(z0y1.groups.keys(), z0y1_d, color=colors[i])
    #     ax.plot(z0y0.groups.keys(), z0y0_d, color=colors[i])

    #     # ∆ Group X
    #     zmy1 = f_df_i[
    #         (f_df_i["Z"] > 0.34*EDGE[2]) & (f_df_i["Z"] < 0.66*EDGE[2]) & (f_df_i["Y"] > 0.999*EDGE[1])
    #     ].groupby("X")
    #     zmy0 = f_df_i[
    #         (f_df_i["Z"] > 0.34*EDGE[2]) & (f_df_i["Z"] < 0.66*EDGE[2]) & (f_df_i["Y"] < 0.001*EDGE[1])
    #     ].groupby("X")

    #     # # ∆ Collect movement and displacement value
    #     zmy1_i = zmy1["Y_du"].idxmax()
    #     zmy1_d = df_i.loc[zmy1_i, "disp_y"]
    #     zmy0_i = zmy0["Y_du"].idxmin()
    #     zmy0_d = df_i.loc[zmy0_i, "disp_y"]

    #     # ∆ Plot test data
    #     ax.plot(zmy1.groups.keys(), zmy1_d, color=colors[i], alpha=0.5)
    #     ax.plot(zmy0.groups.keys(), zmy0_d, color=colors[i], alpha=0.5)

    #     # ∆ Group X
    #     z1y1 = f_df_i[
    #         (f_df_i["Z"] > 0.67*EDGE[2]) & (f_df_i["Y"] > 0.999*EDGE[1])
    #     ].groupby("X")
    #     z1y0 = f_df_i[
    #         (f_df_i["Z"] > 0.67*EDGE[2]) & (f_df_i["Y"] < 0.001*EDGE[1])
    #     ].groupby("X")

    #     # # ∆ Collect movement and displacement value
    #     z1y1_i = z1y1["Y_du"].idxmax()
    #     z1y1_d = df_i.loc[z1y1_i, "disp_y"]
    #     z1y0_i = z1y0["Y_du"].idxmin()
    #     z1y0_d = df_i.loc[z1y0_i, "disp_y"]

    #     # ∆ Plot test data
    #     ax.plot(z1y1.groups.keys(), z1y1_d, color=colors[i], alpha=0.2)
    #     ax.plot(z1y0.groups.keys(), z1y0_d, color=colors[i], alpha=0.2)

    #     ax.set_ylim(-1200, 1200)

    #     # ∆ Hide x-axis 
    #     ax.set_xlabel("")

    #     # ∆ Hide x-axis 
    #     if j < 2:
    #         ax.set_xticks([])

    #     if not(j in [0, 2]):
    #         ax.set_yticks([])

    #     # ∆ Plot control envelop
    #     ax.plot(cz0y1.groups.keys(), cz0y1_d, color="black")
    #     ax.plot(cz0y0.groups.keys(), cz0y0_d, color="black")

    #     ax.set_xlabel("")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(f"_png/y_disp_04614.png", dpi=1000)
    # plt.close()

    # fig, axes = plt.subplots(2, 2, figsize=(4.56, 4.29))
    # axes = axes.flatten()
    # # colors = sns.color_palette("tab20", len(axes))

    # select = [0, 4, 6, 14]

    # # ∆ Loop axes and display control data
    # for j, ax in enumerate(axes):

    #     i = select[j]

    #     ax.set_title(f"R{i}", fontsize=12)

    #     # ∆ Keep Control
    #     if i == 11:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_300.csv")
    #     else:
    #         df_i = pd.read_csv(f"_csv/sim_{i}_20_200.csv")

    #     # ∆ Adding the displacement terms to find true height
    #     df_i["X_du"] = df_i["X"] + df_i["disp_x"]
    #     df_i["Y_du"] = df_i["Y"] + df_i["disp_y"]
    #     df_i["Z_du"] = df_i["Z"] + df_i["disp_z"]

    #     # ∆ Select for the edge of Y and along X values
    #     f_df_i = df_i[
    #         (df_i["Y"] < 0.05*EDGE[1]) | (df_i["Y"] > 0.95*EDGE[1]) |
    #         (df_i["Z"] < 0.05*EDGE[2]) | (df_i["Z"] > 0.95*EDGE[2])
    #     ].copy()

    #     # ∆ Group X
    #     y0z1 = f_df_i[
    #         (f_df_i["Y"] < 0.33*EDGE[1]) & (f_df_i["Z"] > 0.999*EDGE[2])
    #     ].groupby("X")
    #     y0z0 = f_df_i[
    #         (f_df_i["Y"] < 0.33*EDGE[1]) & (f_df_i["Z"] < 0.001*EDGE[2])
    #     ].groupby("X")

    #     # ∆ Collect movement and displacement value
    #     y0z1_i = y0z1["Z_du"].idxmax()
    #     y0z1_d = df_i.loc[y0z1_i, "disp_z"]
    #     y0z0_i = y0z0["Z_du"].idxmin()
    #     y0z0_d = df_i.loc[y0z0_i, "disp_z"]

    #     # ∆ Plot test data
    #     ax.plot(y0z1.groups.keys(), y0z1_d, color=colors[i], alpha=1)
    #     ax.plot(y0z0.groups.keys(), y0z0_d, color=colors[i], alpha=1)

    #     # ∆ Group X
    #     ymz1 = f_df_i[
    #         (f_df_i["Y"] > 0.34*EDGE[1]) & (f_df_i["Y"] < 0.66*EDGE[1]) & (f_df_i["Z"] > 0.999*EDGE[2])
    #     ].groupby("X")
    #     ymz0 = f_df_i[
    #         (f_df_i["Y"] > 0.34*EDGE[1]) & (f_df_i["Y"] < 0.66*EDGE[1]) & (f_df_i["Z"] < 0.001*EDGE[2])
    #     ].groupby("X")

    #     # ∆ Collect movement and displacement value
    #     ymz1_i = ymz1["Z_du"].idxmax()
    #     ymz1_d = df_i.loc[ymz1_i, "disp_z"]
    #     ymz0_i = ymz0["Z_du"].idxmin()
    #     ymz0_d = df_i.loc[ymz0_i, "disp_z"]

    #     # ∆ Plot test data
    #     ax.plot(ymz1.groups.keys(), ymz1_d, color=colors[i], alpha=0.5)
    #     ax.plot(ymz0.groups.keys(), ymz0_d, color=colors[i], alpha=0.5)

    #     # ∆ Group X
    #     y1z1 = f_df_i[
    #         (f_df_i["Y"] > 0.67*EDGE[1]) & (f_df_i["Z"] > 0.999*EDGE[2])
    #     ].groupby("X")
    #     y1z0 = f_df_i[
    #         (f_df_i["Y"] > 0.67*EDGE[1]) & (f_df_i["Z"] < 0.001*EDGE[2])
    #     ].groupby("X")

    #     # ∆ Collect movement and displacement value
    #     y1z1_i = y1z1["Z_du"].idxmax()
    #     y1z1_d = df_i.loc[y1z1_i, "disp_z"]
    #     y1z0_i = y1z0["Z_du"].idxmin()
    #     y1z0_d = df_i.loc[y1z0_i, "disp_z"]

    #     # ∆ Plot test data
    #     ax.plot(y1z1.groups.keys(), y1z1_d, color=colors[i], alpha=0.20)
    #     ax.plot(y1z0.groups.keys(), y1z0_d, color=colors[i], alpha=0.20)

    #     ax.set_ylim(-700, 700)

    #     # ∆ Hide x-axis 
    #     if j < 2:
    #         ax.set_xticks([])

    #     if not(j in [0, 2]):
    #         ax.set_yticks([])


    #     # ∆ Plot control envelop
    #     ax.plot(cy0z1.groups.keys(), cy0z1_d, color="black")
    #     ax.plot(cy0z0.groups.keys(), cy0z0_d, color="black")

    #     ax.set_xlabel("")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig(f"_png/z_disp_04614.png", dpi=1000)
    # plt.close()

    # exit()

    def process_and_save_data(df_data, plot_filename, csv_filename):
        """
        Processes the DataFrame, plots a boxplot, and saves both the plot and
        a CSV with quartile and mean data.
        """
        # ∆ Plot boxplot with group colors
        plt.figure(figsize=(4.3, 5), dpi=500)
        colors = sns.color_palette("tab20", 18) + [(0, 0, 0)]
        # sns.boxplot(
        #     x='Displacement', y='Region', data=df_data, palette=colors
        # )
        palette = {f"R{i}": colors[i] for i in range(18)}
        sns.violinplot(data=df_data, x="Displacement", y="Region", hue="Region",
            palette=palette, dodge=False, inner="quart"
        )

        # Calculate statistics for the control group
        control_df = pd.read_csv(f"_csv/sim_test_20_200.csv")
        control_disps = control_df["disp_y"] if 'disp_y' in control_df.columns else control_df["disp_z"]
        q2 = np.quantile(control_disps, 0.5)
        q1 = np.quantile(control_disps, 0.25)
        q3 = np.quantile(control_disps, 0.75)
        # std_control = np.std(control_disps)

        # Add reference lines from control data
        plt.axvline(q2, color='black', linestyle='--', linewidth=1)
        plt.axvline(q1, color='black', linestyle=':', linewidth=1)
        plt.axvline(q3, color='black', linestyle=':', linewidth=1)
        
        plt.ylabel("")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()

        # ∆ Calculate and save quartile and mean data
        grouped_data = df_data.groupby('Region')['Displacement'].agg([
            'mean',
            'median',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75)
        ]).reset_index()

        grouped_data.columns = ['Region', 'Mean', 'Median', 'Q1', 'Q3']
        grouped_data.to_csv(csv_filename, index=False)

    # --- Process y-displacement data ---
    y_disp_data = []
    for i in range(18):
        file_path = f"_csv/sim_{i}_20_300.csv" if i == 11 else f"_csv/sim_{i}_20_200.csv"
        df = pd.read_csv(file_path)
        disps = df["disp_y"].dropna()
        group = "Anisotropic"
        for d in disps:
            y_disp_data.append({'Region': f'R{i}', 'Displacement': d, 'Group': group})

    y_data_df = pd.DataFrame(y_disp_data)
    process_and_save_data(y_data_df, f"_png/dV_Viplot.png", f"_csv/dV_Boxplot_data.csv")

    # --- Process z-displacement data ---
    z_disp_data = []
    for i in range(18):
        file_path = f"_csv/sim_{i}_20_300.csv" if i == 11 else f"_csv/sim_{i}_20_200.csv"
        df = pd.read_csv(file_path)
        disps = df["disp_z"].dropna()
        group = "Anisotropic"
        for d in disps:
            z_disp_data.append({'Region': f'R{i}', 'Displacement': d, 'Group': group})

    z_data_df = pd.DataFrame(z_disp_data)
    process_and_save_data(z_data_df, f"_png/dW_Viplot.png", f"_csv/dW_Boxplot_data.csv")

# ∆ Plot tensor bars 
def tensor_bar(tensor):

    # ∆ Define parameters
    disp = 20
    comps = ["xy", "xz", "yz"]
    tests = list(map(str, range(18)))
    colors = sns.color_palette("tab20", len(tests))
    bar_labels = [f"R{i}" for i in range(len(tests))]

    # ∆ Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.9), dpi=500)
    axes = axes.flatten()

    # ∆ Load and calculate test set statistics
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
        (df_test["X"].between(0.1 * EDGE[0], 0.9 * EDGE[0])) &
        (df_test["Y"].between(0.1 * EDGE[1], 0.9 * EDGE[1])) &
        (df_test["Z"].between(0.1 * EDGE[2], 0.9 * EDGE[2]))
    ]
    test_means = {}
    test_stds = {}
    for comp in comps:
        data = df_test_r[f"{tensor}_{comp}"]
        filtered = data[data.between(-50, 50)]
        test_means[comp] = filtered.mean()
        test_stds[comp] = filtered.std()

    # ∆ Iterate over components
    for cc, comp in enumerate(comps):
        mean_data = []
        std_data = []

        # ∆ Collect mean and standard deviation values per test
        for tt, t in enumerate(tests):
            # Load simulation data
            if t == "test" or t != "11":
                df = pd.read_csv(f"_csv/sim_{t}_{disp}_200.csv")
            else:
                df = pd.read_csv(f"_csv/sim_{t}_{disp}_300.csv")

            # Internal volume filter
            df_r = df[
                (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
                (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
                (df["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
            ]

            # Confidence interval filter
            data = df_r[f"{tensor}_{comp}"]
            filtered = data[data.between(-20, 20)]
            mean_data.append(filtered.mean())
            std_data.append(filtered.std())

        # ∆ Plot bar chart
        ax = axes[cc]
        ax.barh(bar_labels, mean_data, capsize=3, color=colors)
        ax.set_xlabel("Stress [kPa]" if tensor == "sig" else "Strain [%]", fontsize=8)
        ax.invert_yaxis()

        if cc != 0:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', length=0)

        if tensor == "sig":
            if cc == 0:
                ax.set_xticks(np.arange(-1.5, 1, 0.5))
            elif cc == 1:
                ax.set_xticks(np.arange(-1.5, 1.5, 0.5))
            elif cc == 2:
                ax.set_xticks(np.arange(-0.3, 0.3, 0.1))
        else:
            if cc == 0:
                ax.set_xticks(np.arange(-1.5, 0.5, 0.5))
            elif cc == 1:
                ax.set_xticks(np.arange(-1.5, 1.0, 0.5))
            elif cc == 2:
                ax.set_xticks(np.arange(-0.15, 0.1, 0.05))
        
        # ∆ Add a grid and set it behind the bars
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)

    # ∆ Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"_png/PUB_sigbar.png", bbox_inches="tight", dpi=500)
    plt.close()

# ∆ Plot all tensor
def tensor_plot(tensor):
    # sns.set_style("whitegrid")

    # ∆ Define spatial and test parameters
    disps = np.arange(0, 22, 2)
    regs = range(6)
    comps = ["xx", "yy", "zz", "xy", "xz", "yz"]
    # comp_titles = [
    #     "$\sigma_{xx}$", "$\sigma_{yy}$", "$\sigma_{zz}$", 
    #     "$\sigma_{xy}$", "$\sigma_{xz}$", "$\sigma_{yz}$"
    # ]
    tests = ["test"] + list(map(str, range(18)))
    colors = sns.color_palette("tab20", len(tests) - 1)

    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    axes = axes.flatten()

    summary_rows = []

    for cc, comp in enumerate(comps):
        all_vals = np.zeros((len(tests), len(disps)))

        for tt, t in enumerate(tests):
            for ii, disp in enumerate(disps):

                if t == "test" or t != "11":
                    df = pd.read_csv(f"_csv/sim_{t}_{disp}_200.csv")
                else:
                    df = pd.read_csv(f"_csv/sim_{t}_{disp}_300.csv")

                # ∆ Internal Volume filter
                df_r = df[
                    (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
                    (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
                    (df["Z"].between(0.05 * EDGE[2], 0.95 * EDGE[2]))
                ]
                within_ci = df_r[f"{tensor}_{comp}"].between(-50, 50)
                data = df_r.loc[within_ci, f"{tensor}_{comp}"]
                mean_val = data.mean() 

                if disp == 20:
                    # ∆ Save to summary
                    summary_rows.append({
                        "Test": t,
                        "Component": f"{tensor}_{comp}",
                        "Mean": np.mean(data.values),
                        "Std": np.std(data.values)
                    })

                all_vals[tt, ii] = mean_val

        # ∆ Plot each test
        ax = axes[cc]

        # ax.set_title(comp_titles[cc])
        ax.set_xlabel("Displacement [%]")
        ax.set_ylabel("Stress [kPa]" if tensor == "sig" else "Strain [%]")

        for i, t in enumerate(tests):
            means = all_vals[i]
            color = "black" if i == 0 else colors[i - 1]
            lw = 2
            style = "-" if i == 0 or int(t) in [0, 4, 6, 14] else "--"
            alpha = 1.0 if i == 0 or int(t) in [0, 4, 6, 14] else 0.5
            sns.lineplot(x=disps, y=means, ax=ax, color=color, linewidth=lw, linestyle=style, alpha=alpha, label=t)

        ax.set_xticks(np.arange(0, 24, 4))
        
        # ∆ Add a grid and set it behind the bars
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        ax.set_ylabel("")
        ax.set_xlabel("")

    # ∆ Titles and legend
    # plt.suptitle("Cauchy Stress ($\sigma$) Tensor" if tensor == "sig" else "Green Strain (E) Tensor")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.legend_:
            ax.legend_.remove()
    # fig.legend(handles, labels, bbox_to_anchor=(1, 0.95), loc='upper right', borderaxespad=0.)

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0.3, hspace=0.2)
    plt.savefig(f"_png/PUB_sigplot.png", bbox_inches="tight", dpi=500)
    plt.close()

    # ∆ Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"_csv/{tensor}_summary.csv", index=False)

def tensor_(tensor):
    disps = np.arange(0, 22, 2)
    regs = range(6)
    comps = ["xx", "yy", "zz", "xy", "xz", "yz"]
    tests = ["test"] + list(map(str, range(18)))
    colors = sns.color_palette("tab20", len(tests) - 1)
    bar_labels = [f"{i}" for i in range(len(tests) - 1)]  # exclude 'test'

    # Store means for both line and bar plots
    all_means = {comp: np.zeros((len(tests), len(disps))) for comp in comps}

    # --- Data collection loop ---
    for tt, t in enumerate(tests):
        for ii, disp in enumerate(disps):

            if t == "test" or t != "11":
                df = pd.read_csv(f"_csv/sim_{t}_{disp}_200.csv")
            else:
                df = pd.read_csv(f"_csv/sim_{t}_{disp}_300.csv")

            # Internal volume filter
            df_r = df[
                (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
                (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
                (df["Z"].between(0.05 * EDGE[2], 0.95 * EDGE[2]))
            ]

            for comp in comps:
                within_ci = df_r[f"{tensor}_{comp}"].between(-50, 50)
                data = df_r.loc[within_ci, f"{tensor}_{comp}"]
                all_means[comp][tt, ii] = data.mean()

    # --- Figure 1: Full line plots ---
    fig1, axes1 = plt.subplots(6, 1, figsize=(1.96, 9.84), sharex=True, gridspec_kw={"hspace": 0.2})
    for cc, comp in enumerate(comps):
        ax = axes1[cc]
        for i, t in enumerate(tests):
            means = all_means[comp][i]
            color = "black" if i == 0 else colors[i - 1]
            lw = 2
            style = "-" if i == 0 or (t.isdigit() and int(t) in [0, 4, 6, 14]) else "--"
            alpha = 1.0 if i == 0 or (t.isdigit() and int(t) in [0, 4, 6, 14]) else 0.5
            sns.lineplot(x=disps, y=means, ax=ax, color=color, linewidth=lw, linestyle=style, alpha=alpha)

        # ax.set_ylabel(comp, fontsize=8)
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.linspace(ymin*0.9, ymax*1.1, 5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xticks(np.arange(0, 24, 4))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)
        ax.tick_params(axis='both', labelsize=10)

    # axes1[-1].set_xlabel("Displacement [%]")
    plt.tight_layout()
    plt.savefig(f"_png/{tensor}_line_full.png", bbox_inches="tight", dpi=500)
    plt.close(fig1)

    # --- Figure 2: Bar plots ---
    fig2, axes2 = plt.subplots(6, 1, figsize=(3, 9.84), sharex=False, gridspec_kw={"hspace": 0.2})
    for cc, comp in enumerate(comps):
        mean_data = all_means[comp][1:, -1]  # skip 'test', last disp=20%
        axes2[cc].bar(bar_labels, mean_data, color=colors)
        # axes2[cc].invert_yaxis()
        # axes2[cc].set_ylabel(comp, fontsize=8)
        axes2[cc].grid(True, linestyle='--', linewidth=0.5, alpha=1, zorder=0)
        axes2[cc].tick_params(axis='y', labelsize=10)
        # axes2[cc].tick_params(axis='x', labelsize=8)
        # if cc != 5:
        axes2[cc].set_xticklabels([])
        axes2[cc].tick_params(axis='x', which='both', length=0)
        ymin, ymax = axes2[cc].get_ylim()
        axes2[cc].set_yticks(np.linspace(ymin*0.9, ymax*1.1, 5))
        axes2[cc].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axes2[cc].axhline(all_means[comp][0, -1], color="black", linestyle="--")

    # axes2[-1].set_xlabel("Stress [kPa]" if tensor == "sig" else "Strain [%]")
    
    plt.tight_layout()
    plt.savefig(f"_png/{tensor}_bar.png", bbox_inches="tight", dpi=500)
    plt.close(fig2)

    # # --- Figure 3: Zoomed line plots (last two disps) ---
    # zoom_idx = [-2, -1]  # last two displacement points
    # fig3, axes3 = plt.subplots(6, 1, figsize=(0.35, 9.84),  sharex=True, gridspec_kw={"hspace": 0.2})
    # mx, mn = [], []
    # for cc, comp in enumerate(comps):
    #     ax = axes3[cc]
    #     for i, t in enumerate(tests):
    #         means = all_means[comp][i, zoom_idx]
    #         color = "black" if i == 0 else colors[i - 1]
    #         lw = 2
    #         style = "-" if i == 0 or (t.isdigit() and int(t) in [0, 4, 6, 14]) else "--"
    #         alpha = 1.0 if i == 0 or (t.isdigit() and int(t) in [0, 4, 6, 14]) else 0.8
            
    #         sns.lineplot(x=disps[zoom_idx], y=means, ax=ax, color=color, linewidth=lw, linestyle=style, alpha=alpha)


    #     ymin, ymax = ax.get_ylim()
    #     ax.set_yticks(np.linspace(ymin, ymax, 5))
    #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #     ax.yaxis.tick_right()
    #     ax.yaxis.set_label_position("right")    
    #     # ax.set_xlim([np.min(means)-0.1*np.min(means), np.max(means)+0.1*np.max(means)])
    #     # ax.set_xlim([19.5, 20.5])
    #     ax.set_ylabel(comp, fontsize=10)
    #     ax.set_xticks([20])
    #     ax.grid(True, linestyle='--', linewidth=0.5, alpha=1, zorder=0)
    #     # ax.tick_params(axis='both', labelsize=8)

    # # axes3[-1].set_xlabel("Displacement [%]")
    # plt.tight_layout()
    # plt.savefig(f"_png/{tensor}_line_zoom.png", bbox_inches="tight", dpi=500)
    # plt.close(fig3)

# ∆ Stress trend data
def stress_trend(key):

    # ∆ Settings
    sns.set_style("whitegrid")

    # ∆ Determine points within the 3D space
    xx = np.flip(np.linspace(0.05 * EDGE[0], EDGE[0] - 0.05 * EDGE[0], 5))
    yy = np.linspace(0.05 * EDGE[1], EDGE[1] - 0.05 * EDGE[1], 5)
    zz = np.linspace(0.05 * EDGE[2], EDGE[2] - 0.05 * EDGE[2], 5)

    # ∆ Tests 
    tests = ["test", "0"] #+ [str(x) for x in range(0, 18, 1)]
    disps, regs = list(range(0, 22, 2)), list(range(0, 6, 1))
    sig_dict = {t: {f"sample_{s}": [] for s in regs} for t in tests}
    all_vals = np.zeros((len(tests), len(regs), len(disps)))

    # ∆ Iterate tests 
    for tt, t in enumerate(tests):

        # ∆ Iterate displacements
        for ii, i in enumerate(disps): 

            collate = []

            # ∆ Load data
            df = pd.read_csv(f"_csv/sim_{t}_{i}_200.csv")

            # ∆ Internal Volume Calculation
            df_r = df[(
                (df["X"] >= xx[-1]) & (df["X"] <= xx[0]) & 
                (df["Y"] >= yy[0]) & (df["Y"] <= yy[-1]) & 
                (df["Z"] >= zz[0]) & (df["Z"] <= zz[-1]) 
            )]
            val = df_r.loc[:, key].mean()
            sig_dict[t][f"sample_{5}"].append(val)
            collate.append(val)

            # ∆ Iterate regions
            for v_idx, (xp, yp, zp) in enumerate(zip(xx, yy, zz)):

                # ∆ Create inclusion
                df_r = df[
                    (df["X"] >= xp - 400) & (df["X"] <= xp + 400) &
                    (df["Y"] >= yp - 400) & (df["Y"] <= yp + 400) &
                    (df["Z"] >= zp - 400) & (df["Z"] <= zp + 400)
                ]
                val = df_r.loc[:, key].mean()
                sig_dict[t][f"sample_{v_idx}"].append(val)
                collate.append(val)

            # ∆ Store all the required values
            all_vals[tt, :, ii] = collate

    # ∆ Prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=300)
    axes = axes.flatten()

    # Initialize variables to store handles and labels for the legend
    handles, labels = [], []
    colors = sns.color_palette("tab20", len(tests))

    for r_idx in range(6):
        ax = axes[r_idx]
        for i, t in enumerate(tests):
            vals = sig_dict[t][f"R_{r_idx}"]
            sns.lineplot(x=list(range(0, 20, 2)), y=vals, ax=ax, color=colors[i], label=f"{t}") # Keep label for now, will get handles/labels later

        ax.set_title(f"Region {r_idx}")
        ax.set_xlabel("Strain [%]")
        ax.set_ylabel("Stress [kPa]")

    # ∆ Create a single legend after all plots have been made
    handles, labels = axes[0].get_legend_handles_labels()

    # Remove the individual legends from the subplots
    for ax in axes:
        if ax.legend_ is not None:
            ax.legend_.remove()

    # Place the single legend on the figure
    fig.legend(handles, labels, title="Test", bbox_to_anchor=(1.02, 0.95), loc='upper left', borderaxespad=0.)

    # ∆ Final plot details
    fig.suptitle(f"{key}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.98, 1]) # Adjust layout to make space for the legend
    plt.savefig(f"_png/{key}.png", bbox_inches="tight")
    plt.close()

# ∆ Stress violin plot
def stress_violin(key, alpha=0.05):

    # ∆ Determine central points of 5 cubic regions (excluding region 5)
    xx = np.linspace(500, EDGE[0] - 500, 5)
    yy = np.linspace(500, EDGE[1] - 500, 5)
    zz = np.linspace(500, EDGE[2] - 500, 5)
    regs = list(zip(xx, yy, zz))[:5]

    # ∆ Load control values and apply 90% CI filter
    control_vals = []
    df_control = pd.read_csv(f"_csv/sim_test_18_300.csv")

    for (xp, yp, zp) in regs:
        df_r = df_control[
            (df_control["X"] >= xp - 400) & (df_control["X"] <= xp + 400) &
            (df_control["Y"] >= yp - 400) & (df_control["Y"] <= yp + 400) &
            (df_control["Z"] >= zp - 400) & (df_control["Z"] <= zp + 400)
        ]
        control_vals.extend(df_r[key].dropna().tolist())

    control_vals = np.array(control_vals)
    low, high = np.percentile(control_vals, [5, 95])
    control_vals = control_vals[(control_vals >= low) & (control_vals <= high)]

    # ∆ Load test values with 90% CI filtering
    data_records = []
    test_ids = []

    for t in range(18):  # tests 0–17
        df = pd.read_csv(f"_csv/sim_{t}_18_300.csv")
        test_label = str(t)
        test_ids.append(test_label)

        test_vals = []
        for (xp, yp, zp) in regs:
            df_r = df[
                (df["X"] >= xp - 400) & (df["X"] <= xp + 400) &
                (df["Y"] >= yp - 400) & (df["Y"] <= yp + 400) &
                (df["Z"] >= zp - 400) & (df["Z"] <= zp + 400)
            ]
            test_vals.extend(df_r[key].dropna().tolist())

        test_vals = np.array(test_vals)
        if len(test_vals) > 0:
            lo, hi = np.percentile(test_vals, [5, 95])
            test_vals = test_vals[(test_vals >= lo) & (test_vals <= hi)]

        for val in test_vals:
            data_records.append({
                "Test": test_label,
                "Stress": val
            })

    # ∆ Violin plot
    df_plot = pd.DataFrame(data_records)
    plt.figure(figsize=(14, 6), dpi=300)
    sns.set_style("whitegrid")
    ax = sns.violinplot(x="Test", y="Stress", data=df_plot, inner="box", palette="Set2")
    ax.set_title(f"Stress Distribution per Test (90% CI) - {key}")
    ax.set_ylabel("Stress [kPa]")
    plt.tight_layout()
    plt.savefig(f"_png/{key}_violin_90CI.png")
    plt.close()

    # ∆ Statistical comparison (Mann-Whitney U and KS only)
    p_mwu, p_ks = [], []

    for t in test_ids:
        test_vals = df_plot[df_plot["Test"] == t]["Stress"].values

        # Mann-Whitney U
        _, p1 = mannwhitneyu(test_vals, control_vals, alternative='two-sided')
        # KS test
        _, p2 = ks_2samp(test_vals, control_vals)

        p_mwu.append(p1)
        p_ks.append(p2)

    df_pvals = pd.DataFrame({
        "Test_ID": test_ids,
        "Mann_Whitney_U_p_value": p_mwu,
        "KS_p_value": p_ks
    })
    df_pvals.to_csv(f"_csv/{key}_p_values_90CI.csv", index=False)
    print(f"P-values saved to _csv/{key}_p_values_90CI.csv")

    # ∆ Bar plot of p-values
    x = np.arange(len(test_ids))
    width = 0.3

    plt.figure(figsize=(14, 6), dpi=300)
    bars1 = plt.bar(x - width / 2, p_mwu, width=width, label='Mann–Whitney U', color="#7EDAFF", edgecolor="black")
    bars2 = plt.bar(x + width / 2, p_ks, width=width, label='K–S test', color="#A98BFF", edgecolor="black")

    # Annotate significant tests
    for i in range(len(test_ids)):
        if p_mwu[i] < alpha:
            plt.text(x[i] - width / 2, 0.06, "*", ha='center', va='bottom', fontsize=20)
        if p_ks[i] < alpha:
            plt.text(x[i] + width / 2, 0.06, "*", ha='center', va='bottom', fontsize=20)

    plt.axhline(alpha, color='black', linestyle='--', label=f'p = {alpha}')
    plt.xticks(x, test_ids, rotation=45)
    plt.ylabel("p-value")
    plt.title(f"Statistical Comparison vs Control (90% CI) - {key}")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"_png/{key}_stat_compare_90CI.png")
    plt.close()

# ∆ Main
def main():

    # stress_trend("sig_xx")
    # stress_violin("sig_xz")

    # tensor_plot("sig")
    # tensor_bar("sig")
    # tensor_("sig")

    # displacement_plot()
    # shear_plot()
    quiver_plot()

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Open main
    main()
