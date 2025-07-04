"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _rawMorphologicalStats.py
        Output csv data with key morphological data from zdiscs 
"""

# ∆ Raw
import math
import ast
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA

# ∆ Constants
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]
EXCLUSION = [335, 653, 775, 1108, 1406] 
CUBE = {"x": 1000, "y": 1000, "z": 100}

# ∆ Rot set
def rot_data(data, norm_df):

    # ∆ Dummy slice for calculating shape data
    dummy = np.ones_like(data)
    h, w, d = dummy.shape
    cx, cy = (w-1)/2, (h-1)/2
    dummy = []

    # ∆ Create a new copy to store rotated centroids
    rot_df = norm_df.copy(deep=True)
    new_cents = []

    # ∆ Iterate data and tile
    for _, row in norm_df.iterrows():

        # ∆ Load centroid data
        ori = np.array(ast.literal_eval(row["Centroid"]))
        vx, vy, vz = ori

        # ∆ Rotate data to determine if within desired region
        s_vx = vx - cx
        s_vy = vy - cy
        n45 = -np.pi/4
        rot = np.array([
            [np.cos(n45), -np.sin(n45)],
            [np.sin(n45),  np.cos(n45)]
        ])
        s_rx, s_ry = np.dot(rot, [s_vx, s_vy])
        rx = s_rx + M_D
        ry = s_ry + M_D

        # ∆ New centroid
        new_cents.append([float(rx), float(ry), float(vz)])

    rot_df["Centroid"] = new_cents
    rot_df.to_csv(f"_csv/rot_norm.csv", index=False)

# ∆ Tile data 
def tile_data(data, norm_df):

    # ∆ Dummy slice for calculating shape data
    dummy = np.ones_like(data)
    h, w, d = dummy.shape
    cx, cy = (w-1)/2, (h-1)/2
    dummy = []

    # ∆ Load region data
    reg_df = pd.read_csv("_csv/reg_.csv")

    # ∆ Iterate regions
    for i, r in reg_df.iterrows():
        
        r_id = []
        # ∆ Iterate data and tile
        for _, row in norm_df.iterrows():

            id = row["ID"]
            # ∆ test for exclusion group
            if id in EXCLUSION: continue

            # ∆ Load centroid data
            ori = np.array(ast.literal_eval(row["Centroid"]))
            vx, vy, vz = ori

            # ∆ Rotate data to determine if within desired region
            s_vx = vx - cx
            s_vy = vy - cy
            n45 = -np.pi/4
            rot = np.array([
                [np.cos(n45), -np.sin(n45)],
                [np.sin(n45),  np.cos(n45)]
            ])
            s_rx, s_ry = np.dot(rot, [s_vx, s_vy])
            rx = s_rx + M_D
            ry = s_ry + M_D
            ori = np.array([rx, ry])

            # ∆ Check if within region
            if not np.all([
                (rx <= r["x"] + CUBE["x"] and rx >= r["x"]),
                (ry <= r["y"] + CUBE["y"] and ry >= r["y"]),
                (vz <= r["z"] + CUBE["z"] and vz >= r["z"]),
            ]):
                continue

            # ∆ Save IDs that pass
            print(id, rx, ry)
            print("stop")
            r_id.append(id)

        tile_df = norm_df[norm_df["ID"].isin(r_id)].copy()
        tile_df.to_csv(f"_csv/tile_{i}.csv", index=False)
        
# ∆ Validate data
def validate_data(data, norm_df, iso_df):

    # # ∆ Setup visualisation
    # fig = plt.figure(figsize=(12, 4))
    # ax1 = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)
    # sns.set_palette("deep")
    # colors = ["blue", "green", "red"]

    # ∆ Dummy slice for calculating shape data
    dummy = np.ones_like(data)
    h, w, d = dummy.shape
    cx, cy = (w-1)/2, (h-1)/2
    dummy = []

    # # ∆ Iterate original data
    # for _, row in iso_df.iterrows():

    #     if row["ID"] in EXCLUSION: continue

    #     # ∆ Load centroid data
    #     ori = np.array(ast.literal_eval(row["Centroid"]))

    #     # ∆ Load principal componenets
    #     pcs = np.array((
    #         list(map(float, row["PC1"].strip('[]').split())), 
    #         list(map(float, row["PC2"].strip('[]').split())), 
    #         list(map(float, row["PC3"].strip('[]').split()))
    #     ))

    #     # ∆ Display the principal components
    #     for i, vec in enumerate(pcs):
    #         v = vec * 20 
    #         ini = ori - v
    #         end = ori + v
    #         ax1.plot(
    #             [ini[0], end[0]], [ini[1], end[1]],
    #             color=colors[i], linewidth=1
    #     )
    
    # # ∆ Iterate data and normalise
    # for _, row in norm_df.iterrows():

    #     if row["ID"] in EXCLUSION: continue

    #     # ∆ Load centroid data
    #     ori = np.array(ast.literal_eval(row["Centroid"]))
    #     vx, vy, vz = ori

    #     # ∆ Rotate data to determine if wihtin desired region
    #     s_vx = vx - cx
    #     s_vy = vy - cy
    #     n45 = -np.pi/4
    #     rot = np.array([
    #         [np.cos(n45), -np.sin(n45)],
    #         [np.sin(n45),  np.cos(n45)]
    #     ])
    #     s_rx, s_ry = np.dot(rot, [s_vx, s_vy])
    #     rx = s_rx + M_D
    #     ry = s_ry + M_D
    #     ori = np.array([rx, ry])

    #     # ∆ Load principal componenets
    #     pcs = np.array((
    #         list(map(float, row["PC1_ROT"].strip('[]').split())), 
    #         list(map(float, row["PC2_ROT"].strip('[]').split())), 
    #         list(map(float, row["PC3_ROT"].strip('[]').split()))
    #     ))

    #     # ∆ Display the principal components
    #     for i, vec in enumerate(pcs):
    #         v = vec * 20 
    #         ini = ori - v[:2]
    #         end = ori + v[:2]
    #         ax2.plot(
    #             [ini[0], end[0]], [ini[1], end[1]],
    #             color=colors[i], linewidth=1
    #     )
            
    # # ∆ Create steps for tiling
    # x_step = CUBE["x"] - 200
    # y_step = CUBE["y"]
    # sns_cmap = sns.color_palette("deep", n_colors=6)

    # # ∆ Iterate tile regions
    # for idx_i, i in enumerate([0, 1]):
    #     for idx_j, j in enumerate([0, 1, 2]):

    #         # ∆ Plot tiles 
    #         x = Y_BOUNDS[0] + i * x_step
    #         y = X_BOUNDS[0] + j * y_step
    #         color = sns_cmap[idx_i * 3 + idx_j]
    #         rec = patches.Rectangle((x, y), CUBE["x"], CUBE["y"], facecolor=color, edgecolor="black", alpha=0.6)
    #         ax3.add_patch(rec)

    # # ∆ Format and save
    # ax1.set_xlabel('X [nm]')
    # ax1.set_ylabel('Y [nm]')
    # ax1.set_title("Raw Principal Components")
    # ax2.set_xlabel('X [nm]')
    # ax2.set_ylabel('Y [nm]')
    # ax2.set_title("Normalised Principal Components")
    # ax3.set_xlabel('X [nm]')
    # ax3.set_ylabel('Y [nm]')
    # ax3.set_title("Simulation Regions")
    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')
    # ax3.set_aspect('equal')
    # ax2.set_xlim(1700, 3700)
    # ax2.set_ylim(900, 4200)
    # ax3.set_xlim(1700, 3700)
    # ax3.set_ylim(900, 4200)
    # plt.savefig(f"_png/ROT_PCA.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
    # plt.close()

    # # ∆ Setup visualisation
    # fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 9))

    # # ∆ Iterate tile regions
    # k = [0,1,2] * 6
    # j = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]

    # # ∆ Load region data
    # reg_df = pd.read_csv("_csv/reg_.csv")
    
    # for i, r in reg_df.iterrows():

    #     # ∆ Axis setup
    #     ax = axes[k[i], j[i]]
    #     ax.set_xlim(r["x"], r["x"]+CUBE["x"])
    #     ax.set_ylim(r["y"], r["y"]+CUBE["y"])
    #     ax.set_xlabel('X [nm]')
    #     ax.set_ylabel('Y [nm]')
    #     ax.set_title("Simulation Regions")
    #     ax.set_aspect('equal')
    #     ax.set_title(f'z: {r["z"]}, x: {r["x"]}, y: {r["y"]}')

    #     # ∆ Iterate data and normalise
    #     for _, row in norm_df.iterrows():

    #         if row["ID"] in EXCLUSION: continue

    #         # ∆ Load centroid data
    #         ori = np.array(ast.literal_eval(row["Centroid"]))
    #         vx, vy, vz = ori

    #         # ∆ Rotate data to determine if wihtin desired region
    #         s_vx = vx - cx
    #         s_vy = vy - cy
    #         n45 = -np.pi/4
    #         rot = np.array([
    #             [np.cos(n45), -np.sin(n45)],
    #             [np.sin(n45),  np.cos(n45)]
    #         ])
    #         s_rx, s_ry = np.dot(rot, [s_vx, s_vy])
    #         rx = s_rx + M_D
    #         ry = s_ry + M_D
    #         ori = np.array([rx, ry])

    #         # ∆ Check if within region
    #         if (rx > r["x"] + CUBE["x"]) or (rx < r["x"]): continue
    #         if (ry > r["y"] + CUBE["y"]) or (ry < r["y"]): continue
    #         if (vz > r["z"] + CUBE["z"]) or (vz < r["z"]): continue
            

    #         # ∆ Load principal componenets
    #         pcs = np.array((
    #             list(map(float, row["PC1_ROT"].strip('[]').split())), 
    #             list(map(float, row["PC2_ROT"].strip('[]').split())), 
    #             list(map(float, row["PC3_ROT"].strip('[]').split()))
    #         ))

    #         # ∆ Display the principal components
    #         for i, vec in enumerate(pcs):
    #             v = vec * 20 
    #             ini = ori - v[:2]
    #             end = ori + v[:2]
    #             ax.plot(
    #                 [ini[0], end[0]], [ini[1], end[1]],
    #                 color=colors[i], linewidth=1
    #         )

    # plt.tight_layout()
    # plt.savefig(f"_png/TILES_PCA.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
    # plt.close()

    # ∆ Setup visualisation with seaborn style
    sns.set_style("whitegrid")
    # ∆ Load region data
    try:
        reg_df = pd.read_csv("_csv/reg_.csv")
    except FileNotFoundError:
        print("Error: '_csv/reg_.csv' not found. Please ensure the file exists.")
        # Create dummy data for reg_df for demonstration if file not found
        reg_df = pd.DataFrame({
            'z': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], # Example with 18 regions
            'x': [0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4],
            'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        })
        print(f"Using dummy 'reg_df' with {len(reg_df)} regions for demonstration.")


    # ∆ Load norm data (assuming norm_df exists from previous context)
    try:
        if 'norm_df' not in locals():
            norm_df = pd.DataFrame({
                'Ele_[RAD]': np.random.rand(1000) * 2 * np.pi,
                'ID': np.arange(1000),
                'Centroid': [f'[{np.random.rand()*5},{np.random.rand()*5},{np.random.rand()*5}]' for _ in range(1000)]
            })
            print("Using dummy 'norm_df' for demonstration. Ensure your actual 'norm_df' is loaded.")
    except Exception as e:
        print(f"Error loading or accessing 'norm_df': {e}. Using dummy 'norm_df'.")
        norm_df = pd.DataFrame({
            'Ele_[RAD]': np.random.rand(1000) * 2 * np.pi,
            'ID': np.arange(1000),
            'Centroid': [f'[{np.random.rand()*5},{np.random.rand()*5},{np.random.rand()*5}]' for _ in range(1000)]
        })

    # --- Dynamic Subplot Grid Setup ---
    num_regions = len(reg_df)
    # Calculate a nearly square grid for subplots
    ncols = 6 # Keeping 6 columns as in your original setup for consistency
    nrows = math.ceil(num_regions / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False) # Use squeeze=False for consistent axes indexing

    # ∆ Define common bins for all histograms
    num_bins = 20 # You can adjust this number as needed
    min_azi = norm_df['Ele_[RAD]'].min()
    max_azi = norm_df['Ele_[RAD]'].max()
    common_bins = np.linspace(min_azi, max_azi, num_bins + 1) # +1 for bin edges

    # Define a single color for all region histograms
    single_color = '#FAA52B' # Or 'lightcoral', 'teal', 'mediumseagreen', etc.

    # ∆ Iterate through each region and plot its histogram on a dedicated subplot
    for i, r in reg_df.iterrows():
        row_idx = i // ncols
        col_idx = i % ncols
        ax = axes[row_idx, col_idx]

        ang_reg_df = pd.DataFrame()
        for _, row in norm_df.iterrows():

            if row["ID"] in EXCLUSION:
                continue

            # ∆ Load centroid data
            try:
                ori = np.array(ast.literal_eval(row["Centroid"]))
            except (ValueError, SyntaxError) as e:
                # print(f"Warning: Could not parse Centroid '{row['Centroid']}'. Skipping row. Error: {e}")
                continue

            if len(ori) != 3:
                # print(f"Warning: Centroid '{row['Centroid']}' does not have 3 dimensions. Skipping row.")
                continue

            vx, vy, vz = ori

            # ∆ Rotate data to determine if within desired region
            s_vx = vx - cx
            s_vy = vy - cy
            n45 = -np.pi / 4
            rot = np.array([
                [np.cos(n45), -np.sin(n45)],
                [np.sin(n45), np.cos(n45)]
            ])
            s_rx, s_ry = np.dot(rot, [s_vx, s_vy])
            rx = s_rx + M_D
            ry = s_ry + M_D

            # ∆ Check if within region based on transformed coordinates and z
            if (rx <= r["x"] + CUBE["x"] and rx >= r["x"]) and \
            (ry <= r["y"] + CUBE["y"] and ry >= r["y"]) and \
            (vz <= r["z"] + CUBE["z"] and vz >= r["z"]):
                ang_reg_df = pd.concat([ang_reg_df, pd.DataFrame({'Ele_[RAD]': [row["Ele_[RAD]"]]})], ignore_index=True)

        # ∆ Plot the histogram for the current region
        if not ang_reg_df.empty:
            sns.histplot(ang_reg_df['Ele_[RAD]'], ax=ax, stat='density', edgecolor='black', alpha=0.7, bins=common_bins,
                        color=single_color)
        else:
            ax.text(0.5, 0.5, 'No data in this region', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
            print(f"No data found for region: z={r['z']}, x={r['x']}, y={r['y']}")


        # ∆ Axis and title setup for each subplot
        ax.set_xlabel('Angle [rad]' if row_idx == nrows - 1 else '') # Only label x-axis on bottom row
        ax.set_ylabel('Proportion' if col_idx == 0 else '') # Only label y-axis on left-most column
        ax.set_title(f'Region: z={r["z"]}, x={r["x"]}, y={r["y"]}', fontsize=10)


    # ∆ Hide any unused subplots
    for i in range(num_regions, nrows * ncols):
        row_idx = i // ncols
        col_idx = i % ncols
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle if needed
    fig.suptitle('Ele_[RAD] Distribution by Region (Uniform Bins)', fontsize=16, y=1.02) # Overall title


    plt.savefig(f"_png/SINGLE_HIST_AZI_OVERLAY_UNIFORM_BINS.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
    plt.close()

# ∆ Standardise data and convert into Degrees
def norm_data(zst_df):

    # ∆ Create properties dictionary
    props_dict = {
        "ID": [],
        "Pixels": [],
        "ZDiscs": [],
        "Centroid": [],
        "Azi_[RAD]": [],
        "Ele_[RAD]": [],
        "Sph_[RAD]": [],
        "Azi_[DEG]": [],
        "Ele_[DEG]": [],
        "Sph_[DEG]": [],
        "PC1_ROT": [],
        "PC2_ROT": [],
        "PC3_ROT": []
    }

    # ∆ Create mean dictionary
    mean_dict = {
        "Total_IDs": [len(zst_df["ID"])],
        "Total_ZDiscs": [sum(zst_df["ZDiscs"])],
        "mu_Azi_[RAD]": [np.mean(zst_df["Azi_[RAD]"])],
        "mu_Ele_[RAD]": [np.mean(zst_df["Ele_[RAD]"])],
        "mu_Sph_[RAD]": [np.mean(zst_df["Sph_[RAD]"])],
        "mu_Azi_[DEG]": [np.rad2deg(np.mean(zst_df["Azi_[RAD]"]))],
        "mu_Ele_[DEG]": [np.rad2deg(np.mean(zst_df["Ele_[RAD]"]))],
        "mu_Sph_[DEG]": [np.rad2deg(np.mean(zst_df["Sph_[RAD]"]))],
        "sig_Azi_[RAD]": [np.std(zst_df["Azi_[RAD]"])],
        "sig_Ele_[RAD]": [np.std(zst_df["Ele_[RAD]"])],
        "sig_Sph_[RAD]": [np.std(zst_df["Sph_[RAD]"])],
        "sig_Azi_[DEG]": [np.rad2deg(np.std(zst_df["Azi_[RAD]"]))],
        "sig_Ele_[DEG]": [np.rad2deg(np.std(zst_df["Ele_[RAD]"]))],
        "sig_Sph_[DEG]": [np.rad2deg(np.std(zst_df["Sph_[RAD]"]))]
    }

    # ∆ Iterate data and normalise
    for _, row in zst_df.iterrows():
        props_dict["ID"].append(row["ID"])
        props_dict["Pixels"].append(row["Pixels"])
        props_dict["ZDiscs"].append(row["ZDiscs"])
        props_dict["Centroid"].append(row["Centroid"])

        # ∆ Mean shift the Azi, Ele, and Sph values
        azi = row["Azi_[RAD]"]
        ele = row["Ele_[RAD]"]
        props_dict["Azi_[RAD]"].append(azi - mean_dict["mu_Azi_[RAD]"][0])
        props_dict["Ele_[RAD]"].append(ele - mean_dict["mu_Ele_[RAD]"][0])
        props_dict["Sph_[RAD]"].append(row["Sph_[RAD]"] - mean_dict["mu_Sph_[RAD]"][0])
        props_dict["Azi_[DEG]"].append(np.rad2deg(row["Azi_[RAD]"]) - mean_dict["mu_Azi_[DEG]"][0])
        props_dict["Ele_[DEG]"].append(np.rad2deg(row["Ele_[RAD]"]) - mean_dict["mu_Ele_[DEG]"][0])
        props_dict["Sph_[DEG]"].append(np.rad2deg(row["Sph_[RAD]"]) - mean_dict["mu_Sph_[DEG]"][0])

        # ∆ Rotate components
        # µ rotation about z, anti-, align with x
        ang = -azi if azi < 0 else azi
        azi_rot = np.array([
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang), np.cos(ang), 0],
            [0, 0, 1]
        ])
        # µ rotation about y, anti-, align with y
        ang = ele if ele < 0 else -ele
        ele_rot = np.array([
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-np.sin(ang), 0, np.cos(ang)]
        ])
        # µ rotation matrix
        pc1, pc2, pc3 = (
            list(map(float, row["PC1"].strip('[]').split())), 
            list(map(float, row["PC2"].strip('[]').split())), 
            list(map(float, row["PC3"].strip('[]').split()))
        )
        pcs = np.array([pc1, pc2, pc3])
        rot = azi_rot @ ele_rot
        rot_man = (rot @ pcs.T).T
        props_dict["PC1_ROT"].append(rot_man[0])
        props_dict["PC2_ROT"].append(rot_man[1])
        props_dict["PC3_ROT"].append(rot_man[2])

    # ∆ Save
    df = pd.DataFrame(props_dict)
    df.to_csv("_csv/norm_stats.csv", index=False)
    mean_df = pd.DataFrame(mean_dict)
    mean_df.to_csv("_csv/mean_stats.csv", index=False)

# ∆ Deep segmentation morphological data
def deep_stats(data, iso_df):

    # ∆ Create 3D property data
    props = measure.regionprops(data)
    props_dict = {
        "ID": [],
        "Pixels": [],
        "ZDiscs": [],
        "Centroid": [],
        "PC1": [],
        "PC2": [],
        "PC3": [],
        "Azi_[RAD]": [],
        "Ele_[RAD]": [],
        "Sph_[RAD]": []
    }
    inc_labs = iso_df["ID"].to_numpy()

    # ∆ Loop labeled data
    for lab in props:

        # ∆ Apply properties to dictionary
        # µ Label data
        val = lab.label
        if val not in inc_labs:
            continue
        if lab.area < Z_DISC:
            continue
        props_dict["ID"].append(val)
        # µ Numpy of pixels 
        props_dict["Pixels"].append(lab.num_pixels)
        # µ Ratio of pixels to Z-Disc average volume
        props_dict["ZDiscs"].append(lab.num_pixels // Z_DISC)
        # µ Centroid data
        vx, vy, vz = list(map(float, lab.centroid))
        props_dict["Centroid"].append([vx, vy, vz])
        
        # ∆ Pixel indexes
        idxs = np.argwhere(data == val)
        mu_idxs = idxs * np.array([PIXX, PIXY, PIXZ])

        # ∆ Principal Components Analysis
        pca = PCA(n_components=3)
        pca.fit(mu_idxs)
        
        # ∆ Attach principal components
        props_dict["PC1"].append(pca.components_[0])
        props_dict["PC2"].append(pca.components_[1])
        props_dict["PC3"].append(pca.components_[2])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mu_idxs[::20, 0], mu_idxs[::20, 1], mu_idxs[::20, 2], alpha=0.3, label="raw")

        colors = ['green', 'blue', 'red']  
        labels = ['PC1', 'PC2', 'PC3']
        origin = pca.mean_

        for i, (length, vector) in enumerate(zip(pca.explained_variance_, pca.components_)):
            v = vector * np.sqrt(length) * 3  # scale for visibility
            start = origin - v
            end = origin + v
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=colors[i],
                    linewidth=3,
                    label=labels[i])
            
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        plt.savefig(f"_png/PCA_Raw.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
        plt.close()
        exit()

        # ∆ Find angles from third component
        np90 = np.pi/2
        p3 = pca.components_[2]
        px, py, pz = p3
        # µ azimuthial (angle of rotation about z)
        azi = np.arctan2(py, px if px > 0 else -px)
        props_dict["Azi_[RAD]"].append(np90+(-azi if azi < 0 else azi))
        # µ elevation (angle of rotation about y)
        ele = np.arctan2(pz, px if px > 0 else -px)
        props_dict["Ele_[RAD]"].append(ele if ele < 0 else -ele)
        # µ spherical angle
        sph = np.arccos(np.abs(px))
        props_dict["Sph_[RAD]"].append(sph)

    # ∆ Save
    df = pd.DataFrame(props_dict)
    df.to_csv("_csv/iso_stats.csv", index=False)

    return df

# ∆ Isolate Segmentations
def isolate_segs(data):

    # ∆ Dummy slice for calculating shape data
    dummy = np.ones_like(data[:, :, 0])
    h, w = dummy.shape
    cx, cy = (w-1)/2, (h-1)/2
    dummy = []

    # ∆ Create 3D property data
    props = measure.regionprops(data)
    props_dict = {
        "ID": [],
        "Centroid": []
    }

    # ∆ Loop labeled data
    for lab in props:

        # ∆ Apply properties to dictionary
        # µ Label data
        val = lab.label
        if lab.area < Z_DISC:
            continue
        # µ Centroid data
        vx, vy, vz = list(map(float, lab.centroid))

        # ∆ Rotate data to determine if wihtin desired region
        s_vx = vx - cx
        s_vy = vy - cy
        n45 = -np.pi/4
        rot = np.array([
            [np.cos(n45), -np.sin(n45)],
            [np.sin(n45),  np.cos(n45)]
        ])
        s_rx, s_ry = np.dot(rot, [s_vx, s_vy])
        rx = s_rx + M_D
        ry = s_ry + M_D

        # ∆ Check region
        if ((rx < Y_BOUNDS[1]) & (rx > Y_BOUNDS[0])) & ((ry < X_BOUNDS[1]) & (ry > X_BOUNDS[0])):
            props_dict["ID"].append(val)
            props_dict["Centroid"].append([vx, vy, vz])

    # ∆ Save
    df = pd.DataFrame(props_dict)
    df.to_csv("_csv/reg_stats.csv", index=False)
    return df
            
# ∆ Main
def main(seg_np):

    # ∆ Isolate values
    # iso_df = isolate_segs(seg_np)

    # # ∆ Determine statistical data
    reg_df = pd.read_csv("_csv/reg_stats.csv")
    # deep_stats(seg_np, reg_df)

    # # ∆ Standardise
    iso_df = pd.read_csv("_csv/iso_stats.csv")
    # norm_data(iso_df)

    # ∆ Validation plot
    norm_df = pd.read_csv("_csv/norm_stats.csv")
    # validate_data(seg_np, norm_df, iso_df)

    # ∆ Save a rotated dataset
    rot_data(seg_np, norm_df)

    # ∆ Tile dataset
    # tile_data(seg_np, norm_df)

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Load data
    # path = os.path.join(os.path.dirname(__file__), RAW_SEG)
    # seg_np = np.load(path).astype(np.uint16)
    seg_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)

    # ∆ Open main
    main(seg_np)



    