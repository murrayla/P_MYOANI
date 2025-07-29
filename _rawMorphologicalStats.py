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
EXCLUSION = [335, 653, 775, 1108, 1406, 42, 185, 191, 335, 653, 775, 1108, 1406, 1674, 44, 136, 1652, 1732, 1744]
CUBE = {"x": 1000, "y": 1000, "z": 100}

# ∆ Rot set
def rot_data(data, norm_df):

    # ∆ Dummy slice for calculating shape data
    dummy = np.ones_like(data)
    h, w, d = dummy.shape
    cx, cy = (w-1)/2, (h-1)/2
    dummy = []

    norm_df = norm_df[~norm_df['ID'].isin(EXCLUSION)]

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
    rot_df.to_csv(f"_csv/rot_norm_w.csv", index=False)

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
        tile_df.to_csv(f"_csv/tile_{i}_w.csv", index=False)
        
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

    ids_7 = pd.read_csv("_csv/tile_7_w.png")["ID"].to_numpy()

    # ∆ Loop labeled data
    for lab in props:

        # ∆ Apply properties to dictionary
        # µ Label data
        val = lab.label
        # if val not in inc_labs:
        #     continue
        if val not in ids_7:
            continue
        # if lab.area < Z_DISC:
        #     continue

        if val != 1456:
            continue
        
        # props_dict["ID"].append(val)
        # # µ Numpy of pixels 
        # props_dict["Pixels"].append(lab.num_pixels)
        # # µ Ratio of pixels to Z-Disc average volume
        # props_dict["ZDiscs"].append(lab.num_pixels // Z_DISC)
        # # µ Centroid data
        # vx, vy, vz = list(map(float, lab.centroid))
        # props_dict["Centroid"].append([vx, vy, vz])
        
        # ∆ Pixel indexes
        idxs = np.argwhere(data == val)
        mu_idxs = idxs * np.array([PIXX, PIXY, PIXZ])

        # ∆ Principal Components Analysis
        pca = PCA(n_components=3)
        pca.fit(mu_idxs)
        
        # ∆ Attach principal components
        # props_dict["PC1"].append(pca.components_[0])
        # props_dict["PC2"].append(pca.components_[1])
        # props_dict["PC3"].append(pca.components_[2])

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
        ax.set_xlabel('X [pxls]')
        ax.set_ylabel('Y [pxls]')
        ax.set_zlabel('Z [pxls]')
        ax.set_title(f"Z-Disc: {val}")
        plt.show()
        plt.savefig(f"_png/PCA_Raw.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
        plt.close()
        exit()

        # # ∆ Find angles from third component
        # np90 = np.pi/2
        # p3 = pca.components_[2]
        # px, py, pz = p3
        # # µ azimuthial (angle of rotation about z)
        # azi = np.arctan2(py, px if px > 0 else -px)
        # props_dict["Azi_[RAD]"].append(np90+(-azi if azi < 0 else azi))
        # # µ elevation (angle of rotation about y)
        # ele = np.arctan2(pz, px if px > 0 else -px)
        # props_dict["Ele_[RAD]"].append(ele if ele < 0 else -ele)
        # # µ spherical angle
        # sph = np.arccos(np.abs(px))
        # props_dict["Sph_[RAD]"].append(sph)

    # ∆ Save
    # df = pd.DataFrame(props_dict)
    # df.to_csv("_csv/iso_stats.csv", index=False)

    # return df


# ∆ Plot the PCs 
def pc_seg(data):

    # ∆ Dummy slice for calculating shape data
    dummy = np.ones_like(data)
    h, w, d = dummy.shape
    cx, cy = (w-1)/2, (h-1)/2
    dummy = []

    # ∆ Read in required IDs to plot
    reg_df = pd.read_csv("_csv/tile_7_w.csv")
    reg_id = reg_df["ID"].to_numpy()

    # # ∆ Setup plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # ∆ Formatting 
    # colors = ["#A98BFF", "#FAA52B", "#7EDAFF"]
    # labels = ['PC1', 'PC2', 'PC3']

    # # ∆ Iterate IDs
    # for j, val in enumerate(reg_id):

    #     # ∆ Pixel indexes
    #     idxs = np.argwhere(data == val)
    #     mu_idxs = idxs * np.array([PIXX, PIXY, PIXZ])
    #     # ax.scatter(mu_idxs[::100, 0], mu_idxs[::100, 1], mu_idxs[::100, 2], alpha=0.5, color="red")

    #     vx, vy, vz = np.mean(mu_idxs[:, 0]), np.mean(mu_idxs[:, 1]), np.mean(mu_idxs[:, 2])

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
    #     ori = np.array([rx, ry, vz])

    #     scatter_color = plt.cm.viridis(j / len(reg_id))
    #     rot_idxs = np.concatenate([(rot @ mu_idxs[:, :2].T).T, mu_idxs[:, 2][:, np.newaxis]], axis=1)
    #     ax.scatter(rot_idxs[::100, 0], rot_idxs[::100, 1], rot_idxs[::100, 2], alpha=0.1, color="black")

    #     # ∆ Principal Components Analysis
    #     pca = PCA(n_components=3)
    #     pca.fit(rot_idxs)
    #     origin = pca.mean_

    #     # ∆ plot components
    #     for i, (length, vector) in enumerate(zip(pca.explained_variance_, pca.components_)):
    #         if i == 0:
    #             l = np.sqrt(length) * 2
    #         if i != 2:
    #             continue
    #         v = vector * l
    #         start = origin - v
    #         end = origin + v
    #         if not j:
    #             ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
    #                     color=colors[i], linewidth=5, label=labels[i], alpha=1)
    #         else:
    #             ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
    #                     color=colors[i], linewidth=5, alpha=1)
                
    #     # if not j: break
                
    # ax.legend()
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title("Region 7 [PCA]")
    # ax.view_init(elev=20, azim=-60)
    # plt.show()
    # plt.savefig(f"_png/reg_pca.png", bbox_inches='tight', pad_inches=0.2, dpi=500)
    # plt.close()

    # ∆ Setup plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # ∆ Formatting 
    colors = ["#A98BFF", "#FAA52B", "#7EDAFF"]
    labels = ['PC1', 'PC2', 'PC3']

    # ∆ Iterate IDs
    for j, val in enumerate(reg_id):

        # ∆ Pixel indexes
        idxs = np.argwhere(data == val)
        mu_idxs = idxs * np.array([PIXX, PIXY, PIXZ])
        # ax.scatter(mu_idxs[::100, 0], mu_idxs[::100, 1], mu_idxs[::100, 2], alpha=0.5, color="red")

        # ∆ Rotate data to determine if wihtin desired region
        n45 = -np.pi/4
        rot = np.array([
            [np.cos(n45), -np.sin(n45)],
            [np.sin(n45),  np.cos(n45)]
        ])

        rot_idxs = np.concatenate([(rot @ mu_idxs[:, :2].T).T, mu_idxs[:, 2][:, np.newaxis]], axis=1)
        ax.scatter(rot_idxs[::100, 0], rot_idxs[::100, 1], alpha=0.1, color="black")

        # ∆ Principal Components Analysis
        pca = PCA(n_components=3)
        pca.fit(rot_idxs)
        origin = pca.mean_

        # ∆ plot components
        for i, (length, vector) in enumerate(zip(pca.explained_variance_, pca.components_)):
            if i == 0:
                l = np.sqrt(length) * 2
            if i != 2:
                continue
            # Normalize and scale to uniform length
            unit_vector = vector / np.linalg.norm(vector)
            scaled_vector = unit_vector * 1000

            # Quiver plot from the origin
            ax.quiver(origin[0], origin[1], scaled_vector[0], scaled_vector[1],
                    angles='xy', scale_units='xy', scale=1,
                    color=colors[i], label=labels[i], linewidth=10)
            # v = vector * l
            # start = origin - v
            # end = origin + v
            # ax.plot([start[0], end[0]], [start[1], end[1]],
            #         color=colors[i], linewidth=5, label=labels[i], alpha=1)
            # ax.quiver(start[0], start[1], end[0], end[1], color=colors[i], linewidth=5, edgecolor='black', scale=100)

        # break
                
    ax.set_title("Region 7 [PCA]")
    ax.set_axis_off()
    plt.savefig(f"_png/reg_pca_2D.png", bbox_inches='tight', pad_inches=0.2, dpi=500)
    plt.close()


"""ES-BB6F4BFF8125"""

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
    # reg_df = pd.read_csv("_csv/reg_stats_whole.csv")
    # deep_stats(seg_np, reg_df)

    # # ∆ Standardise
    # iso_df = pd.read_csv("_csv/iso_stats.csv")
    # norm_data(iso_df)

    # ∆ Validation plot
    # norm_df = pd.read_csv("_csv/norm_stats.csv")
    # validate_data(seg_np, norm_df, iso_df)

    # ∆ Save a rotated dataset
    # rot_data(seg_np, norm_df)

    # ∆ Tile dataset
    # tile_data(seg_np, norm_df)

    # ∆ Display Segs and PCs
    pc_seg(seg_np)

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Load data
    seg_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)[:2000, 1000:4000, 90:210]

    # ∆ Open main
    main(seg_np)



    