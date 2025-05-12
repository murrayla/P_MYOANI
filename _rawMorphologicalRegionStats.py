"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _rawMorphologicalStats.py
        Output csv data with key morphological data from zdiscs 
"""

# ∆ Raw
import os
import random
import numpy as np
import pandas as pd
from skimage import measure
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# ∆ Constants
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
# X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]

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
        "PC1": [],
        "PC2": [],
        "PC3": []
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
        props_dict["PC1"].append(row["PC1"])
        props_dict["PC2"].append(row["PC2"])
        props_dict["PC3"].append(row["PC3"])

        # ∆ Mean shift the Azi, Ele, and Sph values
        props_dict["Azi_[RAD]"].append(row["Azi_[RAD]"] - mean_dict["mu_Azi_[RAD]"][0])
        props_dict["Ele_[RAD]"].append(row["Ele_[RAD]"] - mean_dict["mu_Ele_[RAD]"][0])
        props_dict["Sph_[RAD]"].append(row["Sph_[RAD]"] - mean_dict["mu_Sph_[RAD]"][0])
        props_dict["Azi_[DEG]"].append(np.rad2deg(row["Azi_[RAD]"]) - mean_dict["mu_Azi_[DEG]"][0])
        props_dict["Ele_[DEG]"].append(np.rad2deg(row["Ele_[RAD]"]) - mean_dict["mu_Ele_[DEG]"][0])
        props_dict["Sph_[DEG]"].append(np.rad2deg(row["Sph_[RAD]"]) - mean_dict["mu_Sph_[DEG]"][0])

    # ∆ Save
    df = pd.DataFrame(props_dict)
    df.to_csv("norm_stats_nuc.csv", index=False)
    mean_df = pd.DataFrame(mean_dict)
    mean_df.to_csv("mean_stats_nuc.csv", index=False)

# ∆ Deep segmentation morphological data
def deep_stats(data, iso_df):

    # ∆ Create 3D property data
    props = measure.regionprops(data)
    props_dict = {
        "ID": [],
        "Pixels": [],
        "ZDiscs": [],
        "Centroid": [],
        "Azi_[RAD]": [],
        "Ele_[RAD]": [],
        "Sph_[RAD]": [],
        "PC1": [],
        "PC2": [],
        "PC3": []
    }

    # ∆ Loop labeled data
    for lab in props:

        # ∆ Apply properties to dictionary
        # µ Label data
        val = lab.label
        if val not in iso_df["ID"]:
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

        # ∆ Compute relevant angles
        p = pca.components_[0]
        px, py, pz = p
        # if px < 0:
        #     px = -px

        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(mu_idxs[::20, 0], mu_idxs[::20, 1], mu_idxs[::20, 2], alpha=0.3)

        # # Plot full-length principal axes
        # origin = pca.mean_
        # for length, vector in zip(pca.explained_variance_, pca.components_):
        #     v = vector * np.sqrt(length) * 3  # scale for visibility
        #     start = origin - v
        #     end = origin + v
        #     ax.plot([start[0], end[0]],
        #             [start[1], end[1]],
        #             [start[2], end[2]],
        #             color='red', linewidth=3)

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.title('PCA')
        # plt.show()

        props_dict["PC1"].append([px, py, pz])
        props_dict["PC2"].append(pca.components_[1])
        props_dict["PC3"].append(pca.components_[2])

        px, py, pz = pca.components_[2]
        if px < 0:
            px = -px
        props_dict["Azi_[RAD]"].append(np.arctan2(py, px))
        props_dict["Ele_[RAD]"].append(np.arctan2(vz, np.sqrt(px**2 + py**2)))
        props_dict["Sph_[RAD]"].append(np.arccos(px))

    # ∆ Save
    df = pd.DataFrame(props_dict)
    df.to_csv("iso_stats_nuc.csv", index=False)

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
        if ((rx < Y_BOUNDS[1]) & (rx > Y_BOUNDS[0])) & ((ry > 3000)):
            props_dict["ID"].append(val)
            props_dict["Centroid"].append([vx, vy, vz])

    # ∆ Save
    df = pd.DataFrame(props_dict)
    df.to_csv("reg_stats_nuc.csv", index=False)
    return df
            
# ∆ Main
def main(seg_np):

    # ∆ Isolate values
    iso_df = isolate_segs(seg_np)

    # ∆ Determine statistical data
    # iso_df = pd.read_csv("/Users/murrayla/Documents/main_PhD/P_MYOANI/_csv/reg_stats_nuc.csv")
    zst_df = deep_stats(seg_np, iso_df)

    # zst_df = pd.read_csv("/Users/murrayla/Documents/main_PhD/P_MYOANI/_csv/iso_stats_nuc.csv")
    # ∆ Standardise
    norm_data(zst_df)

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Load data
    # path = os.path.join(os.path.dirname(__file__), RAW_SEG)
    # seg_np = np.load(path).astype(np.uint16)
    seg_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)

    # ∆ Open main
    main(seg_np)



    