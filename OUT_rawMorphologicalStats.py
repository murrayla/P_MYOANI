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
    df.to_csv("_csv/OUT_ns.csv", index=False)
    mean_df = pd.DataFrame(mean_dict)
    mean_df.to_csv("_csv/OUT_ms.csv", index=False)

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
        if val not in inc_labs: continue
        
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

        props_dict["PC1"].append(pca.components_[0])
        props_dict["PC2"].append(pca.components_[1])
        props_dict["PC3"].append(pca.components_[2])

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

    df = pd.DataFrame(props_dict)
    df.to_csv("_csv/OUT_is.csv", index=False)

    return df

# ∆ Isolate Segmentations
def isolate_segs(data):

    print("entered")

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

    ids = pd.read_csv("_csv/rot_norm_whole.csv")["ID"].values

    # ∆ Loop labeled data
    print("start loop")
    for lab in props:

        # ∆ Apply properties to dictionary
        # µ Label data
        val = lab.label
        if val in ids: continue
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
        props_dict["ID"].append(val)
        props_dict["Centroid"].append([vx, vy, vz])

    # ∆ Save
    df = pd.DataFrame(props_dict)
    df.to_csv("_csv/OUT_rs.csv", index=False)
    print("end loop")
    return df
            
# ∆ Main
def main(seg_np):

    # ∆ Isolate values
    iso_df = isolate_segs(seg_np)

    # # # ∆ Determine statistical data
    reg_df = pd.read_csv("_csv/OUT_rs.csv")
    deep_stats(seg_np, reg_df)

    # # # ∆ Standardise
    iso_df = pd.read_csv("_csv/OUT_is.csv")
    norm_data(iso_df)

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Load data
    seg_np = np.load("_npy/filtered.npy").astype(np.uint16)

    # ∆ Open main
    main(seg_np)



    