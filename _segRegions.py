"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _segRegions.py
        Determine regions and datasets
"""

# ∆ Raw
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage
import matplotlib.cm as cm
import multiprocessing as mp
import matplotlib.pyplot as plt

# ∆ Constants
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]

# ∆ Region thumbs
def thumbs(raw_np, seg_np):

    # ∆ Generate figure
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), dpi=300)
    axes = axes.flatten()

    # ∆ Load region start-points
    iso_df = pd.read_csv("_csv/reg_.csv")

    # ∆ Iterate through regions
    for i, row in iso_df.iterrows():

        r_raw_np = ndimage.rotate(
            seg_np[:, :, row["z"]+50], -45, reshape=True, order=1
        )[row["x"]:row["x"]+1000, row["y"]:row["y"]+1000]

        # ∆ Apply to axes
        axes[i].imshow(r_raw_np, cmap='gray')
        axes[i].set_title(f"Region {i}")
        axes[i].set_axis_off() 

    plt.savefig(f"_png/seg_grid.png", bbox_inches='tight', pad_inches=0.2, dpi=500)
    plt.close()
    
    # ∆ Iterate through regions
    for i, row in iso_df.iterrows():

        r_raw_np = ndimage.rotate(
            seg_np[:, :, row["z"]+50], -45, reshape=True, order=1
        )[row["x"]:row["x"]+1000, row["y"]:row["y"]+1000]

        # ∆ Apply to axes
        axes[i].imshow(r_raw_np, cmap='gray')
        axes[i].set_title(f"Region {i}")
        axes[i].set_axis_off() 

    # ∆ Generate figure
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), dpi=500)
    axes = axes.flatten()

    # ∆ Load region start-points
    iso_df = pd.read_csv("_csv/reg_.csv")

    for i, row in iso_df.iterrows():
        # ∆ Extract and rotate slice
        slice_ = seg_np[:, :, row["z"]+50]
        r_seg_np = ndimage.rotate(slice_, -45, reshape=True, order=0)
        r_seg_np = r_seg_np[row["x"]:row["x"]+1000, row["y"]:row["y"]+1000]

        rgb = np.ones((*r_seg_np.shape, 3))  
        rgb[r_seg_np > 0] = [0, 0, 1]       

        # ∆ Apply to axes
        axes[i].imshow(rgb)
        axes[i].set_title(f"R{i}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.savefig(f"_png/seg_grid.png", bbox_inches='tight', pad_inches=0.2, dpi=500)
    plt.close()


# ∆ Processing stages
def process_layer(args):
    seg_np, l = args
    r_seg_np = ndimage.rotate(seg_np[:, :, l], -45, reshape=True, order=1)[:, Y_BOUNDS[0]:Y_BOUNDS[1]]
    ids = np.unique(r_seg_np).tolist()
    return ids

# ∆ Determine instances
def instance_id(seg_np):
    print("\t∫ instance classify segmentation")

    # ∆ Data presets
    x, y, z = seg_np.shape

    # ∆ Enter parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_layer, [(seg_np, l) for l in range(0, z, 1)])

    # ∆ Collect results
    labs = []
    for ids in results:
        labs.extend(ids)

    # ∆ Write data to file
    lab_set = list(set(labs))
    print(len(lab_set))
    with open('_txt/labels.txt', 'w') as f:
        for lab in lab_set:
            f.write(f"{lab}\n")

# ∆ Region images
def reg_images(raw_np, seg_np):
    print("\t∫ create images")

    # ∆ Data presets
    x, y, z = raw_np.shape
    layers = list(map(int, np.linspace(0, z-1, 10)))
    iso_df = pd.read_csv("_csv/reg_stats.csv")
    ids = iso_df["ID"].to_list()

    # ∆ Iterate layers
    for l in [50, 100, 150, 200, 250]:

        # ∆ Rotate data
        r_raw_np = ndimage.rotate(raw_np[:, :, l], -45, reshape=True, order=1, cval=np.mean(raw_np[:, :, l]))
        r_seg_np = ndimage.rotate(seg_np[:, :, l], -45, reshape=True, order=1)

        min_val = np.min(r_raw_np)
        max_val = np.max(r_raw_np)

        if (max_val - min_val) == 0: 
            normalized_raw_np = np.zeros_like(r_raw_np, dtype=float) 
        else:
            normalized_raw_np = (r_raw_np - min_val) / (max_val - min_val)

        rgba_image = np.stack([normalized_raw_np, normalized_raw_np, normalized_raw_np, np.ones_like(normalized_raw_np)], axis=-1)
        red_intensity = 1.0  
        alpha_overlay = 0.5  
        mask = r_seg_np > 0

        rgba_image[mask, 0] = red_intensity  
        rgba_image[mask, 1] = 0.0            
        rgba_image[mask, 2] = 0.0            
        rgba_image[mask, 3] = alpha_overlay   

        # ∆ Display images
        plt.figure(figsize=(8, 8)) 
        plt.ylim(Y_BOUNDS[0], Y_BOUNDS[1])
        plt.xlim(X_BOUNDS[0], X_BOUNDS[1])
        plt.axis('off')
        plt.imshow(rgba_image, origin='lower') 
        plt.savefig(f"_png/ralreg_{l}.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
        plt.close()


        # ∆ Rotate data
        r_raw_np = ndimage.rotate(raw_np[:, :, l], -45, reshape=True, order=1, cval=np.mean(raw_np[:, :, l]))
        r_seg_np = ndimage.rotate(seg_np[:, :, l], -45, reshape=True, order=1)

        # ∆ Permute data
        for n in ids:
            r_raw_np[r_seg_np > 0] = 10000
        plt.ylim(Y_BOUNDS[0], Y_BOUNDS[1])
        plt.xlim(X_BOUNDS[0], X_BOUNDS[1])
        plt.axis('off')
        plt.imshow(r_raw_np, cmap="gray", origin='lower')
        plt.savefig(f"_png/ralreg_{l}.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
        plt.close()

# ∆ Main function
def main():

    # ∆ Load data
    raw_np = np.load("big_xyz_100_3700_100_3700_40_339.npy").astype(np.uint16)
    seg_np = np.load("filtered.npy").astype(np.uint16)

    # ∆ Images
    reg_images(raw_np, seg_np)

    # ∆ Instance identify
    instance_id(seg_np)

    # ∆ Output region snapshots
    thumbs(raw_np, seg_np)


# ∆ Initiate
if __name__ == "__main__":
    main()