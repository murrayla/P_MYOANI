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
import matplotlib.pyplot as plt

# ∆ Constants
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [2000, 3500]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]

# ∆ Determine instances
def instance_id(seg_np):
    print("\t∫ instance classify segmentation")

    # ∆ Data presets
    x, y, z = seg_np.shape

    # ∆ Iterate layers
    labs = []
    print(f"dims: {x}, {y}, {z}")
    for l in range(0, z, 1):

        # ∆ Rotate data
        r_seg_np = ndimage.rotate(seg_np[:, :, l], -45, reshape=True)[:, Y_BOUNDS[0]:Y_BOUNDS[1]]

        # ∆ Determine unique data
        ids = np.unique(r_seg_np).tolist()
        labs.extend(ids)

    # ∆ Write data to file
    lab_set = list(set(labs))
    with open('_txt/labels.txt', 'w') as f:
        for lab in lab_set:
            f.write(f"{lab}\n")

# ∆ Region images
def reg_images(raw_np, seg_np):
    print("\t∫ create images")

    # ∆ Data presets
    x, y, z = raw_np.shape
    layers = list(map(int, np.linspace(0, z-1, 10)))

    # ∆ Iterate layers
    for l in layers:

        # ∆ Rotate data
        r_raw_np = ndimage.rotate(raw_np[:, :, l], -45, reshape=True)
        r_seg_np = ndimage.rotate(seg_np[:, :, l], -45, reshape=True)

        # ∆ Permute data
        r_raw_np[r_seg_np >= 1] = 255
        r_raw_np[:Y_BOUNDS[0], :] = 255
        r_raw_np[Y_BOUNDS[1]:, :] = 255
        r_raw_np[:, :X_BOUNDS[0]] = 255
        r_raw_np[:, X_BOUNDS[1]:] = 255

        # ∆ Display images
        plt.gca().set_xticks(np.arange(0, int(2545.58*2), 500))
        plt.gca().set_yticks(np.arange(0, int(2545.58*2), 500))
        plt.grid()
        plt.imshow(r_raw_np, cmap="gray", origin="lower")
        plt.savefig(f"_png/rot_{l}.png", bbox_inches='tight', pad_inches=0.2, dpi=1000)
        plt.close()

# ∆ Main function
def main():

    # ∆ Load data
    raw_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/big_xyz_100_3700_100_3700_40_339.npy").astype(np.uint16)
    seg_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)
    # ∆ Load data
    # if "main_PhD" in os.path.dirname(__file__):
    # seg_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)
    # else:
    # path = os.path.join(os.path.dirname(__file__), "filtered.npy")
    # seg_np = np.load(path).astype(np.uint16)

    # ∆ Images
    reg_images(raw_np, seg_np)

    # ∆ Instance identify
    # instance_id(seg_np)


# ∆ Initiate
if __name__ == "__main__":
    main()