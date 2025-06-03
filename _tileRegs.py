"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _tileRegs.py
        Returns numpy arrays for each of the key regions of interest
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
CUBE = {"x": 1000, "y": 1000, "z": 100}
ROT_45 = np.array([
    [np.cos(-np.pi/4), -np.sin(-np.pi/4), 0],
    [np.sin(-np.pi/4),  np.cos(-np.pi/4), 0],
    [0, 0, 1]
])

def main():

    # ∆ Load region data
    reg_df = pd.read_csv("_csv/reg_.csv")

    # ∆ Load segmentation data
    seg_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)

    # ∆ Iterate regions
    for i, row in reg_df.iterrows():

        x, y, z = row["x"], row["y"], row["z"]

        stack = []
        for j in range(z, z + CUBE["z"]):
            if j >= seg_np.shape[2]:
                break  

            # Get a 2D slice from the z-plane
            slice_2d = seg_np[:, :, j]

            # Rotate the 2D slice
            rotated = ndimage.rotate(slice_2d, -45, reshape=True, order=1)
            cropped = rotated[x:x+CUBE["x"], y:y+CUBE["y"]]

            stack.append(cropped)

        cube_3d = np.stack(stack, axis=2)
        np.save(f"_npy/seg_{i}.npy", cube_3d)


# ∆ Initiate
if __name__ == "__main__":
    main()