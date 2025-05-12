"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _nodeAssignment.py
        Reads zdisc data and creates per simulation tetrations for node assignment of mesh
"""

# ∆ Raw
import os
import ast
import random
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import measure
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# ∆ Constants
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]
EXCLUSION = [335, 653, 775, 1108, 1406] # ∆ Excluded via bounds
CUBE = {"x": 1000, "y": 1000, "z": 100}

# ∆ Assign disc to nodes
def disc_assignment(data, nodes):

    # ∆ Load region data
    reg_df = pd.read_csv("_csv/reg_.csv")

    # ∆ Iterate through tiles
    for i, r_ in reg_df.iterrows():

        # ∆ Load csv with tile data
        disc_vals = np.zeros(len(nodes))
        df = pd.read_csv(f"_csv/tile_{i}.csv")
        ori = np.array([r_["x"], r_["y"], r_["z"]])

        # ∆ Iterate tile data
        for j, row in df.iterrows():

            # ∆ Find Indexes where value exists and transform
            val = row["ID"]
            idxs = np.argwhere(data == val)
            idxs -= ori

            # ∆ Rotate data to appropriate points
            ori_ids = np.array(ast.literal_eval(row["Centroid"]))
            mu_idxs = idxs * np.array([PIXX, PIXY, PIXZ])
            s_mu_idxs = mu_idxs - ori_ids
            n45 = -np.pi/4
            rot = np.array([
                [np.cos(n45), -np.sin(n45), 0],
                [np.sin(n45),  np.cos(n45), 0],
                [0, 0, 1]
            ])
            r_s_mu_idxs = (rot @ s_mu_idxs.T).T
            r_mu_idxs = r_s_mu_idxs + ori

            # ∆ Delaunay to form hull 
            tri = Delaunay(mu_idxs)

            # ∆ Condition on nodes being in the hull
            for k, n in enumerate(nodes):
                cells = tri.find_simplex(n)
                if cells >= 0:
                    disc_vals[k] = val
                    print(cells)

            print(np.unique(disc_vals))
        

        # Optionally save intermediate file for this tile
        np.savetxt(f"disc_assignment_tile_{i}.txt", disc_vals, fmt="%d")

# ∆ Node cartesian values
def node_vals(msh_size):

    # ∆ Load node data
    n_ = []
    f = f"_msh/em_{msh_size}.nodes"

    # ∆ Load coordinate data
    for line in open(f, 'r'):
        n_.append(line.strip().replace('\t', ' ').split(' '))
    node = np.array(n_[1:]).astype(np.float64)

    return node[:, 1:]

            
# ∆ Main
def main(data, msh_size):

    # ∆ Load node data
    nodes = node_vals(msh_size)

    # ∆ Create node assignments
    disc_assignment(data, nodes)


    print("node")

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Load segmentation data
    seg_np = np.load("/Users/murrayla/Documents/main_PhD/BIG_SEG/filtered.npy").astype(np.uint16)

    # ∆ Launch main with mesh size
    msh_size = 1000
    main(seg_np, msh_size)



    