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
from scipy.spatial import Delaunay, KDTree
import matplotlib.patches as patches
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# ∆ Constants
TOL_Z = 50
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]
EXCLUSION = [335, 653, 775, 1108, 1406] # ∆ Excluded via bounds
CUBE = {"x": 1000, "y": 1000, "z": 100}
ROT_45 = np.array([
    [np.cos(-np.pi/4), -np.sin(-np.pi/4), 0],
    [np.sin(-np.pi/4),  np.cos(-np.pi/4), 0],
    [0, 0, 1]
])

EXTENT = 200.0                    # ± extent along direction vector

def sph2cart(azimuth_deg, elevation_deg):
    """Convert azimuth and elevation in degrees to a unit 3D vector"""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])

def process_tile(i, r_, nodes):

    disc_vals = np.zeros(len(nodes), dtype=int)
    df = pd.read_csv(f"_csv/tile_{i}.csv")
    uni = df["ID"].to_list()

    # ∆ Load data and scaling
    data = np.load(f"_npy/seg_{i}.npy").astype(np.uint16)
    data[~np.isin(data, uni)] = 0
    scale = np.array([PIXX, PIXY, PIXZ])

    # ∆ Iterate values within data
    for val in uni:

        # ∆ Locate data for particular value
        idxs = np.argwhere(data == val)
        m_idxs = idxs * scale

        # Build a k-d tree from the voxel locations
        tree = KDTree(m_idxs)

        # Query all nodes for proximity to voxels of this val
        distances, _ = tree.query(nodes, distance_upper_bound=TOL_Z)

        # Set disc_vals where distance is within tolerance
        close_mask = distances <= TOL_Z
        disc_vals[close_mask] = val

        # row = df[df["ID"] == val]
        # azi, ele = row["Azi_[RAD]"].values[0], row["Ele_[RAD]"].values[0]
        # x = np.cos(ele) * np.cos(azi)
        # y = np.cos(ele) * np.sin(azi)
        # z = np.sin(ele)
        # vec =  np.array([x, y, z])
        # vec = vec / np.linalg.norm(vec)

        # for center in m_idxs:
        #     # ∆ Define start and end of the beam segment
        #     start = center - EXTENT * vec
        #     end = center + EXTENT * vec
        #     line_vec = end - start
        #     line_len_sq = np.dot(line_vec, line_vec)

        #     # ∆ Vector from start to each node
        #     v = nodes - start
        #     t = np.clip(np.dot(v, line_vec) / line_len_sq, 0, 1)
        #     proj = start + np.outer(t, line_vec)

        #     dists = np.linalg.norm(nodes - proj, axis=1)
        #     within = dists < TOL_Z
        #     disc_vals[within] = val

        # Build a k-d tree from the voxel locations
        tree = KDTree(m_idxs)

        # Query all nodes for proximity to voxels of this val
        distances, _ = tree.query(nodes, distance_upper_bound=TOL_Z)

        # Set disc_vals where distance is within tolerance
        close_mask = distances < TOL_Z
        disc_vals[close_mask] = val
    
    # ∆ Save file
    print(np.unique(disc_vals, return_counts=True))
    np.savetxt(f"_txt/nodes_tile_{i}.txt", disc_vals, fmt="%d")

# ∆ Assign disc to nodes
def disc_assignment(nodes):


    # ∆ Load region data
    reg_df = pd.read_csv("_csv/reg_.csv")

    for i, row in reg_df.iterrows():
        process_tile(i, row, nodes)
        if i > 1: break

    # # ∆ Run assignment in parallel
    # Parallel(n_jobs=-2)(
    #     delayed(process_tile)(i, row, nodes)
    #     for i, row in reg_df.iterrows()
    # )

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
def main(msh_size):

    # ∆ Load node data
    nodes = node_vals(msh_size)

    # ∆ Create node assignments
    disc_assignment(nodes)

# ∆ Inititate
if __name__ == "__main__":

    # ∆ Launch main with mesh size
    msh_size = 500
    main(msh_size)



    