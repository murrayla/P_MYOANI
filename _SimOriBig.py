"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _simOri.py
       finite element simulation with FENICS
"""

# ∆ Raw
import random
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

# ∆ Dolfin
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx import log, io,  default_scalar_type
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, Expression, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# ∆ Seed
random.seed(17081993)

# ∆ Global Constants
DIM = 3
ORDER = 2 
TOL = 1e-5
Z_TOL = 50
QUADRATURE = 4
X, Y, Z = 0, 1, 2
PIXX, PIXY, PIXZ = 11, 11, 50
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1800, "y": 3000, "z": 300}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]

# ∆ Smooth data globally
def global_smooth(coords, data):

    tol = 1e-8
    # ∆ Set standard deviations
    sx, sy, sz = 1000, 500, 500

    # ∆ Nearest neighbour tree
    tree = KDTree(coords)

    # ∆ Store new data and mask
    s_data = np.zeros_like(data)
    mask = (data != 0).astype(float)

    # ∆ Iterate data and store 
    for i in range(0, len(coords), 1):

        # ∆ Create neighbours
        kn_idx = tree.query_ball_point(coords[i], r=2*sx)

        # ∆ Assign values
        knn = coords[kn_idx]
        vals = data[kn_idx]
        knn_mask = mask[kn_idx]

        # ∆ Compute distances
        dx = knn[:, 0] - coords[i, 0]
        dy = knn[:, 1] - coords[i, 1]
        dz = knn[:, 2] - coords[i, 2]

        # ∆ Compute weights
        wei = np.exp(-0.5 * ((dx / sx)**2 + (dy / sy)**2 + (dz / sz)**2))

        # ∆ Mask weight data
        w_data = wei * vals * knn_mask
        w_mask = wei * knn_mask

        # ∆ Store
        if w_mask.sum() > tol: 
            s_data[i] = w_data.sum() / (w_mask.sum() + tol)
        else:
            s_data[i] = 0.0

    return s_data

# ∆ Assign angles to dofs
def angle_assign(coords):

    # ∆ Create arrays
    azi = np.zeros_like(coords[:, 0])
    ele = np.zeros_like(coords[:, 0]) 
    sph = np.zeros_like(coords[:, 0])
    zid = np.zeros_like(coords[:, 0])
    zs = np.zeros_like(coords[:, 0])
    nvecs = np.zeros_like(coords)

    # ∆ Load tile data and reduce to only unique IDs
    data_df = pd.read_csv(f"_csv/rot_norm_whole.csv")
    uni = data_df["ID"].to_list()

    # ∆ Load segmentation 
    data = np.load(f"_npy/seg_big.npy").astype(np.uint16)
    data = np.transpose(data, (1, 0, 2))

    # ∆ Scale and move data
    scale = np.array([PIXX, PIXY, PIXZ])
    idxs = np.argwhere(data >= 0)
    m_idxs = idxs * scale

    # ∆ Generate nearest neighbours tree
    knn = KDTree(m_idxs)
    node_tree = KDTree(coords)

    # ∆ Distance parameters
    tol_dist = 1000
    sar_dist = 1400
    s_step = 200 

    # ∆ Determine matching
    dist, idx = knn.query(coords, distance_upper_bound=tol_dist)

    # ∆ Iterate through the distances and indices
    for i, (dist, ii) in enumerate(zip(dist, idx)):

        # ∆ Check if tolerance distance is appreciated
        if dist >= tol_dist:
            continue

        # ∆ Generate 3D position for ID value
        ix, iy, iz = m_idxs[ii, :]
        id = data[ix // PIXX, iy // PIXY, iz // PIXZ]

        # ∆ Check if ID is included
        if not id or id not in uni:
            continue

        # ∆ Assign angle values
        azi_val = data_df.loc[data_df["ID"] == id, "Azi_[RAD]"].values[0]
        ele_val = data_df.loc[data_df["ID"] == id, "Ele_[RAD]"].values[0]
        sph_val = data_df.loc[data_df["ID"] == id, "Sph_[DEG]"].values[0]

        # ∆ Calculate vector in direction of angles
        nvec = np.array([
            ufl.cos(azi_val) * ufl.cos(ele_val),
            ufl.sin(azi_val) * ufl.cos(ele_val),
            -ufl.sin(ele_val)
        ])
        # µ Normalise
        nvec /= np.linalg.norm(nvec)

        # ∆ Assign at centre first
        pt = coords[i]
        kn_idx = node_tree.query_ball_point(pt, r=500)
        zs[kn_idx] = 1
        azi[kn_idx] = azi_val
        ele[kn_idx] = ele_val
        sph[kn_idx] = sph_val
        nvecs[kn_idx] = nvec
        zid[kn_idx] = 1

        # ∆ Extend along directions of interest
        for a in np.linspace(-1, 1, s_step):

            # ∆ Calculate point
            pt = coords[i] + a * sar_dist * nvec

            # ∆ Find new points
            kn_idx = node_tree.query_ball_point(pt, r=500)

            # ∆ Assign values
            for j in kn_idx:
                if not(a):
                    zs[j] = 1
                azi[j] = azi_val
                ele[j] = ele_val
                sph[j] = sph_val
                nvecs[j] = nvec
                zid[j] = 1

    # ∆ Global smoothing
    azi = global_smooth(coords, azi)
    ele = global_smooth(coords, ele)
    sph = global_smooth(coords, sph)

    # ∆ Final smoothing 
    def final_pass(coords, data):
        tol = 1e-8
        sx, sy, sz = 500, 500, 500
        s_data = np.zeros_like(data)
        for i in range(0, len(coords), 1):
            dx = coords[:, 0] - coords[i, 0]
            dy = coords[:, 1] - coords[i, 1]
            dz = coords[:, 2] - coords[i, 2]
            wei = np.exp(-0.5 * ((dx / sx)**2 + (dy / sy)**2 + (dz / sz)**2))
            wei /= wei.sum() + tol
            s_data[i] = np.sum(wei * data)
        return s_data
    
    # ∆ Simple smoothing
    # azi = final_pass(coords, azi)
    # ele = final_pass(coords, ele)
    # sph = final_pass(coords, sph)

    return azi, ele, zs, sph

# ∆ Fenics simulation
def fx_(file, gcc):

    # ∆ Load mesh data and set up function spaces
    domain, _, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    P2 = element("Lagrange", domain.basix_cell(), ORDER, shape=(domain.geometry.dim,))
    P1 = element("Lagrange", domain.basix_cell(), ORDER-1)
    Mxs = functionspace(mesh=domain, element=mixed_element([P2, P1]))
    Tes = functionspace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))

    # ∆ Define subdomains
    V, _ = Mxs.sub(0).collapse()
    P, _ = Mxs.sub(1).collapse()

    # ∆ Determine coordinates of space and create mapping tree
    x_n = Function(V)
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    tree = KDTree(coords)

    # ∆ Setup functions for assignment
    ori, z_data = Function(V), Function(V)

    # ∆ Assign angle and z-disc data
    azi, ele, zs, sph = angle_assign(coords)

    # ∆ Store angles
    CA, CE = np.cos(azi), np.cos(ele)
    SA, SE = np.sin(azi), np.sin(ele)

    # ∆ Create interpolate functions
    # µ Basis vector 1
    def nu_1(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        return np.array([CA[idx]*CE[idx], SA[idx]*CE[idx], -SE[idx]])

    # ∆ Create z_disc id data
    z_arr = z_data.x.array.reshape(-1, 3)
    z_arr[:, 0], z_arr[:, 1], z_arr[:, 2] = zs, sph, ele
    z_data.x.array[:] = z_arr.reshape(-1)

    # ∆ Create angular orientation vector
    ori.interpolate(nu_1)

    # ∆ Save foundations data
    z_data.name = "Node Mapping"
    ori.name = "Orientation Vectors"
    with io.VTXWriter(MPI.COMM_WORLD, f"_bp/_BIG/_ZSN.bp", z_data, engine="BP4") as fz:
        fz.write(0)
        fz.close()
    with io.VTXWriter(MPI.COMM_WORLD, f"_bp/_BIG/_ORI.bp", ori, engine="BP4") as fo:
        fo.write(0)
        fo.close()

# ∆ Main
def main(m_ref):

    # ∆ Test vals
    gcc = [1, 11, 10]

    # ∆ Mesh file
    file = f"_msh/em_{m_ref}_BIG.msh"

    # ∆ Simulation 
    fx_(file, gcc)

# ∆ Inititate 
if __name__ == '__main__':
    # ∆ Refinements
    r = 500
    main(r)