"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _Sim.py
       finite element simulation with FENICS
"""

# ∆ Raw
import random
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

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
RADIUS = 1000
QUADRATURE = 4
X, Y, Z = 0, 1, 2
PIXX, PIXY, PIXZ = 11, 11, 50
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
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
def angle_assign(t, coords):

    # ∆ Create arrays
    azi = np.zeros_like(coords[:, 0])
    ele = np.zeros_like(coords[:, 0]) 
    sph = np.zeros_like(coords[:, 0])
    zid = np.zeros_like(coords[:, 0])
    zs = np.zeros_like(coords[:, 0])
    nvecs = np.zeros_like(coords)

    if t == "test":
        return azi, ele, zs

    # ∆ Load tile data and reduce to only unique IDs
    data_df = pd.read_csv(f"_csv/norm_stats_whole.csv")
    uni = data_df["ID"].to_list()

    # ∆ Load segmentation 
    data = np.load(f"_npy/seg_{t}.npy").astype(np.uint16)
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
    azi = final_pass(coords, azi)
    ele = final_pass(coords, ele)
    sph = final_pass(coords, sph)

    return azi, ele, zs

# ∆ Fenics simulation
def fx_(t, file, gcc, r):

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
    azi, ele, zs = angle_assign(t, coords)

    # ∆ Store angles
    CA, CE = np.cos(azi), np.cos(ele)
    SA, SE = np.sin(azi), np.sin(ele)

    # ∆ Create interpolate functions
    # µ Basis vector 1
    def nu_1(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        return np.array([CA[idx]*CE[idx], SA[idx]*CE[idx], -SE[idx]])
    # µ Basis vector 2
    def nu_2(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        return np.array([-SA[idx], CA[idx], np.zeros_like(CA[idx])])
    # µ Basis vector 3
    def nu_3(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        return np.array([CA[idx]*SE[idx], SA[idx]*SE[idx], CE[idx]])

    # ∆ Create z_disc id data
    z_arr = z_data.x.array.reshape(-1, 3)
    z_arr[:, 0], z_arr[:, 1], z_arr[:, 2] = zs, azi, ele
    z_data.x.array[:] = z_arr.reshape(-1)

    # ∆ Create angular orientation vector
    ori.interpolate(nu_1)

    # ∆ Create push tensor function
    Push = Function(Tes)

    # ∆ Define push interpolation
    def forward(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        f00, f01, f02 = CA[idx]*CE[idx], -SA[idx], CA[idx]*SE[idx]
        f10, f11, f12 = SA[idx]*CE[idx], CA[idx], SA[idx]*SE[idx]
        f20, f21, f22 = -SE[idx], np.zeros_like(CE[idx]), CE[idx]
        return np.array([f00, f01, f02, f10, f11, f12, f20, f21, f22])

    # ∆ Interpolate Push as Forward transform
    Push.interpolate(forward)

    # ∆ Load key function variables
    mx = Function(Mxs)
    v, q = ufl.TestFunctions(Mxs)
    u, p = ufl.split(mx)
    u_nu = Push * u

    # ∆ Kinematics Setup
    i, j, k, l, a, b = ufl.indices(6)  
    I = ufl.Identity(DIM)  
    F = ufl.variable(I + ufl.grad(u_nu))

    # ∆ Metric tensors
    # µ [UNDERFORMED] Covariant basis vectors 
    A1, A2, A3 = Function(V), Function(V), Function(V)
    # ¬ create base 1
    A1.interpolate(nu_1)
    # ¬ create base 2
    A2.interpolate(nu_2)
    # ¬ create base 3
    A3.interpolate(nu_3)
    
    # µ [UNDERFORMED] Metric tensors
    G_v = ufl.as_tensor([
        [ufl.dot(A1, A1), ufl.dot(A1, A2), ufl.dot(A1, A3)],
        [ufl.dot(A1, A2), ufl.dot(A2, A2), ufl.dot(A2, A3)],
        [ufl.dot(A1, A3), ufl.dot(A2, A3), ufl.dot(A3, A3)]
    ]) 
    G_v_inv = ufl.inv(G_v)  
    # µ [DEFORMED] Metric covariant tensors
    g_v = ufl.as_tensor([
        [ufl.dot(F * A1, F * A1), ufl.dot(F * A1, F * A2), ufl.dot(F * A1, F * A3)],
        [ufl.dot(F * A2, F * A1), ufl.dot(F * A2, F * A2), ufl.dot(F * A2, F * A3)],
        [ufl.dot(F * A3, F * A1), ufl.dot(F * A3, F * A2), ufl.dot(F * A3, F * A3)]
    ])
    g_v_inv = ufl.inv(g_v)

    # ∆ Christoffel symbols 
    Gamma = ufl.as_tensor(
        0.5 * G_v_inv[k, l] * (ufl.grad(G_v[j, l])[i] + ufl.grad(G_v[i, l])[j] - ufl.grad(G_v[i, j])[l]),
        (i, j, k)
    )

    # ∆ Covariant derivative
    covDev = ufl.as_tensor(ufl.grad(v)[i, j] + Gamma[i, k, j] * v[k], (i, j))

    # ∆ Kinematics Tensors
    C = ufl.variable(F.T * F)  
    B = ufl.variable(F * F.T) 
    E = ufl.as_tensor(0.5 * (g_v - G_v))   
    J = ufl.det(F)   

    # ∆ Extract Constitutive terms
    c0, bf, bt = gcc

    # ∆ Exponent term
    Q = (
        bf * (E[0,0]**2) + 
        bt * (E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2) +
        bt * (E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2)
    )

    # ∆ Seond Piola-Kirchoff 
    spk = c0/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*bf*E[0,0], 2*bt*(E[1,0] + E[0,1]), 2*bt*(E[2,0] + E[0,2])],
        [2*bt*(E[0,1] + E[1,0]), 4*bt*E[1,1], 2*bt*(E[2,1] + E[1,2])],
        [2*bt*(E[0,2] + E[2,0]), 2*bt*(E[1,2] + E[2,1]), 4*bt*E[2,2]],
    ])
    SPK = spk - p * g_v_inv #G_v_inv

    cau = 1/ufl.det(F) * F * spk * F.T

    # ∆ Residual
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUADRATURE})
    R = ufl.as_tensor(SPK[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx

    # ∆ Solver
    problem = NonlinearProblem(R, mx, [])
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.atol = TOL
    solver.rtol = TOL
    solver.max_it = 50
    solver.convergence_criterion = "incremental"

    ksp = solver.krylov_solver
    opts = PETSc.Options()
    ksp.setOptionsPrefix("")
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    opts["pc_factor_mat_solver_type"] = "mumps"
    opts["ksp_monitor"] = None
    ksp.setFromOptions()

    # ∆ Data store
    dis, pre = Function(V), Function(P) 
    sig, eps = Function(Tes), Function(Tes)
    dis.name = "U - Displacement"
    pre.name = "P - Pressure"
    sig.name = "S - Cauchy Stress"
    eps.name = "E - Green Strain"
    dis_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_DISP.bp", dis, engine="BP4")
    pre_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_PRE.bp", pre, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_SIG.bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_EPS.bp", eps, engine="BP4")

    # ∆ Setup boundary terms
    tgs_x0 = ft.find(3000)
    tgs_x1 = ft.find(3001)
    xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x0)
    xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x1)
    yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x0)
    yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x1)
    zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x0)
    zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x1)

    # ∆ Set boundaries
    du_pos = Constant(domain, default_scalar_type(0.0))
    du_neg = Constant(domain, default_scalar_type(0.0))
    d_xy0 = dirichletbc(du_pos, xx0, Mxs.sub(0).sub(X)) 
    d_xy1 = dirichletbc(du_neg, xx1, Mxs.sub(0).sub(X)) 
    d_yy0 = dirichletbc(default_scalar_type(0.0), yx0, Mxs.sub(0).sub(Y))
    d_yy1 = dirichletbc(default_scalar_type(0.0), yx1, Mxs.sub(0).sub(Y))
    d_zy0 = dirichletbc(default_scalar_type(0.0), zx0, Mxs.sub(0).sub(Z))
    d_zy1 = dirichletbc(default_scalar_type(0.0), zx1, Mxs.sub(0).sub(Z))
    bc = [d_xy0, d_yy0, d_zy0, d_xy1, d_yy1, d_zy1]
    problem.bcs = bc

    # ∆ Iterative solver
    incs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    for ii, kk in enumerate(incs):

        # ∆ Apply displacements as boundary conditions
        du = CUBE["x"] * PXLS["x"] * (kk / 100)
        du_pos.value = default_scalar_type(du // 2)
        du_neg.value = default_scalar_type(-du //2)

        # ∆ Solve
        try: 
            num_its, res = solver.solve(mx)
            print(f"SOLVED {kk}% IN:{num_its}, {res}")
        except:
            # ∆ Close files
            dis_file.close()
            pre_file.close()
            sig_file.close()
            eps_file.close()
            return -1, 0 
        
        # ∆ Evaluation
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        pre.interpolate(p_eval)
        # µ Evaluate stress
        cauchy = Expression(
            e=cau,  
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        # µ Evaluate stress
        green = Expression(
            e=E, 
            X=Tes.element.interpolation_points()
        )
        eps.interpolate(green)
        
        # ∆ Format for saving
        sig_arr = sig.x.array
        eps_arr = eps.x.array
        n_nodes = len(sig_arr) // DIM**2
        r_sig = sig_arr.reshape((n_nodes, DIM**2))
        r_eps = eps_arr.reshape((n_nodes, DIM**2))

        # ∆ Repeat for displacement
        disp_arr = dis.x.array
        r_disp = disp_arr.reshape((len(disp_arr) // DIM), DIM)

        # ∆ Store tensor data
        coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
        df = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
                "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
                "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
                "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5],
                "disp_x": r_disp[:, 0], "disp_y": r_disp[:, 1], "disp_z": r_disp[:, 2]
            }
        )
        
        # ∆ Save CSV
        df.to_csv(f"_csv/sim_{t}_{kk}_{r}.csv")  

        # ∆ Write files
        dis_file.write(ii)
        pre_file.write(ii)
        sig_file.write(ii)
        eps_file.write(ii)

    # ∆ Close files
    dis_file.close()
    pre_file.close()
    sig_file.close()
    eps_file.close()

    return num_its, 1

# ∆ Main
def main(tests, m_ref):

    # ∆ Test vals
    # gcc = [3.51, 13.34, 10.37]
    # gcc = [1, 11, 10]
    gcc = [2.8107424166757204, 12.677385904668105, 11.039328556686252]
    # gcc = [4.69, 10.15, 16.57]

    # ∆ Iterate tests 
    wins = []
    for t in tests:

        # ∆ Mesh file
        file = f"_msh/em_{m_ref}.msh"

        # ∆ Simulation 
        _, win = fx_(t, file, gcc, m_ref)
        if win: wins.append(t)

    print(f" ~> WIN: {win}")

# ∆ Inititate 
if __name__ == '__main__':

    # ∆ Indicate test cases
    # tests = ["test"] + [x for x in range(0, 18, 1)]
    tests = [11]

 
    # ∆ Refinements
    r = 300
    main(tests, r)