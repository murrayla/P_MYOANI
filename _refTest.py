"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _refTest.py
        testing mesh refinements
"""

# ∆ Base
import gmsh
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
from dolfinx.fem import Function, functionspace, dirichletbc, Constant, locate_dofs_topological, Expression
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
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
BASE_TAGS = {
    'point': 1000,
    'line': 2000,
    'surf_x': 3000,
    'surf_y': 3010,
    'surf_z': 3020,
    'vol': 4000
}
G = {d: CUBE[d] * PXLS[d] for d in "xyz"}

# ∆ Cauchy
def cauchy_tensor(u, gcc):

    # ∆ Kinematics
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u)  
    C = ufl.variable(F.T * F)  
    E = ufl.variable(0.5*(C-I))

    # ∆ Extract Constitutive terms
    b0, bf, bt = gcc

    # ∆ Exponent term
    Q = (
        bf * E[0,0]**2 + bt * 
        (
            E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2 + 
            E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2
        )
    )

    # ∆ Seond Piola-Kirchoff 
    SPK = b0/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*bf*E[0,0], 2*bt*(E[1,0] + E[0,1]), 2*bt*(E[2,0] + E[0,2])],
        [2*bt*(E[0,1] + E[1,0]), 4*bt*E[1,1], 2*bt*(E[2,1] + E[1,2])],
        [2*bt*(E[0,2] + E[2,0]), 2*bt*(E[1,2] + E[2,1]), 4*bt*E[2,2]],
    ])

    return 1/ufl.det(F) * F * SPK * F.T 

# ∆ Assign angles to dofs
def angle_assign(t, coords):

    # ∆ Create arrays
    azi = np.zeros_like(coords[:, 0])
    ele = np.zeros_like(coords[:, 0]) 
    zs = np.zeros_like(coords[:, 0])
    return azi, ele, zs

# ∆ Fenics simulation
def fx_(t, file, gcc, r, ref_dict, pct_dict):

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
    b0, bf, bt = gcc

    # ∆ Exponent term
    Q = (
        bf * E[0,0]**2 + bt * 
        (
            E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2 + 
            E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2
        )
    )

    # ∆ Seond Piola-Kirchoff 
    SPK = b0/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*bf*E[0,0], 2*bt*(E[1,0] + E[0,1]), 2*bt*(E[2,0] + E[0,2])],
        [2*bt*(E[0,1] + E[1,0]), 4*bt*E[1,1], 2*bt*(E[2,1] + E[1,2])],
        [2*bt*(E[0,2] + E[2,0]), 2*bt*(E[1,2] + E[2,1]), 4*bt*E[2,2]],
    ]) - p * G_v_inv

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
    sig = Function(Tes)

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

    for ii, kk in enumerate([2,6,10,12,14,16,18,20]):

        # ∆ Apply displacements as boundary conditions
        du = CUBE["x"] * PXLS["x"] * (kk / 100)
        du_pos.value = default_scalar_type(du // 2)
        du_neg.value = default_scalar_type(-du //2)

        # ∆ Write displacement store
        with open(f"_txt/ref_gcc.txt", 'a') as simlog:
            simlog.write(f"Applying: {kk}% DISP\n")

        # ∆ Solve
        try: 
            num_its, res = solver.solve(mx)
            with open(f"_txt/ref_gcc.txt", 'a') as simlog:
                simlog.write(f"Solved {kk}% IN:{num_its}, {res}\n")
        except:
            with open(f"_txt/ref_gcc.txt", 'a') as simlog:
                simlog.write(f"Failed.\n")
                for zz in [0, 1, 2, 3, 4]:
                    ref_dict[f"{zz}_{kk}"].append(np.nan)
                    pct_dict[f"{zz}_{kk}"].append(np.nan)
            return -1, ref_dict

        # ∆ Evaluation
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        pre.interpolate(p_eval)
        # µ Evaluate stress
        cauchy = Expression(
            e=cauchy_tensor(u_eval, gcc), 
            X=Tes.element.interpolation_points()
        )
        # µ Evaluate strain
        sig.interpolate(cauchy)
        
        # ∆ Format for saving
        sig_arr = sig.x.array
        n_nodes = len(sig_arr) // DIM**2
        r_sig = sig_arr.reshape((n_nodes, DIM**2))

        # ∆ Store data
        sigs = []
        coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
        df = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
                "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5]
            }
        )

        # ∆ Determine points within the 3D space
        xx = np.linspace(500, EDGE[0] - 500, 5)
        yy = np.linspace(500, EDGE[1] - 500, 5)
        zz = np.linspace(500, EDGE[2] - 500, 5)
        sig_c = []
        pct_sigs = []
        
        # Iterate points
        for jj, (xp, yp, zp) in enumerate(zip(xx, yy, zz)):

            # ∆ Fine within range
            df_p = df[(
                (df["X"] >= xp - 400) & (df["X"] <= xp + 400) & 
                (df["Y"] >= yp - 400) & (df["Y"] <= yp + 400) & 
                (df["Z"] >= zp - 400) & (df["Z"] <= zp + 400) 
            )]

            # ∆ Caluclate values
            sig_c.append(df_p.loc[:, "sig_xx"].mean())
            _, idx = tree.query([xp, yp, zp], k=3)
            vals = [df.iloc[x]["sig_xx"] for x in idx]
            pct = np.mean(
                [((x - vals[0]) / vals[0]) * 100 for x in vals[1:]]
            )
            pct_sigs.append(pct)
        
        with open(f"_txt/ref_gcc.txt", 'a') as simlog:
            simlog.write(f"sigs: {sig_c}.\n")
            simlog.write(f"10% Vals: {pct_sigs}\n")

        for zz in [0, 1, 2, 3, 4]:
            ref_dict[f"{zz}_{kk}"].append(sig_c[zz])
            pct_dict[f"{zz}_{kk}"].append(pct_sigs[zz])

    return num_its, ref_dict, pct_dict

# ∆ .msh generation
def msh_(r):

    # ∆ Initialise
    print("\t" + f"+= Generate mesh with size: {r}")
    gmsh.initialize()
    gmsh.model.add(f"em_{r}")

    # ∆ Create a box
    box = gmsh.model.occ.addBox(0, 0, 0, G["x"], G["y"], G["z"])
    gmsh.model.occ.synchronize()

    # ∆ Label physical groups
    x_face = []
    y_face = []
    tol = 1e-6
    seen_tgs = set()
    for dim in range(DIM + 1):

        # ∆ Iterate dimensions
        for ent_dim, ent_tag in gmsh.model.occ.getEntities(dim):

            # ∆ Determine centre of mass 
            com = gmsh.model.occ.get_center_of_mass(ent_dim, ent_tag)

            # ∆ Assign physical groups
            if dim == 3:
                tag = BASE_TAGS["vol"]
                name = "vol"
            elif dim == 2:
                if abs(com[0]) < tol:
                    axis_str, pos = "x", 0
                    x_face.append(ent_tag)
                elif abs(com[0] - G["x"]) < tol:
                    axis_str, pos = "x", 1
                    x_face.append(ent_tag)
                elif abs(com[1]) < tol:
                    axis_str, pos = "y", 0
                    y_face.append(ent_tag)
                elif abs(com[1] - G["y"]) < tol:
                    axis_str, pos = "y", 1
                    y_face.append(ent_tag)
                elif abs(com[2]) < tol:
                    axis_str, pos = "z", 0
                elif abs(com[2] - G["z"]) < tol:
                    axis_str, pos = "z", 1
                tag = BASE_TAGS[f'surf_{axis_str}'] + pos
                name = f"sf_{axis_str}_{pos}"
            elif dim == 1:
                tag = BASE_TAGS['line'] + ent_tag
                name = f"ln_{ent_tag}"
            else:
                tag = BASE_TAGS['point'] + ent_tag
                name = f"pt_{ent_tag}"

            # ∆ Check tag logic
            if tag not in seen_tgs:
                gmsh.model.add_physical_group(dim, [ent_tag], tag)
                gmsh.model.set_physical_name(dim, tag, name)
                seen_tgs.add(tag)

    # ∆ Set mesh size
    for p in gmsh.model.getEntities(dim=0):
        gmsh.model.mesh.setSize([p], r)

    # ∆ Mesh generation
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(DIM)
    gmsh.model.mesh.setOrder(order=ORDER)

    # ∆ Save to file
    file = f"_msh/em_{r}.msh"
    gmsh.write(file)
    gmsh.finalize()

    return file

# ∆ Main
def main():

    # ∆ Load references
    refs = [200]

    # ∆ Begin 't' log
    with open(f"_txt/ref_gcc.txt", 'w') as simlog:
        simlog.write(f"begin tests with: {refs}\n")

    # ∆ Create dataframes
    ref_dict = {
        "0_2": [], "0_6": [], "0_10": [], "0_12": [], "0_14": [], "0_16": [], "0_18": [], "0_20": [],
        "1_2": [], "1_6": [], "1_10": [], "1_12": [], "1_14": [], "1_16": [], "1_18": [], "1_20": [],
        "2_2": [], "2_6": [], "2_10": [], "2_12": [], "2_14": [], "2_16": [], "2_18": [], "2_20": [],
        "3_2": [], "3_6": [], "3_10": [], "3_12": [], "3_14": [], "3_16": [], "3_18": [], "3_20": [],
        "4_2": [], "4_6": [], "4_10": [], "4_12": [], "4_14": [], "4_16": [], "4_18": [], "4_20": [],
    }

    pct_dict = {
        "0_2": [], "0_6": [], "0_10": [], "0_12": [], "0_14": [], "0_16": [], "0_18": [], "0_20": [],
        "1_2": [], "1_6": [], "1_10": [], "1_12": [], "1_14": [], "1_16": [], "1_18": [], "1_20": [],
        "2_2": [], "2_6": [], "2_10": [], "2_12": [], "2_14": [], "2_16": [], "2_18": [], "2_20": [],
        "3_2": [], "3_6": [], "3_10": [], "3_12": [], "3_14": [], "3_16": [], "3_18": [], "3_20": [],
        "4_2": [], "4_6": [], "4_10": [], "4_12": [], "4_14": [], "4_16": [], "4_18": [], "4_20": [],
    }

    # ∆ Constitutive Values
    gcc = [1, 11, 10]

    # ∆ Iterate testing
    for r in refs:

        # ∆ Create Mesh
        msh_(r)
        # ∆ Run simulation
        file = f"_msh/em_{r}.msh"
        _, ref_dict, pct_dict = fx_("test", file, gcc, r, ref_dict, pct_dict)

    # ∆ Save CSV
    pd.DataFrame(ref_dict).to_csv(f"_csv/ref_gcc_200.csv")  
    pd.DataFrame(pct_dict).to_csv(f"_csv/pct_gcc_200.csv")  

# ∆ Initialise
if __name__ == '__main__':
    main()