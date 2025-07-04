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
from basix.ufl import element, mixed_element
from dolfinx import log, io,  default_scalar_type
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, Expression
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
def cauchy_tensor(u):
    
    # ∆ Kinematics
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u)  
    C = ufl.variable(F.T * F)  
    E = ufl.variable(0.5*(C-I))

    # ∆ Neo Hookean
    Ic = ufl.variable(ufl.tr(C))
    psi = 1 * (Ic - 3)
    SPK = 2 * F * (ufl.diff(psi, Ic)*I) 

    return 1/ufl.det(F) * F * SPK * F.T 

# ∆ Fenics simulation
def fx_(r, ref_dict):

    # ∆ Begin 't' log
    with open(f"_txt/ref_test.txt", 'a') as simlog:
        simlog.write(f"ref: {r}\n")

    # ∆ Load mesh data and set up function spaces
    domain, _, ft = io.gmshio.read_from_msh(filename=f"_msh/em_{r}.msh", comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
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

    # ∆ Push as Forward transform
    Push = ufl.Identity(DIM)

    # ∆ Load key function variables
    mx = Function(Mxs)
    v, q = ufl.TestFunctions(Mxs)
    u, p = ufl.split(mx)
    u_nu = Push * u

    # ∆ Kinematics Setup
    i, j, k, l, a, b = ufl.indices(6)  
    I = ufl.Identity(DIM)  
    F = ufl.variable(I + ufl.grad(u_nu))
    covDev = ufl.as_tensor(ufl.grad(v)[i, j], (i, j))

    # ∆ Kinematics Tensors
    C = ufl.variable(F.T * F)  
    B = ufl.variable(F * F.T)  
    E = ufl.as_tensor(0.5 * (C - I))
    J = ufl.det(F)   

    # # ∆ Seond Piola-Kirchoff 
    # # # ∆ Neo Hookean
    # Ic = ufl.variable(ufl.tr(C))
    # # psi = 0.001 * (Ic - 3)
    # # SPK = 2 * F * (ufl.diff(psi, Ic)*I) + p * I * J * ufl.inv(F).T

    # ∆ Extract Constitutive terms
    b0, bf, bt = [1, 1, 1]

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
    ]) - p * I

    # ∆ Residual
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": 6})
    R = ufl.as_tensor(SPK[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx

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

    log.set_log_level(log.LogLevel.INFO)

    for ii, kk in enumerate([5, 10, 15]):

        # ∆ Apply displacements as boundary conditions
        du = CUBE["x"] * PXLS["x"] * (kk / 100)
        d_xy0 = dirichletbc(default_scalar_type(du), xx0, Mxs.sub(0).sub(X))
        d_xy1 = dirichletbc(default_scalar_type(0.0), xx1, Mxs.sub(0).sub(X))
        d_yy0 = dirichletbc(default_scalar_type(0.0), yx0, Mxs.sub(0).sub(Y))
        d_yy1 = dirichletbc(default_scalar_type(0.0), yx1, Mxs.sub(0).sub(Y))
        d_zy0 = dirichletbc(default_scalar_type(0.0), zx0, Mxs.sub(0).sub(Z))
        d_zy1 = dirichletbc(default_scalar_type(0.0), zx1, Mxs.sub(0).sub(Z))
        bc = [d_xy0, d_yy0, d_zy0, d_xy1, d_yy1, d_zy1]

        # ∆ Write displacement store
        with open(f"_txt/ref_test.txt", 'a') as simlog:
            simlog.write(f"Applying: {kk}% DISP\n")

        # ∆ Solver
        problem = NonlinearProblem(R, mx, bc)
        solver = NewtonSolver(domain.comm, problem)
        solver.atol = TOL
        solver.rtol = TOL
        solver.max_it = 15
        solver.convergence_criterion = "incremental"

        # ∆ Solve
        try: 
            num_its, res = solver.solve(mx)
            with open(f"_txt/ref_test.txt", 'a') as simlog:
                simlog.write(f"Solved {kk}% IN:{num_its}, {res}\n")
        except:
            with open(f"_txt/ref_test.txt", 'a') as simlog:
                simlog.write(f"Failed.\n")
            return -1, ref_dict

        # ∆ Evaluation
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        pre.interpolate(p_eval)
        # µ Evaluate stress
        cauchy = Expression(
            e=cauchy_tensor(u_eval), 
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
        
        with open(f"_txt/ref_test.txt", 'a') as simlog:
            simlog.write(f"sigs: {sig_c}.\n")
            simlog.write(f"10% Vals: {pct_sigs}\n")

        for zz in [0, 1, 2, 3, 4]:
            ref_dict[f"s{zz}_{kk}"].append(sig_c[zz])
            ref_dict[f"s{zz}_{kk}_10%"].append(pct_sigs[zz])

    return num_its, ref_dict

# ∆ .msh generation
def msh_(r):

    # ∆ Initialise
    gmsh.initialize()
    gmsh.model.add(f"em_{r}")

    # ∆ Create a box
    box = gmsh.model.occ.addBox(0, 0, 0, G["x"], G["y"], G["z"])
    gmsh.model.occ.synchronize()

    # ∆ Label physical groups
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
                elif abs(com[0] - G["x"]) < tol:
                    axis_str, pos = "x", 1
                elif abs(com[1]) < tol:
                    axis_str, pos = "y", 0
                elif abs(com[1] - G["y"]) < tol:
                    axis_str, pos = "y", 1
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
    refs = [x for x in range(200, 3100, 3000)]

    # ∆ Begin 't' log
    with open(f"_txt/ref_test.txt", 'w') as simlog:
        simlog.write(f"begin tests with: {refs}\n")

    # ∆ Create dataframes
    ref_dict = {
        "s0_5": [], "s1_5": [], "s2_5": [], "s3_5": [], "s4_5": [],
        "s0_5_10%": [], "s1_5_10%": [], "s2_5_10%": [], "s3_5_10%": [], "s4_5_10%": [],
        "s0_10": [], "s1_10": [], "s2_10": [], "s3_10": [], "s4_10": [],
        "s0_10_10%": [], "s1_10_10%": [], "s2_10_10%": [], "s3_10_10%": [], "s4_10_10%": [],
        "s0_15": [], "s1_15": [], "s2_15": [], "s3_15": [], "s4_15": [],
        "s0_15_10%": [], "s1_15_10%": [], "s2_15_10%": [], "s3_15_10%": [], "s4_15_10%": [],
        "s0_20": [], "s1_20": [], "s2_20": [], "s3_20": [], "s4_20": [],
        "s0_20_10%": [], "s1_20_10%": [], "s2_20_10%": [], "s3_20_10%": [], "s4_20_10%": []
    }

    # ∆ Iterate testing
    for r in refs:

        # ∆ Create Mesh
        msh_(r)
        # ∆ Run simulation
        _, ref_dict = fx_(r, ref_dict)

    # ∆ Save CSV
    pd.DataFrame(ref_dict).to_csv(f"_csv/ref_test.csv")  

# ∆ Initialise
if __name__ == '__main__':
    main()