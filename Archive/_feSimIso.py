"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _feSim.py
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
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]

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

# ∆ Fenics simulation
def fx_(file, gcc):

    t = "test"

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

    # ∆ Load key function variables
    mx = Function(Mxs)
    v, q = ufl.TestFunctions(Mxs)
    u, p = ufl.split(mx)

    # ∆ Kinematics Setup
    # i, j, k, l, a, b = ufl.indices(6)  
    I = ufl.Identity(DIM)  
    F = ufl.variable(I + ufl.grad(u))
    # covDev = ufl.as_tensor(ufl.grad(v)[i, j], (i, j))

    # ∆ Kinematics Tensors
    C = ufl.variable(F.T * F)  
    E = ufl.as_tensor(0.5 * (C - I))
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
    ]) - p * I

    # ∆ Residual
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUADRATURE})
    # R = ufl.as_tensor(SPK[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx
    R =  ufl.inner(ufl.grad(v), F * SPK) * dx + q * (J - 1) * dx 

    # log.set_log_level(log.LogLevel.INFO)

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
    dis.name = "U - Displacement"
    pre.name = "P - Pressure"
    sig.name = "S - Cauchy Stress"
    dis_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_DISP.bp", dis, engine="BP4")
    pre_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_PRE.bp", pre, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_SIG.bp", sig, engine="BP4")

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

    for ii, kk in enumerate([0, 5, 10, 15, 20]):

        # ∆ Apply displacements as boundary conditions
        du = CUBE["x"] * PXLS["x"] * (kk / 100)
        du_pos.value = default_scalar_type(du // 2)
        du_neg.value = default_scalar_type(-du //2)

        # ∆ Solve
        try: 
            num_its, res = solver.solve(mx)
            print(f"SOLVED {kk}% IN:{num_its}, {res}")
        except:
            return -1, 0 
        
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
        # µ 0 - corner ymax
        xx = np.linspace(500, EDGE[0] - 500, 5)
        yy = np.linspace(500, EDGE[1] - 500, 5)
        zz = np.linspace(500, EDGE[2] - 500, 5)
        sig_c = []
        for (xp, yp, zp) in zip(xx, yy, zz):
            df_p = df[(
                (df["X"] >= xp - 400) & (df["X"] <= xp + 400) & 
                (df["Y"] >= yp - 400) & (df["Y"] <= yp + 400) & 
                (df["Z"] >= zp - 400) & (df["Z"] <= zp + 400) 
            )]
            sig_c.append(df_p.loc[:, 'sig_xx'].mean())

        # ∆ Retain mean value
        sigs.append(sig_c)
        
        peak_df = df[(
            (df["X"] >= EDGE[0] - 400) & 
            (df["Y"] >= 500) & (df["Y"] <= EDGE[1] - 500) & 
            (df["Z"] >= 100) & (df["Z"] <= EDGE[2] - 100) 
        )]
        
        # ∆ Save CSV
        df.to_csv(f"_csv/testgcc_{kk}.csv")  

        # ∆ Write files
        dis_file.write(ii)
        pre_file.write(ii)
        sig_file.write(ii)

    # ∆ Close files
    dis_file.close()
    pre_file.close()
    sig_file.close()

    return num_its, 1

# ∆ Main
def main(m_ref):

    # ∆ Test vals
    gcc = [1, 11, 10]
    
    # ∆ Mesh file
    file = f"_msh/em_{m_ref}.msh"

    # ∆ Simulation 
    _, win = fx_(file, gcc)

    print(f" ~> WIN: {win}")

# ∆ Inititate 
if __name__ == '__main__':

    # ∆ Indicate test cases
    r = 400
    main(r)