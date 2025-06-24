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

# ∆ Cauchy
def cauchy_tensor(u, p, Push, gcc):
    
    u_nu = Push * u

    # ∆ Kinematics
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u_nu)  
    C = ufl.variable(F.T * F)  
    E = ufl.variable(0.5*(C-I))

    # ∆ Output constants
    b0, bf, bt = gcc

    # ∆ Exponential term
    Q = (
        bf * E[0,0]**2 + bt * 
        (
            E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2 + 
            E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2
        )
    )

    # ∆ Seond Piola-Kirchoff 
    # SPK = b0/4 * ufl.exp(Q) * ufl.as_matrix([
    #     [4*bf*E[0,0], 2*bt*(E[1,0] + E[0,1]), 2*bt*(E[2,0] + E[0,2])],
    #     [2*bt*(E[0,1] + E[1,0]), 4*bt*E[1,1], 2*bt*(E[2,1] + E[1,2])],
    #     [2*bt*(E[0,2] + E[2,0]), 2*bt*(E[1,2] + E[2,1]), 4*bt*E[2,2]],
    # ]) 

    Ic = ufl.variable(ufl.tr(C))
    IIc = ufl.variable((Ic**2 - ufl.inner(C, C))/2)
    psi = 1 * (Ic - 3) + 0 *(IIc - 3) 
    term1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
    term2 = -ufl.diff(psi, IIc)
    SPK = 2 * F * (term1*I + term2*C) 

    return 1/ufl.det(F) * F * SPK * F.T 

# ∆ Fenics simulation
def fx_(t, file, m_ref, gcc, sim, dpm):

    # ∆ Begin 't' log
    with open(f"_txt/{sim}.txt", 'a') as simlog:
        simlog.write(f"test: {t}, gcc: {gcc}\n")

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
    R = ufl.as_tensor(SPK[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx

    log.set_log_level(log.LogLevel.INFO)

    # ∆ Data store
    dis, pre = Function(V), Function(P) 
    sig, eps = Function(Tes), Function(Tes)
    dis.name = "U - Displacement"
    pre.name = "P - Pressure"
    eps.name = "E - Green Strain"
    sig.name = "S - Cauchy Stress"
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

    for ii, kk in enumerate([20]):

        # ∆ Apply displacements as boundary conditions
        du = CUBE["x"] * PXLS["x"] * (kk / 100)
        print(f"APPLY {dpm}% DISP | val ~ {np.round(du,2)}")
        d_xy0 = dirichletbc(default_scalar_type(du), xx0, Mxs.sub(0).sub(X))
        d_xy1 = dirichletbc(default_scalar_type(0.0), xx1, Mxs.sub(0).sub(X))
        d_yy0 = dirichletbc(default_scalar_type(0.0), yx0, Mxs.sub(0).sub(Y))
        d_yy1 = dirichletbc(default_scalar_type(0.0), yx1, Mxs.sub(0).sub(Y))
        d_zy0 = dirichletbc(default_scalar_type(0.0), zx0, Mxs.sub(0).sub(Z))
        d_zy1 = dirichletbc(default_scalar_type(0.0), zx1, Mxs.sub(0).sub(Z))
        bc = [d_xy0, d_yy0, d_zy0, d_xy1, d_yy1, d_zy1]

        # ∆ Write displacement store
        with open(f"_txt/{sim}.txt", 'a') as simlog:
            simlog.write(f"APPLY {dpm}% DISP | val ~ {np.round(du,2)}\n")

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
            print(f"SOLVED {dpm}% IN:{num_its}, {res}")
            with open(f"_txt/{sim}.txt", 'a') as simlog:
                simlog.write(f"SOLVED {dpm}% IN:{num_its}, {res}\n")
        except:
            with open(f"_txt/{sim}.txt", 'a') as simlog:
                simlog.write(f"FAILED.\n")
            return -1, 0 
        
        # ∆ Evaluation
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        pre.interpolate(p_eval)
        # µ Evaluate stress
        cauchy = Expression(
            e=cauchy_tensor(u_eval, p_eval, Push, gcc), 
            X=Tes.element.interpolation_points()
        )
        # µ Evaluate strain
        green = Expression(e=E, X=Tes.element.interpolation_points())
        sig.interpolate(cauchy)
        eps.interpolate(green)
        
        # ∆ Format for saving
        sig_arr, eps_arr = sig.x.array, eps.x.array
        n_nodes = len(sig_arr) // DIM**2
        r_sig = sig_arr.reshape((n_nodes, DIM**2))
        r_eps = eps_arr.reshape((n_nodes, DIM**2))

        # ∆ Store data
        sigs = []
        coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
        df = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
                "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
                "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
                "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5]
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
        peak = (peak_df.loc[:, 'sig_xx'].mean()**2 + peak_df.loc[:, 'sig_xy'].mean()**2 + peak_df.loc[:, 'sig_xz'].mean()**2)**0.5
        
        
        # print(sig_c)
        print(peak)
        print(sigs)

        
        # ∆ Save CSV
        df.to_csv(f"_csv/{t}_{dpm}.csv")  

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
def main(tests, m_ref, sim, dpm):

    # ∆ Open log file
    with open(f"_txt/{sim}.txt", 'w') as simlog:
        simlog.write(f"tests: {tests}\nm_ref: {m_ref}\n")

    # ∆ Iterate tests 
    wins = []
    for t in tests:

        print("\t" + " ~> Test: {}".format(t))
        # ∆ Load constitutive values
        # gcc = [4, 10, 14]
        # gcc = [2.63, 23.22, 16.75] #23.42
        gcc = [1, 11, 10]
        # gcc = [4.667357969363915,  9.171759219672838,  49.651646465638066]
         
        # ∆ Mesh file
        file = f"_msh/em_{m_ref}.msh"

        # ∆ Simulation 
        _, win = fx_(t, file, m_ref, gcc, sim, dpm)
        if win: wins.append(t)

    # ∆ Append win dataz
    with open(f"_txt/{sim}.txt", 'a') as simlog:
        simlog.write(f"wins: {wins}")
    print(f" ~> WINS: {wins}")

# ∆ Inititate 
if __name__ == '__main__':

    # ∆ Indicate test cases
    # tests = ["test"] + [x for x in [str(y) for y in range(0, 17, 1)]]
    # ∆ Open log file

    # refs = [x for x in range(200, 3100, 100)]
    # refs = refs[::-1]
    # for r in refs:
    r = 200
    tests = ["test"]
    dpm = 20
    sim = "N_TEST" + str(dpm)
    main(tests, r, sim, dpm)