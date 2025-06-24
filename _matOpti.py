"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _matOpti.py
       parameter optimiser for fe script
"""

# ∆ Raw
import os
import random
import numpy as np
import pandas as pd
import multiprocessing
from scipy.spatial import KDTree
from scipy.optimize import differential_evolution

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
QUADRATURE = 4
X, Y, Z = 0, 1, 2
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
    
# ∆ Cauchy
def cauchy_tensor(u, p, gcc):
    
    # ∆ Kinematics
    I = ufl.Identity(DIM)  
    F = I + ufl.grad(u)  
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
    SPK = b0/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*bf*E[0,0], 2*bt*(E[1,0] + E[0,1]), 2*bt*(E[2,0] + E[0,2])],
        [2*bt*(E[0,1] + E[1,0]), 4*bt*E[1,1], 2*bt*(E[2,1] + E[1,2])],
        [2*bt*(E[0,2] + E[2,0]), 2*bt*(E[1,2] + E[2,1]), 4*bt*E[2,2]],
    ])

    return 1/ufl.det(F) * F * SPK * F.T -  1/ufl.det(F) * F * p * F.T
    
# ∆ Run simulation
def run_simulation(gcc, env):
    depth = 1

    # ∆ Retain from environment setup
    domain = env["domain"]
    ft = env["ft"]
    Mxs = env["Mxs"]
    Tes = env["Tes"]

    # ∆ Define subdomains
    V, _ = Mxs.sub(0).collapse()

    # ∆ Determine coordinates of space and create mapping tree
    x_n = Function(V)
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    tree = KDTree(coords)
    
    # ∆ Push as Forward transform
    Push = ufl.Identity(DIM)

    # ∆ Variational terms
    print("\t" * depth + "+= Setup Variables")
    mx = Function(Mxs)
    v, q = ufl.TestFunctions(Mxs)
    u, p = ufl.split(mx)
    u_nu = Push * u

    # ∆ Kinematics Setup
    i, j, k, l, a, b = ufl.indices(6)  
    I = ufl.Identity(DIM)  
    F = ufl.variable(I + ufl.grad(u_nu))

    # ∆ Covariant derivative
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

    # ∆ Setup boundary terms
    tgs_x0 = ft.find(3000)
    tgs_x1 = ft.find(3001)
    xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x0)
    xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x1)
    yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x0)
    yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x1)
    zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x0)
    zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x1)

    # ∆ Iterate strain
    sig_xx_mean = []
    sigs = []
    sig = Function(Tes)

    for dmp in [0, 20]:
        # ∆ Apply displacements as boundary conditions
        du = CUBE["x"] * PXLS["x"] * (dmp / 100)
        d_xy0 = dirichletbc(default_scalar_type(du//2), xx0, Mxs.sub(0).sub(X))
        d_xy1 = dirichletbc(default_scalar_type(-du//2), xx1, Mxs.sub(0).sub(X))
        d_yy0 = dirichletbc(default_scalar_type(0.0), yx0, Mxs.sub(0).sub(Y))
        d_yy1 = dirichletbc(default_scalar_type(0.0), yx1, Mxs.sub(0).sub(Y))
        d_zy0 = dirichletbc(default_scalar_type(0.0), zx0, Mxs.sub(0).sub(Z))
        d_zy1 = dirichletbc(default_scalar_type(0.0), zx1, Mxs.sub(0).sub(Z))
        bc = [d_xy0, d_yy0, d_zy0, d_xy1, d_yy1, d_zy1]

        # ∆ Solver
        problem = NonlinearProblem(R, mx, bc)
        solver = NewtonSolver(domain.comm, problem)
        solver.atol = TOL
        solver.rtol = TOL
        solver.max_it = 20
        solver.convergence_criterion = "incremental"

        # ∆ Solve
        print(gcc)
        try:
            num_its, _ = solver.solve(mx)
            print("\t" * depth + " ... converged in {} its".format(num_its))
        except:
            return None
            
        # ∆ Evaluation
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        # µ Evaluate stress
        cauchy = Expression(
            e=cauchy_tensor(u_eval, p_eval, gcc), 
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        
        # ∆ Format for saving
        n_comps = 9
        sig_arr = sig.x.array
        n_nodes = len(sig_arr) // n_comps
        r_sig = sig_arr.reshape((n_nodes, n_comps))

        # ∆ Store data
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
                (df["X"] >= xp - 300) & (df["X"] <= xp + 300) & 
                (df["Y"] >= yp - 300) & (df["Y"] <= yp + 300) & 
                (df["Z"] >= zp - 300) & (df["Z"] <= zp + 300) 
            )]
            sig_c.append(df_p.loc[:, 'sig_xx'].mean())

        peak_df = df[(
            (df["X"] >= EDGE[0] - 400) & 
            (df["Y"] >= 500) & (df["Y"] <= EDGE[1] - 500) & 
            (df["Z"] >= 100) & (df["Z"] <= EDGE[2] - 100) 
        )]
        peak = (peak_df.loc[:, 'sig_xx'].mean()**2 + peak_df.loc[:, 'sig_xy'].mean()**2 + peak_df.loc[:, 'sig_xz'].mean()**2)**0.5
        

        # ∆ Retain mean value
        sig_xx_mean.append(peak)
        sigs.append(sig_c)
        print(peak)

    # ∆ Update text files on progress 
    # µ Mean data
    with open(f"gcc_opti_s.txt", "a") as f:
        for x in sig_xx_mean:
            f.write(str(x)  +  "  ")
        f.write("\n")
        f.close()
    # µ Constitutive datas
    with open(f"gcc_opti_c.txt", "a") as f:
        for x in gcc:
            f.write(str(x)  +  "  ")
        f.write("\n")
        f.close()

    return sig_xx_mean

# ∆ Error calculation
def error_function(gcc, stress_exp, env):

    sig_sim = run_simulation(gcc, env)
    if sig_sim is None:
        return 1e10

    mse = np.mean((sig_sim - stress_exp) ** 2)
    return mse

# ∆ Simulation environment
def setup_simulation_environment(file, order=2):

    # ∆ Domain
    domain, ct, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    P2 = element("Lagrange", domain.basix_cell(), order, shape=(domain.geometry.dim,))
    P1 = element("Lagrange", domain.basix_cell(), order - 1)
    Mxs = functionspace(domain, mixed_element([P2, P1]))
    Tes = functionspace(mesh=domain, element=("Lagrange", order, (DIM, DIM)))

    return {
        "domain": domain,
        "ct": ct,
        "ft": ft,
        "Mxs": Mxs,
        "Tes": Tes
    }

def my_callback(xk, convergence):
    print(f"Current best parameters: {xk}")
    print(f"Convergence: {convergence}")
    return False  # Return True to stop the optimization early


# ∆ Initiate
if __name__ == '__main__':
    
    # ∆ Test setup
    tnm = "test"  # Or your test name
    r = 1
    file = f"_msh/em_{r}.msh"  
    pct = 20 

    # ∆ Experimental data to converge to
    eps_exp = np.array([0.0, 0.20])
    sig_exp = np.array([0.0, 50])
    bounds = [(0.01, 5), (0.001, 50), (0.001, 50)]
    init_gcc = [1, 14, 10]

    # ∆ Setup
    env = setup_simulation_environment(file=file)
    n_cpu = multiprocessing.cpu_count()
    result = differential_evolution(
        error_function,
        bounds=bounds,
        args=(sig_exp, env),
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=17081993,
        workers=1, 
        updating='deferred',
        polish=True,
        callback=my_callback
    )

    # Save final result
    np.savetxt("best_parameters.txt", result.x)
    print("Optimization complete. Best parameters:")
    print(result.x)

    

