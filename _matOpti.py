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
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx import log, io,  default_scalar_type
from dolfinx.fem import Function, functionspace, Constant, dirichletbc, locate_dofs_topological, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# ∆ Seed
random.seed(17081993)

# ∆ Global Constants
DIM = 3
ORDER = 2 
TOL = 1e-5
RADIUS = 1000
QUADRATURE = 4
X, Y, Z = 0, 1, 2
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
    P, _ = Mxs.sub(1).collapse()

    # ∆ Determine coordinates of space and create mapping tree
    x_n = Function(V)
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    tree = KDTree(coords)

    # ∆ Setup functions for assignment
    ori, z_data = Function(V), Function(V)

    # ∆ Assign angles to dofs
    def angle_assign(coords):

        # ∆ Create arrays
        azi = np.zeros_like(coords[:, 0])
        ele = np.zeros_like(coords[:, 0]) 
        zs = np.zeros_like(coords[:, 0])
        return azi, ele, zs

    # ∆ Assign angle and z-disc data
    azi, ele, zs = angle_assign(coords)

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
    spk = b0/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*bf*E[0,0], 2*bt*(E[1,0] + E[0,1]), 2*bt*(E[2,0] + E[0,2])],
        [2*bt*(E[0,1] + E[1,0]), 4*bt*E[1,1], 2*bt*(E[2,1] + E[1,2])],
        [2*bt*(E[0,2] + E[2,0]), 2*bt*(E[1,2] + E[2,1]), 4*bt*E[2,2]],
    ])
    SPK = spk - p * G_v_inv

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

    # ∆ Iterate strain
    sig_xx_mean = []
    sigs = []
    sig = Function(Tes)

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
            # ∆ Close files
            return None
            
        # ∆ Evaluation
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        pre.interpolate(p_eval)

        if kk not in [20]:
            continue

        # µ Evaluate stress
        cauchy = Expression(
            e=cau, #cauchy_tensor(u_eval, gcc), 
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)
        
        ## ∆ Format for saving
        sig_arr = sig.x.array
        n_nodes = len(sig_arr) // DIM**2
        r_sig = sig_arr.reshape((n_nodes, DIM**2))

        # ∆ Store data
        coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
        df = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
                "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5]
            }
        )

        # ∆ Find centre sphere
        df['DS'] = (
            (df["X"] - EDGE[0] // 2)**2 +
            (df["Y"] - EDGE[1] // 2)**2 +
            (df["Z"] - EDGE[2] // 2)**2
        )
        peak_df = df[df['DS'] <= RADIUS**2]
        peak = abs(peak_df.loc[:, 'sig_xx'].mean())

        # ∆ Retain mean value
        sig_xx_mean.append(peak)

    # ∆ Update text files on progress 
    with open(f"opti_200.txt", "a") as f:
        for x in gcc:
            f.write(str(x)  +  "  ")
        for x in sig_xx_mean:
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
    r = 200
    file = f"_msh/em_{r}.msh"  
    pct = 20 

    # ∆ Experimental data to converge to
    eps_exp = np.array([0.20])
    sig_exp = np.array([8])
    bounds = [(0.5, 1.5), (5, 20), (5, 20)]
    init_gcc = [1, 11, 10]

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

    

