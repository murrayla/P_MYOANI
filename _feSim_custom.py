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
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, Expression, form
from dolfinx.fem.petsc import NonlinearProblem, create_matrix, create_vector, assemble_matrix, assemble_vector, apply_lifting, set_bc
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
def cauchy_tensor(u):

    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
    J = ufl.variable(ufl.det(F))
    c1 = 2
    c2 = 6
    psi = c1 * (Ic - 3) + c2 *(IIc - 3) 
    gamma1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
    gamma2 = -ufl.diff(psi, IIc)
    firstPK = 2 * F * (gamma1*I + gamma2*C)

    return 1/ufl.det(F) * firstPK * F.T 

# ∆ Fenics simulation
def fx_(t, file, m_ref):

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

    # ∆ Kinematics 
    I = ufl.variable(ufl.Identity(DIM))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    Ic = ufl.variable(ufl.tr(C))
    IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
    J = ufl.variable(ufl.det(F))
    c1 = 2
    c2 = 6
    psi = c1 * (Ic - 3) + c2 *(IIc - 3) 
    gamma1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
    gamma2 = -ufl.diff(psi, IIc)
    firstPK = 2 * F * (gamma1*I + gamma2*C) + p * J * ufl.inv(F).T
    cau = (1/J * firstPK * F).T

    # ∆ Residual
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUADRATURE})
    R = ufl.inner(ufl.grad(v), firstPK) * dx + q * (J - 1) * dx
    Rdiv = ufl.derivative(R, u)

    # ∆ Data store
    dis, pre = Function(V), Function(P) 
    sig, eps = Function(Tes), Function(Tes)
    dis.name = "U - Displacement"
    pre.name = "P - Pressure"
    eps.name = "E - Green Strain"
    sig.name = "S - Cauchy Stress"
    dis_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}_{m_ref}/_DISP.bp", dis, engine="BP4")
    pre_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}_{m_ref}/_PRE.bp", pre, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}_{m_ref}/_SIG.bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}_{m_ref}/_EPS.bp", eps, engine="BP4")

    # ∆ Setup boundary terms
    tgs_x0 = ft.find(3000)
    tgs_x1 = ft.find(3001)
    xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x0)
    xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x1)
    yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x0)
    yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x1)
    zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x0)
    zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x1)

    # ∆ Unmoving BCs
    d_yy0 = dirichletbc(default_scalar_type(0.0), yx0, Mxs.sub(0).sub(Y))
    d_yy1 = dirichletbc(default_scalar_type(0.0), yx1, Mxs.sub(0).sub(Y))
    d_zy0 = dirichletbc(default_scalar_type(0.0), zx0, Mxs.sub(0).sub(Z))
    d_zy1 = dirichletbc(default_scalar_type(0.0), zx1, Mxs.sub(0).sub(Z))

    # ∆ Apply moving BCs
    du = CUBE["x"] * PXLS["x"] * (20 / 100)
    d_xy0 = dirichletbc(default_scalar_type(du//2), xx0, Mxs.sub(0).sub(X))
    d_xy1 = dirichletbc(default_scalar_type(-du//2), xx1, Mxs.sub(0).sub(X))
    bc = [d_xy0, d_yy0, d_zy0, d_xy1, d_yy1, d_zy1]

    # ∆ Manually assign residual and jacobian terms
    res = form(R)
    jac = form(Rdiv)
    du = Function(V)

    # ∆ Form matrices
    L = create_vector(res)
    A = create_matrix(jac)

    solver = PETSc.KSP().create(domain.comm)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    opts = PETSc.Options()
    prefix = f"solver_{id(solver)}"
    solver.setOptionsPrefix(prefix)
    option_prefix = solver.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

    solver.setFromOptions()
    solver.setOperators(A)

    relaxation_param = 1  # full Newton's Method (<1 for restraint)
    max_it = int(1e3)
    tol = 1e-8

    def NewtonSolve(print_steps=True, print_solution=True):
        i = 0  # number of iterations of the Newton solver
        converged = False
        while i < max_it:
            # Assemble Jacobian and residual
            with L.localForm() as loc_L:
                loc_L.set(0)
            A.zeroEntries()
            assemble_matrix(A, jac, bcs=bc)
            A.assemble()
            assemble_vector(L, res)
            L.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            L.scale(-1)

            # Compute b - J(u_D-u_(i-1))
            apply_lifting(L, [jac], [bc], x0=[u.vector], scale=1)
            # Set dx|_bc = u_{i-1}-u_D
            set_bc(L, bc, u.vector, 1.0)
            L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                          mode=PETSc.ScatterMode.FORWARD)

            # Solve linear problem
            solver.solve(L, du.vector)
            du.x.scatter_forward()

            # Update u_{i+1} = u_i + relaxation_param * delta x_i
            u.x.array[:] += du.x.array[:]
            i += 1
            # Compute norm of update
            correction_norm = du.vector.norm(0)
            error_norm = L.norm(0)
            if print_steps:
                print(
                    f"    Iteration {i}: Correction norm {correction_norm}, Residual {error_norm}")

            if correction_norm < tol:
                converged = True
                break

        if print_solution:
            if converged:
                # (Residual norm {error_norm})")
                print(f"Solution reached in {i} iterations.")
            else:
                print(
                    f"No solution found after {i} iterations. Revert to previous solution and adjust solver parameters.")

        return converged, i

    log.set_log_level(log.LogLevel.INFO)

    converged, num_its = NewtonSolve()

    # while i < 50:
        
    #     # Assemble Jacobian and residual
    #     with L.localForm() as loc_L:
    #         loc_L.set(0)

    #     A.zeroEntries()
    #     assemble_matrix(A, jac, bcs=[bc])
    #     A.assemble()
        
    #     assemble_vector(L, res)
    #     L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    #     L.scale(-1)

    #     # Compute b - J(u_D-u_(i-1))
    #     apply_lifting(L, [jacobian], [[bc]], x0=[uh.x.petsc_vec], alpha=1)
    #     # Set du|_bc = u_{i-1}-u_D
    #     dolfinx.fem.petsc.set_bc(L, [bc], uh.x.petsc_vec, 1.0)
    #     L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    #     # Solve linear problem
    #     solver.solve(L, du.x.petsc_vec)
    #     du.x.scatter_forward()

    #     # Update u_{i+1} = u_i + delta u_i
    #     uh.x.array[:] += du.x.array
    #     i += 1

    #     # Compute norm of update
    #     correction_norm = du.x.petsc_vec.norm(0)

    #     # Compute L2 error comparing to the analytical solution
    #     L2_error.append(np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM)))
    #     du_norm.append(correction_norm)

    #     print(f"Iteration {i}: Correction norm {correction_norm}, L2 error: {L2_error[-1]}")
    #     if correction_norm < 1e-10:
    #         break

    # for ii, kk in enumerate([0, 5, 10, 15, 20]):

    #     # ∆ Apply displacements as boundary conditions
    #     du = CUBE["x"] * PXLS["x"] * (kk / 100)
    #     d_xy0 = dirichletbc(default_scalar_type(du//2), xx0, Mxs.sub(0).sub(X))
    #     d_xy1 = dirichletbc(default_scalar_type(-du//2), xx1, Mxs.sub(0).sub(X))
    #     bc = [d_xy0, d_yy0, d_zy0, d_xy1, d_yy1, d_zy1]

    #     # ∆ Solver
    #     problem = NonlinearProblem(R, mx, bc)
    #     solver = NewtonSolver(domain.comm, problem)
    #     solver.atol = TOL
    #     solver.rtol = TOL
    #     solver.max_it = 15
    #     solver.convergence_criterion = "incremental"

    #     # ∆ Solve
    #     try: 
    #         num_its, res = solver.solve(mx)
    #     except:
    #         return -1, 0 
        
    #     # ∆ Evaluation
    #     u_eval = mx.sub(0).collapse()
    #     p_eval = mx.sub(1).collapse()
    #     dis.interpolate(u_eval)
    #     pre.interpolate(p_eval)
    #     # µ Evaluate stress
    #     cauchy = Expression(
    #         e=cauchy_tensor(u_eval), 
    #         X=Tes.element.interpolation_points()
    #     )
    #     # µ Evaluate strain
    #     sig.interpolate(cauchy)
        
    #     # ∆ Format for saving
    #     sig_arr, eps_arr = sig.x.array, eps.x.array
    #     n_nodes = len(sig_arr) // DIM**2
    #     r_sig = sig_arr.reshape((n_nodes, DIM**2))
    #     r_eps = eps_arr.reshape((n_nodes, DIM**2))

    #     # ∆ Store data
    #     sigs = []
    #     coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    #     df = pd.DataFrame(
    #         data={
    #             "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
    #             "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
    #             "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
    #             "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
    #             "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5]
    #         }
    #     )

    #     # ∆ Determine points within the 3D space
    #     # µ 0 - corner ymax
    #     xx = np.linspace(500, EDGE[0] - 500, 5)
    #     yy = np.linspace(500, EDGE[1] - 500, 5)
    #     zz = np.linspace(500, EDGE[2] - 500, 5)
    #     sig_c = []
    #     for (xp, yp, zp) in zip(xx, yy, zz):
    #         df_p = df[(
    #             (df["X"] >= xp - 400) & (df["X"] <= xp + 400) & 
    #             (df["Y"] >= yp - 400) & (df["Y"] <= yp + 400) & 
    #             (df["Z"] >= zp - 400) & (df["Z"] <= zp + 400) 
    #         )]
    #         sig_c.append(df_p.loc[:, 'sig_xx'].mean())

    #     # ∆ Retain mean value
    #     sigs.append(sig_c)

    #     # ∆ Save CSV
    #     df.to_csv(f"_csv/{t}_{kk}_{m_ref}.csv")  

    #     # ∆ Write files
    #     dis_file.write(ii)
    #     pre_file.write(ii)
    #     sig_file.write(ii)
    #     eps_file.write(ii)

    # # ∆ Close files
    # dis_file.close()
    # pre_file.close()
    # sig_file.close()
    # eps_file.close()

    return num_its, 1

# ∆ Main
def main(m_ref):

    # ∆ Mesh file
    file = f"_msh/em_{m_ref}.msh"

    # ∆ Simulation 
    _, win = fx_("test", file, m_ref)

    # ∆ Append win dataz
    print(f" ~> WINS: {win}")

# ∆ Inititate 
if __name__ == '__main__':

    r = 200
    main(r)