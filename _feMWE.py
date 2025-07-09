"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _feSim.py
       finite element simulation with FENICS
"""

# ∆ Dolfin
import numpy as np
import ufl
import time
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx import log, io,  default_scalar_type, mesh
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import (Identity, grad, TestFunctions, split, det, as_tensor, variable, dx)

# ∆ Parameters
DIM = 3
ORDER = 2
TOL = 1e-5
QUAD_DEGREE = 4

t0 = time.time()

# ∆ Mesh
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
domain.name = "mesh"

# ∆ Function space
P2 = element("Lagrange", domain.basix_cell(), ORDER, shape=(domain.geometry.dim,))
P1 = element("Lagrange", domain.basix_cell(), ORDER-1)
Mxs = functionspace(mesh=domain, element=mixed_element([P2, P1]))
Tes = functionspace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))

# ∆ Define subdomains
V, _ = Mxs.sub(0).collapse()

t1 = time.time()

total = t1-t0
print(f"FIRST REGION [DOMAIN]: {total}")
t0 = time.time()

# ∆ Trial & Test
u_p = Function(Mxs)
v, q = TestFunctions(Mxs)
u, p = split(u_p)

# ∆ Kinematics
I = Identity(DIM)
F = variable(I + grad(u))
C = variable(F.T * F)
E = as_tensor(0.5 * (C - I))
J = det(F)

# ∆ Kinematics 
I = ufl.variable(ufl.Identity(DIM))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
Ic = ufl.variable(ufl.tr(C))
IIc = ufl.variable((Ic**2 - ufl.inner(C,C))/2)
J = ufl.variable(ufl.det(F))
psi = 1 * (Ic - 3) + 0 *(IIc - 3) 
gamma1 = ufl.diff(psi, Ic) + Ic * ufl.diff(psi, IIc)
gamma2 = -ufl.diff(psi, IIc)
firstPK = 2 * F * (gamma1*I + gamma2*C) + p * J * ufl.inv(F).T
cau = (1/J * firstPK * F).T

# ∆ Residual
dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUAD_DEGREE})
R = ufl.inner(ufl.grad(v), firstPK) * dx + q * (J - 1) * dx

total = t1-t0
print(f"SECOND REGION [RESIDUAL]: {total}")
t0 = time.time()

# ∆ Solver
problem = NonlinearProblem(R, u_p, [])
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

# ∆ Output
displacement = Function(V)
displacement.name = "Displacement"
xdmf = io.VTXWriter(domain.comm, "disp.bp", displacement, engine="BP4")

X, Y, Z = 0, 1, 2

def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], max(x[0]))

fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# ∆ Setup boundary terms
xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, facet_tag.find(1))
xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, facet_tag.find(2))
yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, facet_tag.find(1))
yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, facet_tag.find(2))
zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, facet_tag.find(1))
zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, facet_tag.find(2))

# ∆ Set boundaries
du = Constant(domain, default_scalar_type(0.0))
d_xy0 = dirichletbc(du, xx0, Mxs.sub(0).sub(X)) # Initial value doesn't matter much
d_xy1 = dirichletbc(default_scalar_type(0.0), xx1, Mxs.sub(0).sub(X))
d_yy0 = dirichletbc(default_scalar_type(0.0), yx0, Mxs.sub(0).sub(Y))
d_yy1 = dirichletbc(default_scalar_type(0.0), yx1, Mxs.sub(0).sub(Y))
d_zy0 = dirichletbc(default_scalar_type(0.0), zx0, Mxs.sub(0).sub(Z))
d_zy1 = dirichletbc(default_scalar_type(0.0), zx1, Mxs.sub(0).sub(Z))
bc = [d_xy0, d_yy0, d_zy0, d_xy1, d_yy1, d_zy1]
problem.bcs = bc

total = t1-t0
print(f"THIRD REGION [SOLVER SETUP]: {total}")


log.set_log_level(log.LogLevel.INFO)

# ∆ Iterate
for ii, kk in enumerate([0, 5, 10, 15, 20]):

    t0 = time.time()

    # ∆ Apply displacements as boundary conditions
    du.value = default_scalar_type(kk / 100)

    # ∆ Solve
    try:
        n_iter, conv = solver.solve(u_p)
        print(f"Solved {kk}%: iterations={n_iter}, residual={conv}")
    except RuntimeError:
        print(f"Failed to converge at {kk}%")
        break

    # ∆ Store output
    displacement.interpolate(u_p.sub(0).collapse())
    xdmf.write(ii)

    total = t1-t0
    print(f"FOURTH REGION [IT {ii}] [SOLVER SETUP]: {total}")

xdmf.close()
