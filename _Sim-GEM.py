# Standard library imports
import random

# Third-party library imports
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns         # Added for plotting

# DolfinX related imports
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, io, log
from dolfinx.fem import Constant, Expression, Function, functionspace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

# Seed for reproducibility
random.seed(17081993)

# Global Constants
DIM = 3               # Dimension of the problem (e.g., 3 for 3D)
ORDER = 2             # Polynomial order for function spaces
TOL = 1e-5            # Solver tolerance
QUADRATURE = 4        # Quadrature degree for numerical integration
RADIUS = 1000         # Radius for sphere-based stress calculation

# Indices for coordinate axes
X, Y, Z = 0, 1, 2
# Pixel dimensions (used for scaling segmentation data)
PIXX, PIXY, PIXZ = 11, 11, 50
PXLS = {"x": 11, "y": 11, "z": 50} # Dictionary for pixel dimensions
# Cube dimensions (likely in physical units)
CUBE = {"x": 1000, "y": 1000, "z": 100}
# Calculated edge lengths of the domain
EDGE = [PXLS[d] * CUBE[d] for d in ["x", "y", "z"]]


# Smooth data globally
def global_smooth(coords, data):
    """Applies global smoothing to data using a Gaussian kernel."""

    tol = 1e-8
    sx, sy, sz = 1000, 500, 500 # Standard deviations for weighting

    tree = KDTree(coords) # Nearest neighbour tree

    s_data = np.zeros_like(data) # Store new data
    mask = (data != 0).astype(float) # Mask for non-zero data

    for i in range(len(coords)): # Iterate through data points
        kn_idx = tree.query_ball_point(coords[i], r=2*sx) # Find neighbours

        knn = coords[kn_idx]
        vals = data[kn_idx]
        knn_mask = mask[kn_idx]

        # Compute distances
        dx = knn[:, 0] - coords[i, 0]
        dy = knn[:, 1] - coords[i, 1]
        dz = knn[:, 2] - coords[i, 2]

        # Compute Gaussian weights
        wei = np.exp(-0.5 * ((dx / sx)**2 + (dy / sy)**2 + (dz / sz)**2))

        # Apply mask to weighted data and mask sums
        w_data = wei * vals * knn_mask
        w_mask = wei * knn_mask

        # Store smoothed value
        if w_mask.sum() > tol:
            s_data[i] = w_data.sum() / (w_mask.sum() + tol)
        else:
            s_data[i] = 0.0

    return s_data


# Assign angles to dofs
def angle_assign(t, coords):
    """
    Assigns orientation angles (azimuth, elevation) and a Z-disc indicator
    to mesh coordinates based on segmentation data.
    """
    # Create arrays for angles, Z-disc indicator, and normal vectors
    azi = np.zeros_like(coords[:, 0])
    ele = np.zeros_like(coords[:, 0])
    sph = np.zeros_like(coords[:, 0])
    zid = np.zeros_like(coords[:, 0])
    zs = np.zeros_like(coords[:, 0])
    nvecs = np.zeros_like(coords)

    if t == "test":
        return azi, ele, zs

    # Load tile data and unique IDs
    data_df = pd.read_csv(f"_csv/norm_stats_whole.csv")
    uni = data_df["ID"].to_list()

    # Load segmentation data
    data = np.load(f"_npy/seg_{t}.npy").astype(np.uint16)
    data = np.transpose(data, (1, 0, 2))

    # Scale segmentation indices to match physical dimensions
    scale = np.array([PIXX, PIXY, PIXZ])
    idxs = np.argwhere(data >= 0)
    m_idxs = idxs * scale

    # Generate nearest neighbours trees for segmentation data and mesh nodes
    knn = KDTree(m_idxs)
    node_tree = KDTree(coords)

    # Distance parameters for neighbor search and extension
    tol_dist = 1000
    sar_dist = 1400
    s_step = 200

    # Determine matching points between mesh coordinates and segmentation data
    dist, idx = knn.query(coords, distance_upper_bound=tol_dist)

    # Iterate through the distances and indices to assign angles
    for i, (dist_val, seg_idx) in enumerate(zip(dist, idx)):
        if dist_val >= tol_dist: # Skip if outside tolerance distance
            continue

        # Get 3D position and ID from segmentation data
        ix, iy, iz = m_idxs[seg_idx, :]
        seg_id = data[ix // PIXX, iy // PIXY, iz // PIXZ]

        if not seg_id or seg_id not in uni: # Check if ID is valid and included
            continue

        # Assign angle values from loaded data
        azi_val = data_df.loc[data_df["ID"] == seg_id, "Azi_[RAD]"].values[0]
        ele_val = data_df.loc[data_df["ID"] == seg_id, "Ele_[RAD]"].values[0]
        sph_val = data_df.loc[data_df["ID"] == seg_id, "Sph_[DEG]"].values[0]

        # Calculate vector in direction of angles and normalize
        nvec = np.array([
            ufl.cos(azi_val) * ufl.cos(ele_val),
            ufl.sin(azi_val) * ufl.cos(ele_val),
            -ufl.sin(ele_val)
        ])
        nvec /= np.linalg.norm(nvec) # Normalize vector

        # Assign values at the current mesh node and its neighbors
        pt = coords[i]
        kn_idx = node_tree.query_ball_point(pt, r=500)
        zs[kn_idx] = 1
        azi[kn_idx] = azi_val
        ele[kn_idx] = ele_val
        sph[kn_idx] = sph_val
        nvecs[kn_idx] = nvec
        zid[kn_idx] = 1

        # Extend assignment along directions of interest
        for a in np.linspace(-1, 1, s_step):
            pt_extended = coords[i] + a * sar_dist * nvec # Calculate extended point
            kn_idx_extended = node_tree.query_ball_point(pt_extended, r=500) # Find new points

            # Assign values to extended neighbors
            for j in kn_idx_extended:
                if not(a): # Only set zs if 'a' is zero (central point)
                    zs[j] = 1
                azi[j] = azi_val
                ele[j] = ele_val
                sph[j] = sph_val
                nvecs[j] = nvec
                zid[j] = 1

    # Apply global smoothing to the assigned angles
    azi = global_smooth(coords, azi)
    ele = global_smooth(coords, ele)
    sph = global_smooth(coords, sph)

    # Define a final smoothing pass helper function
    def final_pass(coords, data):
        tol = 1e-8
        sx, sy, sz = 500, 500, 500
        s_data = np.zeros_like(data)
        for i in range(len(coords)):
            dx = coords[:, 0] - coords[i, 0]
            dy = coords[:, 1] - coords[i, 1]
            dz = coords[:, 2] - coords[i, 2]
            wei = np.exp(-0.5 * ((dx / sx)**2 + (dy / sy)**2 + (dz / sz)**2))
            wei /= wei.sum() + tol
            s_data[i] = np.sum(wei * data)
        return s_data

    # Apply final smoothing pass
    azi = final_pass(coords, azi)
    ele = final_pass(coords, ele)
    sph = final_pass(coords, sph)

    return azi, ele, zs


def compute_principal_stresses(stress_components: np.ndarray) -> np.ndarray:
    """
    Computes the principal stresses (eigenvalues) from a 3D symmetric stress tensor.

    Args:
        stress_components (np.ndarray): A (N, 9) array where N is the number of points,
                                        and the 9 columns are [s_xx, s_xy, s_xz, s_yx, s_yy, s_yz, s_zx, s_zy, s_zz].

    Returns:
        np.ndarray: A (N, 3) array containing the three principal stresses for each point,
                    sorted in descending order (sigma_1, sigma_2, sigma_3).
    """
    num_points = stress_components.shape[0]
    principal_stresses = np.zeros((num_points, 3))
    principal_directions = np.zeros((num_points, 3, 3))

    for i in range(num_points):
        # Extract components from the flattened 9-component array (r_sig)
        s_xx = stress_components[i, 0]
        s_xy = stress_components[i, 1]
        s_xz = stress_components[i, 2]
        s_yy = stress_components[i, 4]
        s_yz = stress_components[i, 5]
        s_zz = stress_components[i, 8]

        # Construct the symmetric stress tensor for the current point
        stress_tensor = np.array([
            [s_xx, s_xy, s_xz],
            [s_xy, s_yy, s_yz], # s_yx is s_xy
            [s_xz, s_yz, s_zz]  # s_zx is s_xz, s_zy is s_yz
        ])

        # Compute eigenvalues (principal stresses)
        eigenvalues = np.linalg.eigvalsh(stress_tensor)
        

        eigenvalues, eigenvectors = np.linalg.eigh(stress_tensor)

        # Sort eigenvalues in descending order and sort eigenvectors accordingly
        sort_indices = np.argsort(eigenvalues)[::-1] # Get indices for descending sort
        principal_stresses[i, :] = eigenvalues[sort_indices]
        principal_directions[i, :, :] = eigenvectors[:, sort_indices] # Sort columns of eigenvectors
        # print(principal_directions)


        # Sort eigenvalues in descending order (sigma_1 >= sigma_2 >= sigma_3)
        principal_stresses[i, :] = np.sort(eigenvalues)[::-1]

    return principal_stresses


def plot_stress_distributions(df: pd.DataFrame, t: str, kk: int):
    """
    Generates and saves histograms for Cauchy stress components (sig_xx, sig_yy, sig_zz)
    and the maximum principal stress (sig_p1), plotting only the 90% confidence interval
    of the data for each.

    Args:
        df (pd.DataFrame): DataFrame containing stress results.
        t (str): Test case identifier.
        kk (int): Current load step percentage.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150) # Changed to 2x2 subplots
    axes = axes.flatten() # Flatten for easy iteration

    fig.suptitle(f"Stress Distributions for Test '{t}' at {kk}% Strain (90% Confidence Interval)", fontsize=18)

    # List of stress components to plot
    stress_cols = ['sig_xx', 'sig_yy', 'sig_zz', 'sig_p1']
    titles = [
        r'$\sigma_{xx}$ Distribution',
        r'$\sigma_{yy}$ Distribution',
        r'$\sigma_{zz}$ Distribution',
        r'Max Principal Stress ($\sigma_1$) Distribution'
    ]
    colors = ['skyblue', 'lightgreen', 'salmon', 'lightcoral']

    for i, col in enumerate(stress_cols):
        ax = axes[i]
        # Calculate 90% confidence interval
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Plot histogram
        sns.histplot(df_filtered[col], kde=True, ax=ax, color=colors[i], bins=50)
        ax.set_title(titles[i] + ' (90% CI)')
        ax.set_xlabel('Stress [kPa]')
        ax.set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.savefig(f"_png/stress_dist_{t}_{kk}_all_90CI.png", bbox_inches="tight")
    plt.close(fig) # Close


# Fenics simulation
def fx_(t, file, gcc, r):
    """
    Performs a FenicsX non-linear solid mechanics simulation with anisotropic
    material properties.

    Args:
        t (str): Test case identifier.
        file (str): Path to the mesh file.
        gcc (list): Global constitutive constants [b0, bf, bt].
        r (int): Refinement parameter, used for output file naming.

    Returns:
        tuple: (num_iterations, status_code) where status_code is 1 for success, -1 for failure.
    """
    # Load mesh data and set up function spaces
    domain, _, ft = io.gmshio.read_from_msh(filename=file, comm=MPI.COMM_WORLD, rank=0, gdim=DIM)
    P2 = element("Lagrange", domain.basix_cell(), ORDER, shape=(domain.geometry.dim,))
    P1 = element("Lagrange", domain.basix_cell(), ORDER - 1)
    Mxs = functionspace(mesh=domain, element=mixed_element([P2, P1]))
    Tes = functionspace(mesh=domain, element=("Lagrange", ORDER, (DIM, DIM)))

    # Define subdomains (collapsed function spaces for displacement and pressure)
    V, _ = Mxs.sub(0).collapse()
    P, _ = Mxs.sub(1).collapse()

    # Determine coordinates of space and create mapping tree
    x_n = Function(V)
    # Initial DOF coordinates are static for the simulation, so compute once
    coords = np.array(x_n.function_space.tabulate_dof_coordinates()[:])
    tree = KDTree(coords)

    # Setup functions for assignment
    ori, z_data = Function(V), Function(V)

    # Assign angle and z-disc data
    azi, ele, zs = angle_assign(t, coords)

    # Store cosine and sine components of angles
    CA, CE = np.cos(azi), np.cos(ele)
    SA, SE = np.sin(azi), np.sin(ele)

    # Create interpolate functions for basis vectors
    def nu_1(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        return np.array([CA[idx] * CE[idx], SA[idx] * CE[idx], -SE[idx]])

    def nu_2(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        return np.array([-SA[idx], CA[idx], np.zeros_like(CA[idx])])

    def nu_3(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        return np.array([CA[idx] * SE[idx], SA[idx] * SE[idx], CE[idx]])

    # Create z_disc id data
    z_arr = z_data.x.array.reshape(-1, 3)
    z_arr[:, 0], z_arr[:, 1], z_arr[:, 2] = zs, azi, ele
    z_data.x.array[:] = z_arr.reshape(-1)

    # Create angular orientation vector
    ori.interpolate(nu_1)

    # Create push tensor function
    Push = Function(Tes)

    # Define push interpolation (Forward transform)
    def forward(phi_xyz):
        _, idx = tree.query(phi_xyz.T, k=1)
        f00, f01, f02 = CA[idx] * CE[idx], -SA[idx], CA[idx] * SE[idx]
        f10, f11, f12 = SA[idx] * CE[idx], CA[idx], SA[idx] * SE[idx]
        f20, f21, f22 = -SE[idx], np.zeros_like(CE[idx]), CE[idx]
        return np.array([f00, f01, f02, f10, f11, f12, f20, f21, f22])

    # Interpolate Push as Forward transform
    Push.interpolate(forward)

    # Load key function variables for mixed space
    mx = Function(Mxs)
    v, q = ufl.TestFunctions(Mxs)
    u, p = ufl.split(mx)
    u_nu = Push * u

    # Kinematics Setup
    i, j, k, l, a, b = ufl.indices(6)
    I = ufl.Identity(DIM)
    F = ufl.variable(I + ufl.grad(u_nu))

    # Metric tensors
    # Underformed Covariant basis vectors
    A1, A2, A3 = Function(V), Function(V), Function(V)
    A1.interpolate(nu_1) # Create base 1
    A2.interpolate(nu_2) # Create base 2
    A3.interpolate(nu_3) # Create base 3

    # Underformed Metric tensors
    G_v = ufl.as_tensor([
        [ufl.dot(A1, A1), ufl.dot(A1, A2), ufl.dot(A1, A3)],
        [ufl.dot(A1, A2), ufl.dot(A2, A2), ufl.dot(A2, A3)],
        [ufl.dot(A1, A3), ufl.dot(A2, A3), ufl.dot(A3, A3)]
    ])
    G_v_inv = ufl.inv(G_v)
    # Deformed Metric covariant tensors
    g_v = ufl.as_tensor([
        [ufl.dot(F * A1, F * A1), ufl.dot(F * A1, F * A2), ufl.dot(F * A1, F * A3)],
        [ufl.dot(F * A2, F * A1), ufl.dot(F * A2, F * A2), ufl.dot(F * A2, F * A3)],
        [ufl.dot(F * A3, F * A1), ufl.dot(F * A3, F * A2), ufl.dot(F * A3, F * A3)]
    ])

    # Christoffel symbols
    Gamma = ufl.as_tensor(
        0.5 * G_v_inv[k, l] * (ufl.grad(G_v[j, l])[i] + ufl.grad(G_v[i, l])[j] - ufl.grad(G_v[i, j])[l]),
        (i, j, k)
    )

    # Covariant derivative
    covDev = ufl.as_tensor(ufl.grad(v)[i, j] + Gamma[i, k, j] * v[k], (i, j))

    # Kinematics Tensors
    C = ufl.variable(F.T * F) # Right Cauchy-Green deformation tensor
    B = ufl.variable(F * F.T) # Left Cauchy-Green deformation tensor (unused but kept)
    E = ufl.as_tensor(0.5 * (g_v - G_v)) # Green-Lagrange strain tensor
    J = ufl.det(F) # Determinant of deformation gradient

    # Extract Constitutive terms
    b0, bf, bt = gcc

    # Exponent term for strain energy density
    Q = (
        bf * E[0,0]**2 + bt *
        (
            E[1,1]**2 + E[2,2]**2 + E[1,2]**2 + E[2,1]**2 +
            E[0,1]**2 + E[1,0]**2 + E[0,2]**2 + E[2,0]**2
        )
    )

    # Second Piola-Kirchoff stress tensor
    spk = b0/4 * ufl.exp(Q) * ufl.as_matrix([
        [4*bf*E[0,0], 2*bt*(E[1,0] + E[0,1]), 2*bt*(E[2,0] + E[0,2])],
        [2*bt*(E[0,1] + E[1,0]), 4*bt*E[1,1], 2*bt*(E[2,1] + E[1,2])],
        [2*bt*(E[0,2] + E[2,0]), 2*bt*(E[1,2] + E[2,1]), 4*bt*E[2,2]],
    ])
    SPK = spk - p * G_v_inv # Add pressure term

    cau = 1/ufl.det(F) * F * spk * F.T # Cauchy stress

    # Residual (Weak form of equilibrium equations and incompressibility)
    dx = ufl.Measure(integral_type="dx", domain=domain, metadata={"quadrature_degree": QUADRATURE})
    R = ufl.as_tensor(SPK[a, b] * F[j, b] * covDev[j, a]) * dx + q * (J - 1) * dx

    # log.set_log_level(log.LogLevel.INFO) # Uncomment for detailed solver logs

    # Solver setup
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
    opts["ksp_monitor"] = None # Monitor KSP convergence
    ksp.setFromOptions()

    # Data storage for results
    dis, pre = Function(V), Function(P)
    sig, eps = Function(Tes), Function(Tes)
    dis.name = "U - Displacement"
    pre.name = "P - Pressure"
    sig.name = "S - Cauchy Stress"
    eps.name = "E - Green Strain"

    # VTX writers for visualization output
    dis_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_DISP.bp", dis, engine="BP4")
    pre_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_PRE.bp", pre, engine="BP4")
    sig_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_SIG.bp", sig, engine="BP4")
    eps_file = io.VTXWriter(MPI.COMM_WORLD, f"_bp/_{t}/_EPS.bp", eps, engine="BP4")

    # Setup boundary terms using mesh tags
    tgs_x0 = ft.find(3000)
    tgs_x1 = ft.find(3001)
    xx0 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x0)
    xx1 = locate_dofs_topological(Mxs.sub(0).sub(X), domain.topology.dim - 1, tgs_x1)
    yx0 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x0)
    yx1 = locate_dofs_topological(Mxs.sub(0).sub(Y), domain.topology.dim - 1, tgs_x1)
    zx0 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x0)
    zx1 = locate_dofs_topological(Mxs.sub(0).sub(Z), domain.topology.dim - 1, tgs_x1)

    # Set Dirichlet boundary conditions for displacement components
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

    # Determine points within the 3D space for stress sampling
    xx = np.flip(np.linspace(0.05 * EDGE[0], EDGE[0] - 0.05 * EDGE[0], 5))
    yy = np.linspace(0.05 * EDGE[1], EDGE[1] - 0.05 * EDGE[1], 5)
    zz = np.linspace(0.05 * EDGE[2], EDGE[2] - 0.05 * EDGE[2], 5)

    # Iterative solver loop for applying displacements
    for ii, kk in enumerate([0, 5, 10, 15, 20]):

        # Apply displacements as boundary conditions
        du = CUBE["x"] * PXLS["x"] * (kk / 100)
        du_pos.value = default_scalar_type(du // 2)
        du_neg.value = default_scalar_type(-du // 2)

        # Solve the nonlinear problem
        try:
            num_its, res = solver.solve(mx)
            print(f"SOLVED {kk}% IN:{num_its}, {res}")
        except Exception as e:
            print(f"Solver failed at {kk}%: {e}")
            # Close files on failure
            dis_file.close()
            pre_file.close()
            sig_file.close()
            eps_file.close()
            return -1, 0

        # Evaluate and interpolate results
        u_eval = mx.sub(0).collapse()
        p_eval = mx.sub(1).collapse()
        dis.interpolate(u_eval)
        pre.interpolate(p_eval)

        # Evaluate Cauchy stress
        cauchy = Expression(
            e=cau,
            X=Tes.element.interpolation_points()
        )
        sig.interpolate(cauchy)

        # Evaluate Green strain
        green = Expression(
            e=E,
            X=Tes.element.interpolation_points()
        )
        eps.interpolate(green)

        # Format stress and strain for saving
        sig_arr = sig.x.array
        eps_arr = eps.x.array
        n_nodes = len(sig_arr) // DIM**2
        r_sig = sig_arr.reshape((n_nodes, DIM**2))
        r_eps = eps_arr.reshape((n_nodes, DIM**2))

        # Compute principal stresses from Cauchy stress components
        principal_stresses = compute_principal_stresses(r_sig)

        # Format displacement for saving
        disp_arr = dis.x.array
        r_disp = disp_arr.reshape((len(disp_arr) // DIM), DIM)

        # Store data in pandas DataFrames
        df = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "sig_xx": r_sig[:, 0], "sig_yy": r_sig[:, 4], "sig_zz": r_sig[:, 8],
                "sig_xy": r_sig[:, 1], "sig_xz": r_sig[:, 2], "sig_yz": r_sig[:, 5],
                "eps_xx": r_eps[:, 0], "eps_yy": r_eps[:, 4], "eps_zz": r_eps[:, 8],
                "eps_xy": r_eps[:, 1], "eps_xz": r_eps[:, 2], "eps_yz": r_eps[:, 5],
                "sig_p1": principal_stresses[:, 0], # Max principal stress
                "sig_p2": principal_stresses[:, 1], # Intermediate principal stress
                "sig_p3": principal_stresses[:, 2]  # Min principal stress
            }
        )

        df_disp = pd.DataFrame(
            data={
                "X": coords[:, 0], "Y": coords[:, 1], "Z": coords[:, 2],
                "disp_x": r_disp[:, 0], "disp_y": r_disp[:, 1], "disp_z": r_disp[:, 2]
            }
        )

        # Generate and save immediate stress distribution plots
        plot_stress_distributions(df, t, kk)

        # Calculate mean stress at specific points for the current load step
        sigs = [] # Reset for each iteration to store current step's sample means
        for (xp, yp, zp) in zip(xx, yy, zz):
            df_p = df[(
                (df["X"] >= xp - 400) & (df["X"] <= xp + 400) &
                (df["Y"] >= yp - 400) & (df["Y"] <= yp + 400) &
                (df["Z"] >= zp - 400) & (df["Z"] <= zp + 400)
            )]
            sigs.append(df_p.loc[:, 'sig_xx'].mean())


        # Find mean stress within the central sphere
        df['DS'] = (
            (df["X"] - EDGE[0] // 2)**2 +
            (df["Y"] - EDGE[1] // 2)**2 +
            (df["Z"] - EDGE[2] // 2)**2
        )
        peak_df = df[df['DS'] <= RADIUS**2]
        peak = peak_df.loc[:, 'sig_xx'].mean()

        print(f"PT SIGS: {sigs} kPa")
        print(f"PEAK SIG: {peak} kPa")

        # Save CSV files
        df.to_csv(f"_csv/sim_{t}_{kk}_{r}.csv")
        df_disp.to_csv(f"_csv/dis_{t}_{kk}_{r}.csv")

        # Write VTX files for current iteration
        dis_file.write(ii)
        pre_file.write(ii)
        sig_file.write(ii)
        eps_file.write(ii)

    # Close all VTX files after simulation loop
    dis_file.close()
    pre_file.close()
    sig_file.close()
    eps_file.close()

    return num_its, 1


# Main simulation control function
def main(tests, m_ref):
    """
    Controls the overall simulation workflow, iterating through test cases.

    Args:
        tests (list): List of test case identifiers (e.g., ["test"]).
        m_ref (int): Mesh refinement parameter, used for output file naming.
    """
    # Constitutive model parameters [b0, bf, bt]
    # gcc = [1.19,  15.44,  16.80]
    gcc = [2.8107424166757204, 12.677385904668105, 11.039328556686252]

    wins = [] # List to track successful simulations
    for t in tests:
        file = f"_msh/em_{m_ref}.msh" # Mesh file path
        _, win_status = fx_(t, file, gcc, m_ref) # Run simulation
        if win_status:
            wins.append(t)

    print(f" ~> WIN: {wins}") # Print successful test cases

# Initiate simulation when script is run directly
if __name__ == '__main__':
    # Define test cases to run
    # refs = ["test"] + [x for x in range(0, 18, 1)] # Example for multiple tests
    refs = ["test"]

    r = 400 # Refinement parameter (passed to fx_, used for file naming)
    main(refs, r)
