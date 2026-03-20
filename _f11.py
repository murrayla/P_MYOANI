"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _angHist.py
        Create publicaiton figure 11 plots
"""

# . imports
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# . fig preset
plt.rcParams.update({
    'font.size': 10,          # Base font size for all text
    'axes.titlesize': 10,     # Title size
    'axes.labelsize': 10,     # Axis label size
    'xtick.labelsize': 10,    # X tick label size
    'ytick.labelsize': 10,    # Y tick label size
    'legend.fontsize': 10,    # Legend font size
    'legend.title_fontsize': 10  # Legend title size
})

# . Constants
Z_DISC = 14**3
RAW_SEG = "filtered.npy"
PIXX, PIXY, PIXZ = 11, 11, 50
M_D, E_D = 2545.58, 2545.58*2
Y_BOUNDS = [1800, 3600]
X_BOUNDS = [int((Y_BOUNDS[1]-M_D)), int(E_D-(Y_BOUNDS[1]-M_D))]
PIXX, PIXY, PIXZ = 11, 11, 50
PXLS = {"x": 11, "y": 11, "z": 50}
CUBE = {"x": 1000, "y": 1000, "z": 100}
EDGE = [PXLS[d]*CUBE[d] for d in ["x", "y", "z"]]
EXCLUSION = [335, 653, 775, 1108, 1406, 42, 185, 191, 335, 653, 775, 1108, 1406, 1674, 44, 136, 1652, 1732, 1744]
CUBE = {"x": 1000, "y": 1000, "z": 100}
regs = pd.read_csv("_csv/reg_.csv")
whole = pd.read_csv("_csv/rot_norm_w.csv")

def _oriStatsRegs():
    # List to store S values for plotting
    s_values = []
    regions = list(range(18))

    # Calculations
    for j in regions:
        df = pd.read_csv(f"_csv/tile_{j}_w.csv")
        vectors = np.zeros((len(df), 3))
        n = 0
        for i, row in df.iterrows():
            y, x, z = np.fromstring(row["PC3_ROT"].strip('[]'), sep=' ')
            vectors[n, :] = abs(x), y, z
            n += 1
            
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        u = vectors / (norms + 1e-9)
        T = np.dot(u.T, u) / len(u)
        eigenvalues, eigenvectors = np.linalg.eigh(T)
        idx = eigenvalues.argsort()[::-1]
        vals = eigenvalues[idx]
        S = (3 * vals[0] - 1) / 2
        s_values.append(S)
        print(f"Region {j} | Order Parameter (S): {S:.4f}")

    # --- PLOTTING WITH AXIS BREAK ---
    mm_to_inch = 1 / 25.4
    fig_w, fig_h = 120 * mm_to_inch, 80 * mm_to_inch  

    # Create two subplots sharing the x-axis
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(fig_w, fig_h), 
                                           dpi=300, gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.1) # Small gap between parts

    # Plot bars on both
    colors = sns.color_palette("tab20", 18)
    ax_top.bar(regions, s_values, color=colors, edgecolor='navy', zorder=3)
    ax_bottom.bar(regions, s_values, color=colors, edgecolor='navy', zorder=3)

    # Set the range for the break (Adjust these based on your data)
    ax_top.set_ylim(0.8, 1.05)    # Top part focus
    ax_bottom.set_ylim(0, 0.15)   # Bottom part (0-baseline)

    # Hide the spines between the two plots
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    # Add reference line to top plot
    ax_top.axhline(y=1, color='black', linestyle='--', label='Major Axis', zorder=2)

    # Add diagonal "break" lines
    d = .015 
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  
    kwargs.update(transform=ax_bottom.transAxes)  
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels and Styling
    ax_top.set_title("Order (S)")
    ax_bottom.set_xlabel("Region")
    ax_bottom.set_xticks(regions)
    
    # Common Y label centered between axes
    # fig.text(0.02, 0.5, 'Order (S)', va='center', rotation='vertical', fontsize=10)
    
    ax_top.legend(frameon=True, fontsize=7, loc='upper right')
    ax_top.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    ax_bottom.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    # Save
    plt.savefig("_png/SUP_S.png", bbox_inches='tight')

def _oriStats():

   # 1. Load the data
    # Replace 'orientations.csv' with your actual filename
    # df = pd.read_csv('_csv/rot_norm_w.csv')

    df = pd.read_csv(f"_csv/OUT_ns.csv")
    
    vectors = np.zeros((382, 3))
    n = 0
    for i, row in df.iterrows():

        if row["Sph_[DEG]"] > 30: continue

        x, y, _ = ast.literal_eval(row["Centroid"])

        if x <= 2500 and y <= 2500:

            y, x, z = np.fromstring(row["PC3_ROT"].strip('[]'), sep=' ')
            vectors[n, :] = abs(x), y, z
            n += 1
        

    # 2. Normalize vectors (Crucial for accurate S calculation)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    u = vectors / norms

    # 3. Build Second-Moment Orientation Tensor T
    # T_ij = avg(u_i * u_j)
    T = np.dot(u.T, u) / len(u)

    # 4. Compute Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(T)

    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    vals = eigenvalues[idx]
    vecs = eigenvectors[:, idx]

    # 5. Extract Director and Scalar Order Parameter (S)
    director = vecs[:, 0]
    S = (3 * vals[0] - 1) / 2

    # --- REPORTING RESULTS ---
    print("-" * 30)
    print(f"ORIENTATION REPORT")
    print("-" * 30)
    print(f"Director (n): {director}")
    print(f"Order Parameter (S): {S:.4f}")
    print(f"Eigenvalues: {vals}")
    print("-" * 30)
    print(vecs)

    # # 6. Visualization
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot a subset of original vectors for clarity (every 10th vector)
    # step = max(1, len(u) // 100) 
    # ax.quiver(0, 0, 0, u[::step, 0], u[::step, 1], u[::step, 2], 
    #         color='gray', alpha=0.3, label='Data Vectors')

    # # Plot the Director (Bold red line)
    # ax.quiver(0, 0, 0, director[0], director[1], director[2], 
    #         color='red', linewidth=3, length=1.5, label='Director (n)')
    # # Also plot the negative director since it's a symmetric axis
    # ax.quiver(0, 0, 0, -director[0], -director[1], -director[2], 
    #         color='red', linewidth=3, length=1.5)

    # ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    # ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    # ax.set_title(f'Nematic Alignment (S = {S:.3f})')
    # plt.legend()
    # plt.show()

# . Main
def main():

    _oriStatsRegs()

# . Inititate
if __name__ == "__main__":
    main()