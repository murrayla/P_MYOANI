"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _angHist.py
        Create publicaiton figure 9 plots
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

def _sigDisp():

    # . setup
    d_comps = ["x", "y", "z"]
    s_comps = ["xy", "xz", "yz"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, s_tensor, d_tensor = 20, "sig", "disp"
    sig_list = []
    disp_list = []
    
    # . load and process data
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
    t_means = {c: df_test_r[f"{s_tensor}_{c}"].mean() for c in s_comps}

    region_stats = []
    for tt in range(18):
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        df_r = df[
            (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
        
        # Calculate Means AND Standard Deviations
        stat = {
            'Region': f'R{tt}', 
            'index': tt
        }
        
        for c in s_comps:
            data = df_r[f"{s_tensor}_{c}"]
            data = data[data.between(-20, 20)]
            stat[f'$\sigma$_{c}'] = data.mean() + t_means[c]

        for c in d_comps:
            data = df_r[f"{d_tensor}_{c}"]
            stat[f'disp_{c}'] = data.mean()
            
        region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats) 

    # . figure setup
    fig, axes = plt.subplots(3, 3, figsize=(6.9, 6.9), dpi=300)
    
    for row_idx, d_comp in enumerate(d_comps):    
        for col_idx, s_comp in enumerate(s_comps):
            ax = axes[row_idx, col_idx]
            y_col = f'$\sigma$_{s_comp}'
            x_row = f'disp_{d_comp}'

            # 2. Plot Scatters (Larger markers)d
            sns.scatterplot(
                data=_df, x=x_row, y=y_col, hue='Region', 
                palette=regasi, s=90, ax=ax, legend=False, zorder=3
            )
            
            # 3. Trend line (High visibility)
            x_vals = np.array([-100, 100])
            res = linregress(_df[x_row], _df[y_col])
            ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                    linestyle='--', alpha=1.0, lw=1)
            
            # 4. Equation text (12pt)
            eqn_text = f"$y={res.slope:.2f}x + {res.intercept:.2f}$\n$R={res.rvalue:.2f}$"
            ax.text(0.05, 0.95, eqn_text, transform=ax.transAxes, fontsize=8, 
                    fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # . format
            # ax.set_xlim(-8, 8)
            if col_idx == 0: 
                ax.set_ylim(-1.5, 1.5)
                ax.set_yticks(np.arange(-1.5, 2, 0.5))
            elif col_idx == 1: 
                ax.set_ylim(-1.5, 1.5)
                ax.set_yticks(np.arange(-1.5, 2, 0.5))
            elif col_idx == 2: 
                ax.set_ylim(-0.3, 0.3)
                ax.set_yticks(np.arange(-0.3, 0.4, 0.1))

            # . strip text
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)

    # . adjust layout for maximum space between plots
    plt.tight_layout(pad=1.0, w_pad=1.0)
    plt.savefig("_png/PUB_SD.png", bbox_inches='tight')

# . scatter and trends for stress and orientation
def _sigsSepOris():

    # . setup
    sns.set_style("whitegrid")
    comps = ["xy", "xz", "yz"]
    vars = ["Significance (p)", "Length", "Width", "Depth"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "sig"
    
    # . load
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
    t_means = {c: df_test_r[f"{tensor}_{c}"].mean() for c in ["xx"] + comps}

    # . loop tests
    for tt in range(18):

        region_stats = []
        
        # . extract stress data
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        
        # . load angles
        ids = pd.read_csv(f"_csv/tile_{tt}_w.csv")["ID"].values
        angs = whole.loc[whole["ID"].isin(ids), "Sph_[DEG]"]
        cents = whole.loc[whole["ID"].isin(ids), "Centroid"]

        # . store data
        for a, m in zip(angs, cents):
            stat = {'Region': f'R{tt}', '[$\degree$]': a, 'index': tt}
            mx, my, mz = ast.literal_eval(m)
            x, y, z = (mx - regs["x"][tt]) * PIXX, (my - regs["y"][tt]) * PIXY, (mz - regs["z"][tt]) * PIXZ
            df_r = df[
                (df["X"].between(x - 1000, x + 1000)) &
                (df["Y"].between(y - 1000, y + 1000)) &
                (df["Z"].between(z - 1000, z + 1000))
            ]
            for c in comps:
                data = df_r[f"{tensor}_{c}"]
                data = data[data.between(-20, 20)]
                stat[f'$\sigma$_{c}'] = data.mean() #- t_means[c]
            region_stats.append(stat)
    
        _df = pd.DataFrame(region_stats)

        # . figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300, sharex=True)
        

        # . type
        df_plot = _df.copy()

        # . iterate plot columns
        for col_idx, comp in enumerate(comps):
            ax = axes[col_idx]
            y_col = f'$\sigma$_{comp}'
            
            # . plot
            sns.scatterplot(data=df_plot, x='[$\degree$]', y=y_col, hue='Region', 
                            palette=regasi, s=100, ax=ax, legend=(col_idx == 2))
            
            # . plot trends
            styles = ['-', '--', '-.', ':']
            x_vals = np.array([-7.5, 7.5])
            eqn_texts = []
            res = linregress(df_plot['[$\degree$]'], df_plot[y_col])
            ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                    linestyle=styles[0], alpha=0.6, lw=1.5)
            eqn_texts.append(f"$y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")

            # . format
            # if col_idx == 0: ax.set_ylim(-1.5, 0.5)
            # elif col_idx == 1: ax.set_ylim(-1.5, 1)
            # elif col_idx == 2: ax.set_ylim(-0.5, 0.5)
            # ax.set_xlim(-7.5, 7.5)
            ax.text(0.05, 0.95, "\n".join(eqn_texts), transform=ax.transAxes, fontsize=8, 
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.set_title(f"Component: {comp.upper()}")
            if col_idx == 0: ax.set_ylabel(f"[kPa]")

        # Handle Row Legends
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].get_legend().remove()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 1.0), ncol=1, fontsize='small')

        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        plt.savefig(f"_png/sup_sigsSepOris_{tt}.png", bbox_inches='tight')
        plt.close()

# . scatter and trends for stress and orientation
def _dispsOris():

    # . setup
    sns.set_style("whitegrid")
    comps = ["x", "y", "z"]
    vars = ["Significance (p)", "Length", "Width", "Depth"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "disp"
    
    # . define groups
    sig_list = [0, 1, 5, 6, 8, 9, 12, 13, 17]
    x_bins = {tuple([0, 1, 2, 9, 10, 11]): "x >= 22", tuple([3, 4, 5, 12, 13, 14]): "11 >= x < 22", tuple([6, 7, 8, 15, 16, 17]): "x < 11"}
    y_bins = {tuple([0, 1, 2, 3, 4, 5, 6, 7, 8]): "y >= 9.9", tuple([9, 10, 11, 12, 13, 14, 15, 16, 17]): "y < 9.9"}
    z_bins = {tuple([0, 3, 6, 9, 12, 15]): "z >= 10", tuple([1, 4, 7, 10, 13, 16]): "5 >= z < 10", tuple([2, 5, 8, 11, 14, 17]): "z < 5"}
    
    # . load
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]

    # . loop tests
    region_stats = []
    for tt in range(18):
        
        # . extract stress data
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        
        # . load angles
        ids = pd.read_csv(f"_csv/tile_{tt}_w.csv")["ID"].values
        angs = whole.loc[whole["ID"].isin(ids), "Sph_[DEG]"]
        cents = whole.loc[whole["ID"].isin(ids), "Centroid"]

        # . store data
        for a, m in zip(angs, cents):
            stat = {'Region': f'R{tt}', '[$\degree$]': a, 'index': tt}
            mx, my, mz = ast.literal_eval(m)
            x, y, z = (mx - regs["x"][tt]) * PIXX, (my - regs["y"][tt]) * PIXY, (mz - regs["z"][tt]) * PIXZ
            df_r = df[
                (df["X"].between(x - 500, x + 500)) &
                (df["Y"].between(y - 500, y + 500)) &
                (df["Z"].between(z - 500, z + 500))
            ]
            for c in comps:
                data = df_r[f"{tensor}_{c}"]
                # data = data[data.between(-20, 20)]
                stat[f'disp_{c}'] = data.mean() 
            region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats)

    # . figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=300, sharex=True)
    
    # . iterate plots rows
    for row_idx, var_name in enumerate(vars):

        # . type
        df_plot = _df.copy()
        if row_idx == 0: df_plot['Group'] = df_plot['index'].apply(lambda i: "p < 0.05" if i in sig_list else "p >= 0.05")
        elif row_idx == 1: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in x_bins.items() if i in k), "Other"))
        elif row_idx == 2: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in y_bins.items() if i in k), "Other"))
        elif row_idx == 3: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in z_bins.items() if i in k), "Other"))

        # . iterate plot columns
        for col_idx, comp in enumerate(comps):
            ax = axes[row_idx, col_idx]
            y_col = f'disp_{comp}'
            
            # . plot
            sns.scatterplot(data=df_plot, x='[$\degree$]', y=y_col, hue='Region', style='Group', 
                            palette=regasi, s=100, ax=ax, legend=(col_idx == 2))
            
            # . plot trends
            styles = ['-', '--', '-.', ':']
            x_vals = np.array([-7.5, 7.5])
            eqn_texts = []
            if not(row_idx):
                res = linregress(df_plot['[$\degree$]'], df_plot[y_col])
                ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                        linestyle=styles[0], alpha=0.6, lw=1.5)
                eqn_texts.append(f"$y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")
            else:
                for g_idx, (g_name, g_df) in enumerate(df_plot.groupby('Group')):
                    if len(g_df) > 1:
                        res = linregress(g_df['[$\degree$]'], g_df[y_col])
                        ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                                linestyle=styles[g_idx % 4], alpha=0.6, lw=1.5)
                        eqn_texts.append(f"{g_name}: $y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")

            # . format
            # if col_idx == 0: ax.set_ylim(-1.5, 0.5)
            # elif col_idx == 1: ax.set_ylim(-1.5, 1)
            # elif col_idx == 2: ax.set_ylim(-0.5, 0.5)
            # ax.set_xlim(-7.5, 7.5)
            ax.text(0.05, 0.95, "\n".join(eqn_texts), transform=ax.transAxes, fontsize=8, 
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if row_idx == 0: ax.set_title(f"Component: {comp.upper()}")
            if col_idx == 0: ax.set_ylabel(f"{var_name}\n[nm]")

        # Handle Row Legends
        handles, labels = axes[row_idx, 2].get_legend_handles_labels()
        axes[row_idx, 2].get_legend().remove()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 1.0 - (row_idx+1)*0.22), ncol=1, fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig("_png/sup_dispsOris.png", bbox_inches='tight')

def _sigsOrisPrincipal():
    # . setup
    sns.set_style("whitegrid")
    # We now look at Principal Stresses 1, 2, and 3
    comps = ["1", "2", "3"]
    vars = ["Significance (p)", "Length", "Width", "Depth"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "sig"
    
    # . define groups (keeping your original logic)
    sig_list = [0, 1, 5, 6, 8, 9, 12, 13, 17]
    x_bins = {tuple([0, 1, 2, 9, 10, 11]): "x >= 22", tuple([3, 4, 5, 12, 13, 14]): "11 >= x < 22", tuple([6, 7, 8, 15, 16, 17]): "x < 11"}
    y_bins = {tuple([0, 1, 2, 3, 4, 5, 6, 7, 8]): "y >= 9.9", tuple([9, 10, 11, 12, 13, 14, 15, 16, 17]): "y < 9.9"}
    z_bins = {tuple([0, 3, 6, 9, 12, 15]): "z >= 10", tuple([1, 4, 7, 10, 13, 16]): "5 >= z < 10", tuple([2, 5, 8, 11, 14, 17]): "z < 5"}
    
    region_stats = []
    
    # . loop tests
    for tt in range(18):
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        
        # . load angles and centroids
        ids = pd.read_csv(f"_csv/tile_{tt}_w.csv")["ID"].values
        # 'whole' is assumed to be defined globally as in your snippet
        angs = whole.loc[whole["ID"].isin(ids), "Sph_[DEG]"]
        cents = whole.loc[whole["ID"].isin(ids), "Centroid"]

        for a, m in zip(angs, cents):
            stat = {'Region': f'R{tt}', '[$\degree$]': a, 'index': tt}
            mx, my, mz = ast.literal_eval(m)
            x, y, z = (mx - regs["x"][tt]) * PIXX, (my - regs["y"][tt]) * PIXY, (mz - regs["z"][tt]) * PIXZ
            
            # Spatial filter
            df_r = df[
                (df["X"].between(x - 1000, x + 1000)) &
                (df["Y"].between(y - 1000, y + 1000)) &
                (df["Z"].between(z - 1000, z + 1000))
            ]

            if not df_r.empty:
                # 1. Get means of all 6 stress components
                c_list = ["xx", "yy", "zz", "xy", "xz", "yz"]
                m_vals = {c: df_r[f"{tensor}_{c}"].mean() for c in c_list}

                # 2. Build the symmetric Stress Tensor matrix
                # Matrix format: [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
                matrix = [
                    [m_vals["xx"], m_vals["xy"], m_vals["xz"]],
                    [m_vals["xy"], m_vals["yy"], m_vals["yz"]],
                    [m_vals["xz"], m_vals["yz"], m_vals["zz"]]
                ]

                # 3. Calculate eigenvalues (Principal Stresses)
                # eigvalsh is for symmetric matrices; returns sorted [min ... max]
                evals = np.linalg.eigvalsh(matrix)
                
                # 4. Map to Sigma 1 (Max), 2 (Mid), 3 (Min)
                stat['$\sigma_1$'] = evals[2]
                stat['$\sigma_2$'] = evals[1]
                stat['$\sigma_3$'] = evals[0]
                
                region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats)

    # . figure setup
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=300, sharex=True)
    
    for row_idx, var_name in enumerate(vars):
        df_plot = _df.copy()
        # Grouping logic
        if row_idx == 0: df_plot['Group'] = df_plot['index'].apply(lambda i: "p < 0.05" if i in sig_list else "p >= 0.05")
        elif row_idx == 1: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in x_bins.items() if i in k), "Other"))
        elif row_idx == 2: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in y_bins.items() if i in k), "Other"))
        elif row_idx == 3: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in z_bins.items() if i in k), "Other"))

        for col_idx, comp in enumerate(comps):
            ax = axes[row_idx, col_idx]
            y_col = f'$\sigma_{comp}$'
            
            sns.scatterplot(data=df_plot, x='[$\degree$]', y=y_col, hue='Region', style='Group', 
                            palette=regasi, s=100, ax=ax, legend=(col_idx == 2))
            
            # Trends
            styles = ['-', '--', '-.', ':']
            x_vals = np.array([df_plot['[$\degree$]'].min(), df_plot['[$\degree$]'].max()])
            eqn_texts = []
            
            if row_idx == 0:
                res = linregress(df_plot['[$\degree$]'], df_plot[y_col])
                ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', alpha=0.6)
                eqn_texts.append(f"$y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")
            else:
                for g_idx, (g_name, g_df) in enumerate(df_plot.groupby('Group')):
                    if len(g_df) > 1:
                        res = linregress(g_df['[$\degree$]'], g_df[y_col])
                        ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                                linestyle=styles[g_idx % 4], alpha=0.6)
                        eqn_texts.append(f"{g_name}: $R={res.rvalue:.2f}$")

            ax.text(0.05, 0.95, "\n".join(eqn_texts), transform=ax.transAxes, fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if row_idx == 0: ax.set_title(f"Principal Stress $\sigma_{comp}$")
            if col_idx == 0: ax.set_ylabel(f"{var_name}\n[kPa]")

        # Clean up legends per row
        handles, labels = axes[row_idx, 2].get_legend_handles_labels()
        axes[row_idx, 2].get_legend().remove()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.85 - row_idx*0.22), ncol=1, fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig("_png/sup_sigsOrisPri.png", bbox_inches='tight')

# . scatter and trends for stress and orientation
def _sigsOris():

    # . setup
    sns.set_style("whitegrid")
    comps = ["xy", "xz", "yz"]
    vars = ["Significance (p)", "Length", "Width", "Depth"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "sig"
    
    # . define groups
    sig_list = [0, 1, 5, 6, 8, 9, 12, 13, 17]
    x_bins = {tuple([0, 1, 2, 9, 10, 11]): "x >= 22", tuple([3, 4, 5, 12, 13, 14]): "11 >= x < 22", tuple([6, 7, 8, 15, 16, 17]): "x < 11"}
    y_bins = {tuple([0, 1, 2, 3, 4, 5, 6, 7, 8]): "y >= 9.9", tuple([9, 10, 11, 12, 13, 14, 15, 16, 17]): "y < 9.9"}
    z_bins = {tuple([0, 3, 6, 9, 12, 15]): "z >= 10", tuple([1, 4, 7, 10, 13, 16]): "5 >= z < 10", tuple([2, 5, 8, 11, 14, 17]): "z < 5"}
    
    # . load
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
    t_means = {c: df_test_r[f"{tensor}_{c}"].mean() for c in ["xx"] + comps}

    # . loop tests
    region_stats = []
    for tt in range(18):
        
        # . extract stress data
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        
        # . load angles
        ids = pd.read_csv(f"_csv/tile_{tt}_w.csv")["ID"].values
        angs = whole.loc[whole["ID"].isin(ids), "Sph_[DEG]"]
        cents = whole.loc[whole["ID"].isin(ids), "Centroid"]

        # . store data
        for a, m in zip(angs, cents):
            stat = {'Region': f'R{tt}', '[$\degree$]': a, 'index': tt}
            mx, my, mz = ast.literal_eval(m)
            x, y, z = (mx - regs["x"][tt]) * PIXX, (my - regs["y"][tt]) * PIXY, (mz - regs["z"][tt]) * PIXZ
            df_r = df[
                (df["X"].between(x - 500, x + 500)) &
                (df["Y"].between(y - 500, y + 500)) &
                (df["Z"].between(z - 500, z + 500))
            ]
            for c in comps:
                data = df_r[f"{tensor}_{c}"]
                data = data[data.between(-20, 20)]
                stat[f'$\sigma$_{c}'] = data.mean() #- t_means[c]
            region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats)

    # . figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=300, sharex=True)
    
    # . iterate plots rows
    for row_idx, var_name in enumerate(vars):

        # . type
        df_plot = _df.copy()
        if row_idx == 0: df_plot['Group'] = df_plot['index'].apply(lambda i: "p < 0.05" if i in sig_list else "p >= 0.05")
        elif row_idx == 1: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in x_bins.items() if i in k), "Other"))
        elif row_idx == 2: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in y_bins.items() if i in k), "Other"))
        elif row_idx == 3: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in z_bins.items() if i in k), "Other"))

        # . iterate plot columns
        for col_idx, comp in enumerate(comps):
            ax = axes[row_idx, col_idx]
            y_col = f'$\sigma$_{comp}'
            
            # . plot
            sns.scatterplot(data=df_plot, x='[$\degree$]', y=y_col, hue='Region', style='Group', 
                            palette=regasi, s=100, ax=ax, legend=(col_idx == 2))
            
            # . plot trends
            styles = ['-', '--', '-.', ':']
            x_vals = np.array([-7.5, 7.5])
            eqn_texts = []
            if not(row_idx):
                res = linregress(df_plot['[$\degree$]'], df_plot[y_col])
                ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                        linestyle=styles[0], alpha=0.6, lw=1.5)
                eqn_texts.append(f"$y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")
            else:
                for g_idx, (g_name, g_df) in enumerate(df_plot.groupby('Group')):
                    if len(g_df) > 1:
                        res = linregress(g_df['[$\degree$]'], g_df[y_col])
                        ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                                linestyle=styles[g_idx % 4], alpha=0.6, lw=1.5)
                        eqn_texts.append(f"{g_name}: $y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")

            # . format
            # if col_idx == 0: ax.set_ylim(-1.5, 0.5)
            # elif col_idx == 1: ax.set_ylim(-1.5, 1)
            # elif col_idx == 2: ax.set_ylim(-0.5, 0.5)
            # ax.set_xlim(-7.5, 7.5)
            ax.text(0.05, 0.95, "\n".join(eqn_texts), transform=ax.transAxes, fontsize=8, 
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if row_idx == 0: ax.set_title(f"Component: {comp.upper()}")
            if col_idx == 0: ax.set_ylabel(f"{var_name}\n[kPa]")

        # Handle Row Legends
        handles, labels = axes[row_idx, 2].get_legend_handles_labels()
        axes[row_idx, 2].get_legend().remove()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 1.0 - (row_idx+1)*0.22), ncol=1, fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig("_png/sup_sigsOris.png", bbox_inches='tight')

# . scatter and trends for stress and orientation
def _sigOri():

    # . setup
    sns.set_style("whitegrid")
    comps = ["xy", "xz", "yz"]
    vars = ["Significance (p)", "Length", "Width", "Depth"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "sig"
    
    # . define groups
    sig_list = [0, 1, 5, 6, 8, 9, 12, 13, 17]
    x_bins = {tuple([0, 1, 2, 9, 10, 11]): "x >= 22", tuple([3, 4, 5, 12, 13, 14]): "11 >= x < 22", tuple([6, 7, 8, 15, 16, 17]): "x < 11"}
    y_bins = {tuple([0, 1, 2, 3, 4, 5, 6, 7, 8]): "y >= 9.9", tuple([9, 10, 11, 12, 13, 14, 15, 16, 17]): "y < 9.9"}
    z_bins = {tuple([0, 3, 6, 9, 12, 15]): "z >= 10", tuple([1, 4, 7, 10, 13, 16]): "5 >= z < 10", tuple([2, 5, 8, 11, 14, 17]): "z < 5"}
    
    # . load
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
    t_means = {c: df_test_r[f"{tensor}_{c}"].mean() for c in ["xx"] + comps}

    # . loop tests
    region_stats = []
    for tt in range(18):
        
        # . extract stress data
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        df_r = df[
            (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
        
        # . load angles
        df_tile = pd.read_csv(f"_csv/tile_{tt}_w.csv")
        ang = df_tile["Sph_[DEG]"].dropna()
        mean_ang = ang[ang.abs() <= 30].mean()

        # . store data
        stat = {'Region': f'R{tt}', '[$\degree$]': mean_ang, 'index': tt}
        for c in comps:
            data = df_r[f"{tensor}_{c}"]
            data = data[data.between(-20, 20)]
            stat[f'$\sigma$_{c}'] = data.mean() - t_means[c]
        region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats)

    # . figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=300, sharex=True)
    
    # . iterate plots rows
    for row_idx, var_name in enumerate(vars):

        # . type
        df_plot = _df.copy()
        if row_idx == 0: df_plot['Group'] = df_plot['index'].apply(lambda i: "p < 0.05" if i in sig_list else "p >= 0.05")
        elif row_idx == 1: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in x_bins.items() if i in k), "Other"))
        elif row_idx == 2: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in y_bins.items() if i in k), "Other"))
        elif row_idx == 3: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in z_bins.items() if i in k), "Other"))

        # . iterate plot columns
        for col_idx, comp in enumerate(comps):
            ax = axes[row_idx, col_idx]
            y_col = f'$\sigma$_{comp}'
            
            # . plot
            sns.scatterplot(data=df_plot, x='[$\degree$]', y=y_col, hue='Region', style='Group', 
                            palette=regasi, s=100, ax=ax, legend=(col_idx == 2))
            
            # . plot trends
            styles = ['-', '--', '-.', ':']
            x_vals = np.array([-7.5, 7.5])
            eqn_texts = []
            if not(row_idx):
                res = linregress(df_plot['[$\degree$]'], df_plot[y_col])
                ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                        linestyle=styles[0], alpha=0.6, lw=1.5)
                eqn_texts.append(f"$y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")
            else:
                for g_idx, (g_name, g_df) in enumerate(df_plot.groupby('Group')):
                    if len(g_df) > 1:
                        res = linregress(g_df['[$\degree$]'], g_df[y_col])
                        ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                                linestyle=styles[g_idx % 4], alpha=0.6, lw=1.5)
                        eqn_texts.append(f"{g_name}: $y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")

            # . format
            if col_idx == 0: ax.set_ylim(-1.5, 0.5)
            elif col_idx == 1: ax.set_ylim(-1.5, 1)
            elif col_idx == 2: ax.set_ylim(-0.5, 0.5)
            ax.set_xlim(-7.5, 7.5)
            ax.text(0.05, 0.95, "\n".join(eqn_texts), transform=ax.transAxes, fontsize=8, 
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if row_idx == 0: ax.set_title(f"Component: {comp.upper()}")
            if col_idx == 0: ax.set_ylabel(f"{var_name}\n[kPa]")

        # Handle Row Legends
        handles, labels = axes[row_idx, 2].get_legend_handles_labels()
        axes[row_idx, 2].get_legend().remove()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 1.0 - (row_idx+1)*0.22), ncol=1, fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig("_png/sup_sigOri.png", bbox_inches='tight')

# . scatter and trends for displacement and orientation
def _dispOri():

    # . setup
    # sns.set_style("whitegrid")
    comps = ["xy", "xz", "yz"]
    vars = ["Significance (p)", "Length", "Width", "Depth"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "sig"

    # . define groups
    sig_list = [0, 1, 5, 6, 8, 9, 12, 13, 17]
    x_bins = {tuple([0, 1, 2, 9, 10, 11]): "x >= 22", tuple([3, 4, 5, 12, 13, 14]): "11 >= x < 22", tuple([6, 7, 8, 15, 16, 17]): "x < 11"}
    y_bins = {tuple([0, 1, 2, 3, 4, 5, 6, 7, 8]): "y >= 9.9", tuple([9, 10, 11, 12, 13, 14, 15, 16, 17]): "y < 9.9"}
    z_bins = {tuple([0, 3, 6, 9, 12, 15]): "z >= 10", tuple([1, 4, 7, 10, 13, 16]): "5 >= z < 10", tuple([2, 5, 8, 11, 14, 17]): "z < 5"}

    # . load
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
    t_means = {c: df_test_r[f"{tensor}_{c}"].mean() for c in ["xx"] + comps}

    tensor = 'disp'
    comps = ["x", "y", "z"]

    # . loop tests
    region_stats = []
    for tt in range(18):
        
        # . extract stress data
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        df_r = df[
            (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
        
        # . load angles
        df_tile = pd.read_csv(f"_csv/tile_{tt}_w.csv")
        ang = df_tile["Sph_[DEG]"].dropna()
        mean_ang = ang[ang.abs() <= 30].mean()

        # . store data
        stat = {'Region': f'R{tt}', '[$\degree$]': mean_ang, 'index': tt}
        for c in comps:
            data = df_r[f"{tensor}_{c}"]
            # data = data[data.between(-20, 20)]
            stat[f'disp_{c}'] = data.mean()# - t_means[c]
        region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats)

    # . figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=1000, sharex=True)
    
    # . iterate plots rows
    for row_idx, var_name in enumerate(vars):

        # . type
        df_plot = _df.copy()
        if row_idx == 0: df_plot['Group'] = df_plot['index'].apply(lambda i: "p < 0.05" if i in sig_list else "p >= 0.05")
        elif row_idx == 1: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in x_bins.items() if i in k), "Other"))
        elif row_idx == 2: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in y_bins.items() if i in k), "Other"))
        elif row_idx == 3: df_plot['Group'] = df_plot['index'].apply(lambda i: next((v for k, v in z_bins.items() if i in k), "Other"))

        # . iterate plot columns
        for col_idx, comp in enumerate(comps):
            ax = axes[row_idx, col_idx]
            y_col = f'disp_{comp}'
            
            # . plot
            sns.scatterplot(data=df_plot, x='[$\degree$]', y=y_col, hue='Region', style='Group', 
                            palette=regasi, s=100, ax=ax, legend=(col_idx == 2))
            
            # . plot trends
            styles = ['-', '--', '-.', ':']
            x_vals = np.array([-7.5, 7.5])
            eqn_texts = []
            if not(row_idx):
                res = linregress(df_plot['[$\degree$]'], df_plot[y_col])
                ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                        linestyle=styles[0], alpha=0.6, lw=1.5)
                eqn_texts.append(f"$y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")
            else:
                for g_idx, (g_name, g_df) in enumerate(df_plot.groupby('Group')):
                    if len(g_df) > 1:
                        res = linregress(g_df['[$\degree$]'], g_df[y_col])
                        ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                                linestyle=styles[g_idx % 4], alpha=0.6, lw=1.5)
                        eqn_texts.append(f"{g_name}: $y={res.slope:.2f}x + {res.intercept:.2f}$ ($R={res.rvalue:.2f}$)")

            # . format
            ax.set_xlim(-7.5, 7.5)
            ax.text(0.05, 0.95, "\n".join(eqn_texts), transform=ax.transAxes, fontsize=8, 
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if row_idx == 0: ax.set_title(f"Component: {comp.upper()}")
            if col_idx == 0: ax.set_ylabel(f"{var_name}\n[nm]")

        # . legend
        handles, labels = axes[row_idx, 2].get_legend_handles_labels()
        axes[row_idx, 2].get_legend().remove()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, (row_idx+1)*0.22), ncol=1, fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig("_png/sup_dispOri.png", bbox_inches='tight')

def _so():

    # . setup
    comps = ["xy", "xz", "yz"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "sig"
    sig_list = []
    
    # . load and process data
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
    t_means = {c: df_test_r[f"{tensor}_{c}"].mean() for c in comps}

    region_stats = []
    for tt in range(18):
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        df_r = df[
            (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
        
        df_tile = pd.read_csv(f"_csv/tile_{tt}_w.csv")
        ang_data = df_tile["Sph_[DEG]"].dropna()
        ang_filtered = ang_data[ang_data.abs() <= 30]
        
        # Calculate Means AND Standard Deviations
        stat = {
            'Region': f'R{tt}', 
            '[$\degree$]': ang_filtered.mean(),
            'ang_std': ang_filtered.std() / np.sqrt(len(ang_filtered)), # X-axis variance
            'index': tt
        }
        
        for c in comps:
            data = df_r[f"{tensor}_{c}"]
            data = data[data.between(-20, 20)]
            stat[f'$\sigma$_{c}'] = data.mean() + t_means[c]
            stat[f'std_{c}'] = data.std() / np.sqrt(len(data)) # Y-axis variance
            
        region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats) 
    _df['Group'] = _df['index'].apply(lambda i: "p < 0.05" if i in sig_list else "p >= 0.05")

    # . figure setup
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), dpi=300)
    
    for col_idx, comp in enumerate(comps):
        ax = axes[col_idx]
        y_col = f'$\sigma$_{comp}'
        y_err = f'std_{comp}'
        
        # 1. Plot Error Bars first (so they are behind the markers)
        for i, row in _df.iterrows():
            ax.errorbar(
                row['[$\degree$]'], row[y_col],
                xerr=row['ang_std'], yerr=row[y_err],
                fmt='none',           # No marker here
                ecolor=regasi[row['Region']], 
                elinewidth=1.2, 
                capsize=1,            # Clean look, no horizontal caps
                alpha=0.75             # Transparent so it doesn't clutter
            )

        # 2. Plot Scatters (Larger markers)
        sns.scatterplot(
            data=_df, x='[$\degree$]', y=y_col, hue='Region', style='Group', 
            palette=regasi, s=90, ax=ax, legend=False, zorder=3
        )
        
        # 3. Trend line (High visibility)
        x_vals = np.array([-10, 10])
        res = linregress(_df['[$\degree$]'], _df[y_col])
        ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                linestyle='--', alpha=1.0, lw=1)
        
        # 4. Equation text (12pt)
        eqn_text = f"$y={res.slope:.2f}x + {res.intercept:.2f}$\n$R={res.rvalue:.2f}$"
        ax.text(0.05, 0.95, eqn_text, transform=ax.transAxes, fontsize=8, 
                fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # . format
        ax.set_xlim(-8, 8)
        if col_idx == 0: 
            ax.set_ylim(-1.5, 1.5)
            ax.set_yticks(np.arange(-1.5, 2, 0.5))
        elif col_idx == 1: 
            ax.set_ylim(-1.5, 1.5)
            ax.set_yticks(np.arange(-1.5, 2, 0.5))
        elif col_idx == 2: 
            ax.set_ylim(-0.3, 0.3)
            ax.set_yticks(np.arange(-0.3, 0.4, 0.1))

        # . strip text
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)

    # . adjust layout for maximum space between plots
    plt.tight_layout(pad=1.0, w_pad=1.0)
    plt.savefig("_png/PUB_SO.png", bbox_inches='tight')

def _soVar():

    # . setup
    comps = ["xy", "xz", "yz"]
    colors = sns.color_palette("tab20", 18)
    regasi = {f'R{i}': colors[i] for i in range(18)}
    disp, tensor = 20, "sig"
    sig_list = []
    
    # . load and process data
    df_test = pd.read_csv(f"_csv/sim_test_{disp}_200.csv")
    df_test_r = df_test[
            (df_test["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df_test["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df_test["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
    t_means = {c: df_test_r[f"{tensor}_{c}"].var() for c in comps}

    region_stats = []
    for tt in range(18):
        suffix = "300" if tt == 11 else "200"
        df = pd.read_csv(f"_csv/sim_{tt}_{disp}_{suffix}.csv")
        df_r = df[
            (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df["Z"].between(0.01 * EDGE[2], 0.95 * EDGE[2]))
        ]
        
        df_tile = pd.read_csv(f"_csv/tile_{tt}_w.csv")
        ang_data = df_tile["Sph_[DEG]"].dropna()
        ang_filtered = ang_data[ang_data.abs() <= 30]
        
        # Calculate Means AND Standard Deviations
        stat = {
            'Region': f'R{tt}', 
            '[$\degree$]': ang_filtered.var(),
            'ang_std': ang_filtered.std() / np.sqrt(len(ang_filtered)), # X-axis variance
            'index': tt
        }
        
        for c in comps:
            data = df_r[f"{tensor}_{c}"]
            data = data[data.between(-20, 20)]
            stat[f'$\sigma$_{c}'] = data.var()# + t_means[c]
            stat[f'std_{c}'] = data.std() / np.sqrt(len(data)) # Y-axis variance
            
        region_stats.append(stat)
    
    _df = pd.DataFrame(region_stats) 
    _df['Group'] = _df['index'].apply(lambda i: "p < 0.05" if i in sig_list else "p >= 0.05")

    # . figure setup
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), dpi=300)
    
    for col_idx, comp in enumerate(comps):
        ax = axes[col_idx]
        y_col = f'$\sigma$_{comp}'

        # 2. Plot Scatters (Larger markers)
        sns.scatterplot(
            data=_df, x='[$\degree$]', y=y_col, hue='Region', style='Group', 
            palette=regasi, s=90, ax=ax, legend=False, zorder=3
        )
        
        # 3. Trend line (High visibility)
        x_vals = np.array([-20, 70])
        res = linregress(_df['[$\degree$]'], _df[y_col])
        ax.plot(x_vals, res.slope * x_vals + res.intercept, color='black', 
                linestyle='--', alpha=1.0, lw=1)
        
        # 4. Equation text (12pt)
        eqn_text = f"$y={res.slope:.2f}x + {res.intercept:.2f}$\n$R={res.rvalue:.2f}$"
        ax.text(0.05, 0.95, eqn_text, transform=ax.transAxes, fontsize=8, 
                fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # . format
        ax.set_xlim(-5, 65)
        if col_idx == 0: 
            ax.set_ylim(5, 10.5)
            ax.set_yticks(np.arange(5, 11, 1))
        elif col_idx == 1: 
            ax.set_ylim(5, 10.5)
            ax.set_yticks(np.arange(5, 11, 1))
        elif col_idx == 2: 
            ax.set_ylim(0, 4)
            ax.set_yticks(np.arange(0, 5, 1))

        # . strip text
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)

    # . adjust layout for maximum space between plots
    plt.tight_layout(pad=1.0, w_pad=1.0)
    plt.savefig("_png/PUB_SOVAR.png", bbox_inches='tight')

# . Main
def main():

    # _so()
    # _soVar()
    # _sigOri()
    # _sigsOris()
    # _sigsOrisPrincipal()
    # _dispOri()
    # _dispsOris()
    # _sigsSepOris()
    _sigDisp()

# . Inititate
if __name__ == "__main__":
    main()