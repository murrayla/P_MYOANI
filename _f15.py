"""
    Author: Murray, L, A.
    Contact: murrayla@student.unimelb.edu.au
             liam.a.murr@gmail.com
    ORCID: https://orcid.org/0009-0003-9276-6627
    File Name: _angHist.py
        Create publicaiton figure 11 plots
"""

# . imports
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind

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

# . comp all
def _stressBarMax(r0, r1):

    # . setup
    max_disp = 20 
    rs = [str(r0), f"{r0}#{r1}", str(r1)]
    comps = ["xy", "xz", "yz"]

    # . type of plotting
    plot_groups = [
        (rs[0], sns.color_palette("tab20")[r0], f"R{r0}"),
        (rs[1], "black", "Combined"),
        (rs[2], sns.color_palette("tab20")[r1], f"R{r1}"),
    ]

    # . iterate data
    bar_data = {g[2]: [] for g in plot_groups}
    for r, c, label in plot_groups:

        # . load data
        df = None
        try: 
            df = pd.read_csv(f"_csv/sim/sim_{r}_20.csv")
        except:
            try:
                df = pd.read_csv(f"_csv/sim/sim_{r}_20_200.csv")
            except:
                df = pd.read_csv(f"_csv/sim/sim_{r}_20_300.csv")

        # . select data
        df_r = df[
            (df["X"].between(0.05 * EDGE[0], 0.95 * EDGE[0])) &
            (df["Y"].between(0.05 * EDGE[1], 0.95 * EDGE[1])) &
            (df["Z"].between(0.05 * EDGE[2], 0.95 * EDGE[2]))
        ]

        # . take mean 
        for c in comps:
            val = df_r.loc[df_r[f"sig_{c}"].between(-50, 50), f"sig_{c}"].mean()
            bar_data[label].append(val)

    # . plotting
    mm_to_inch = 1 / 25.4
    fig, ax = plt.subplots(figsize=(180 * mm_to_inch, 120 * mm_to_inch), dpi=300)
    
    x = np.arange(len(comps))
    width = 0.1
    multiplier = 0

    # . plot bar data
    for label, values in bar_data.items():
        color = next(g[1] for g in plot_groups if g[2] == label)
        offset = width * multiplier
        ax.bar(
            x + offset, 
            values, width, 
            label=label, color=color, 
            edgecolor='white', lw=0.5)
        multiplier += 1

    # . formatting
    ax.set_ylabel('Shear Stress (MPa)', fontweight='bold')
    ax.set_title(f'Shear Components at {max_disp}mm', fontsize=11, fontweight='bold')
    
    # Re-centering the X-ticks based on the 7 bars per group
    ax.set_xticks(x + width) 
    ax.set_xticklabels([f"$\sigma_{{{c}}}$" for c in comps])
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    sns.despine()

    plt.tight_layout()
    plt.savefig("_png/PUB_BARCOMP.png")

def _stressAve(r0, r1):

    comps = ["xz", "yz"]
    raw_results = {r: [] for r in [str(r0), str(r1), f"{r0}#{r1}"]}
    
    for r in raw_results.keys():
        for path in [f"_csv/sim/sim_{r}_20.csv", f"_csv/sim/sim_{r}_20_200.csv", f"_csv/sim/sim_{r}_20_300.csv"]:
            try:
                df = pd.read_csv(path)
                df_r = df[(df["X"].between(0.05*EDGE[0], 0.95*EDGE[0])) & 
                          (df["Y"].between(0.05*EDGE[1], 0.95*EDGE[1])) & 
                          (df["Z"].between(0.05*EDGE[2], 0.95*EDGE[2]))]
                for c in comps:
                    val = df_r.loc[df_r[f"sig_{c}"].between(-50, 50), f"sig_{c}"].mean()
                    raw_results[r].append(val)
                break
            except: continue

    avg_vals = [(raw_results[str(r0)][i] + raw_results[str(r1)][i]) / 2 for i in range(len(comps))]
    colors = [sns.color_palette("tab20")[r0], sns.color_palette("tab20")[r1], "#d1d1d1", "black"]
    short_labels = [f"R{r0}", f"R{r1}", "Avg", "#"]

    plt.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(80 * mm, 80 * mm), dpi=300, sharex=False)
    
    for i, ax in enumerate(axes):
        # . Pull data for this specific component
        vals = [raw_results[str(r0)][i], raw_results[str(r1)][i], avg_vals[i], raw_results[f"{r0}#{r1}"][i]]
        x = np.arange(len(vals))
        
        bars = ax.bar(x, vals, color=colors, edgecolor='none', width=0.8)
        
        # . Formatting
        ax.set_title(f"$\sigma_{{{comps[i]}}}$", fontweight='bold', pad=10)
        ax.set_xticks([]) # Remove x-ticks for article cleanliness
        ax.grid(axis='y', lw=0.4, alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout(w_pad=2.0)
    plt.savefig("_png/PUB_BARAVE.png")

# def _stressComp(pairs, c):

#     # . setup
#     disps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
#     # . fig setup: 3x3 grid
#     mm_to_inch = 1 / 25.4
#     fig, axes  = plt.subplots(3, 3, figsize=(90 * mm_to_inch, 100 * mm_to_inch), dpi=300)
#     axes       = axes.flatten()
    
#     for pidx, (r0, r1) in enumerate(pairs):
#         ax = axes[pidx]
#         rs = [str(r0), str(r1), f"{r0}#{r1}"]
#         av = np.zeros((2, len(disps)))
        
#         try:
#             for idx, r in enumerate(rs):
#                 region_data = {c: []}
                
#                 for disp in disps:
#                     df   = None
#                     fpath = ""
                    
#                     if idx == 2:
#                         fpath = f"_csv/sim/sim_{r}_{disp}.csv"
#                     elif r == "11":
#                         fpath = f"_csv/sim/sim_{r}_{disp}_300.csv"
#                     else:
#                         fpath = f"_csv/sim/sim_{r}_{disp}_200.csv"
                    
#                     # . check file exists
#                     if not os.path.exists(fpath):
#                         raise FileNotFoundError(f"Missing: {fpath}")
                    
#                     df = pd.read_csv(fpath)

#                     # . select data
#                     df_r = df[
#                         (df["X"].between(0.10 * EDGE[0], 0.90 * EDGE[0])) &
#                         (df["Y"].between(0.10 * EDGE[1], 0.90 * EDGE[1])) &
#                         (df["Z"].between(0.10 * EDGE[2], 0.90 * EDGE[2]))
#                     ]

#                     col = f"sig_{c}"
#                     region_data[c].append(df_r.loc[df_r[col].between(-50, 50), col].mean())

#                 # . plot
#                 color = "black" if "#" in r else sns.color_palette("tab20")[int(rs[idx])]
                
#                 ax.plot(
#                     disps, region_data[c], 
#                     color=color, 
#                     lw=2.5,
#                     alpha=1.0 
#                 )

#                 if r == rs[0]:
#                     av[0, :] = region_data[c]
#                 if r == rs[1]:
#                     av[1, :] = region_data[c]

#             av = np.mean(av, axis=0)
#             ax.plot(disps, av, linestyle="--", color="gray", lw=2.5, alpha=1.0)

#         except (FileNotFoundError, Exception) as e:
#             # . leave plot empty but maintain formatting
#             print(f"Skipping pair ({r0}, {r1}): {e}")

#         # . formatting
#         ax.set_xlim(-1, 21)
#         ax.set_xticks(np.arange(0, 30, 10))
#         ax.set_ylim(-1.2, 1.2)
#         ax.set_yticks(np.arange(-1, 1.5, 0.5))
#         # ax.invert_yaxis()
#         ax.set_title("")
#         ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)
        
#         # . axis labels: only leftmost get y, only bottom get x
#         if pidx % 3 != 0:
#             ax.set_yticklabels([])
#         if pidx < 6:
#             ax.set_xticklabels([])

#     plt.tight_layout()
#     plt.savefig(f"_png/PUB_{c}_grid.png")

def _stressCompBar(pairs):

    # . setup
    disp  = 20
    c     = "xz"
    
    # . fig setup: 3x3 grid
    mm_to_inch = 1 / 25.4
    fig, axes  = plt.subplots(3, 3, figsize=(150 * mm_to_inch, 150 * mm_to_inch), dpi=300)
    axes       = axes.flatten()
    
    for pidx, (r0, r1) in enumerate(pairs):
        ax  = axes[pidx]
        rs  = [str(r0), str(r1), f"{r0}#{r1}"]
        vals = []
        lbls = []
        clrs = []
        
        try:
            for idx, r in enumerate(rs):
                fpath = ""
                
                if idx == 2:
                    fpath = f"_csv/sim/sim_{r}_{disp}.csv"
                elif r == "11":
                    fpath = f"_csv/sim/sim_{r}_{disp}_300.csv"
                else:
                    fpath = f"_csv/sim/sim_{r}_{disp}_200.csv"
                
                # . check file exists
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing: {fpath}")
                
                df = pd.read_csv(fpath)

                # . select data
                df_r = df[
                    (df["X"].between(0.10 * EDGE[0], 0.90 * EDGE[0])) &
                    (df["Y"].between(0.10 * EDGE[1], 0.90 * EDGE[1])) &
                    (df["Z"].between(0.10 * EDGE[2], 0.90 * EDGE[2]))
                ]

                col = f"sig_{c}"
                val = df_r.loc[df_r[col].between(-50, 50), col].mean()
                
                vals.append(val)
                lbls.append(r)
                clrs.append("black" if "#" in r else sns.color_palette("tab20")[int(r)])

            # . add average bar
            avg = np.mean([vals[0], vals[1]])
            vals.append(avg)
            lbls.append("avg")
            clrs.append("gray")

            # . plot bars
            ax.violin(range(len(vals)), vals, color=clrs, alpha=0.8, width=0.7)

        except (FileNotFoundError, Exception) as e:
            # . leave plot empty but maintain formatting
            print(f"Skipping pair ({r0}, {r1}): {e}")

        # . formatting
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks(np.arange(-5, 5, 1))
        ax.invert_yaxis()
        ax.set_title("")
        ax.set_xticks([])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, axis='y', zorder=0)
        
        # . axis labels: only leftmost get y
        if pidx % 3 != 0:
            ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(f"_png/PUB_XY_grid_bars.png")

def _stressCompRows(pairs, c):

    # . setup
    disp  = 20
    
    # . fig setup: 3x1 grid
    mm_to_inch = 1 / 25.4
    fig, axes  = plt.subplots(3, 1, figsize=(80 * mm_to_inch, 100 * mm_to_inch), dpi=300)
    
    # . collect all data
    r0_vals    = []
    r1_vals    = []
    combo_vals = []
    avg_vals   = [] # . predicted mean of R0 and R1
    r0_clrs    = []
    r1_clrs    = []
    
    for (r0, r1) in pairs:
        rs = [str(r0), str(r1), f"{r0}#{r1}"]
        
        try:
            row_vals = []
            for idx, r in enumerate(rs):
                fpath = ""
                
                if idx == 2:
                    fpath = f"_csv/sim/sim_{r}_{disp}.csv"
                elif r == "11":
                    fpath = f"_csv/sim/sim_{r}_{disp}_300.csv"
                else:
                    fpath = f"_csv/sim/sim_{r}_{disp}_200.csv"
                
                # . check file exists
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing: {fpath}")
                
                df = pd.read_csv(fpath)

                # . select data
                df_r = df[
                    (df["X"].between(0.10 * EDGE[0], 0.90 * EDGE[0])) &
                    (df["Y"].between(0.10 * EDGE[1], 0.90 * EDGE[1])) &
                    (df["Z"].between(0.10 * EDGE[2], 0.90 * EDGE[2]))
                ]

                col = f"sig_{c}"
                val = df_r.loc[df_r[col].between(-50, 50), col].mean()
                row_vals.append(val)
            
            # . collect results
            r0_v, r1_v, c_v = row_vals[0], row_vals[1], row_vals[2]
            
            r0_vals.append(r0_v)
            r1_vals.append(r1_v)
            combo_vals.append(c_v)
            avg_vals.append((r0_v + r1_v) / 2) # . calculate prediction
            
            r0_clrs.append(sns.color_palette("tab20")[int(rs[0])])
            r1_clrs.append(sns.color_palette("tab20")[int(rs[1])])
            
        except (FileNotFoundError, Exception) as e:
            print(f"Skipping pair ({r0}, {r1}): {e}")
            r0_vals.append(0); r1_vals.append(0); combo_vals.append(0); avg_vals.append(0)
            r0_clrs.append("white"); r1_clrs.append("white")
    
    x_pos = range(len(r1_vals))
    
    # . plot R1s (top row)
    ax = axes[0]
    ax.bar(x_pos, r1_vals, color=r1_clrs, alpha=0.8, width=0.7)
    # ax.set_ylim(0, 1.2)
    # ax.set_yticks(np.arange(0, 1.5, 0.5))
    ax.set_ylim(-.25, .25)
    ax.set_yticks(np.arange(-0.2, 0.3, 0.1))
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_xticklabels([])
    sns.despine(ax=ax, bottom=False, top=True, right=False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6, axis='y', zorder=0)
    
    # . plot combos + overlay (middle row)
    ax = axes[1]
    # . actual simulation result (wide bar)
    ax.bar(x_pos, combo_vals, color="#878787", alpha=0.6, width=0.7)
    # . calculated mean overlay (thinner "target" bar)
    ax.bar(x_pos, avg_vals, color='none', edgecolor='black', linewidth=0.8, 
           linestyle='--', alpha=0.75, width=0.7)
    
    # ax.set_ylim(-1.2, 1.2)
    # ax.set_yticks(np.arange(-1, 2, 1))
    ax.set_ylim(-.25, .25)
    ax.set_yticks(np.arange(-0.2, 0.3, 0.1))
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_xticklabels([])
    sns.despine(ax=ax, bottom=True, right=False)
    ax.axhline(0, color="k", linewidth=0.8, clip_on=False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6, axis='y', zorder=0)
    
    # . plot R0s (bottom row)
    ax = axes[2]
    ax.bar(x_pos, r0_vals, color=r0_clrs, alpha=0.8, width=0.7)
    # ax.set_ylim(-1.2, 0)
    # ax.set_yticks(np.arange(-1.0, 0.5, 0.5))
    ax.set_ylim(-.25, .25)
    ax.set_yticks(np.arange(-0.2, 0.3, 0.1))
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_xticklabels([])
    sns.despine(ax=ax, right=False, bottom=True, top=False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6, axis='y', zorder=0)

    plt.tight_layout(h_pad=3)
    plt.savefig(f"_png/PUB_{c}_BAR.png")

def _stressCompTrend(pairs, c):

    # . setup
    disps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # . fig setup: 3x1 grid
    mm_to_inch = 1 / 25.4
    fig, axes  = plt.subplots(3, 1, figsize=(50 * mm_to_inch, 100 * mm_to_inch), dpi=300)
    
    # . collect all data
    r0_trends = []
    r1_trends = []
    combo_trends = []
    avg_trends = []
    r0_clrs = []
    r1_clrs = []
    
    for (r0, r1) in pairs:
        rs = [str(r0), str(r1), f"{r0}#{r1}"]
        av = np.zeros((2, len(disps)))
        
        try:
            pair_data = []
            for idx, r in enumerate(rs):
                region_data = {c: []}
                
                for disp in disps:
                    df   = None
                    fpath = ""
                    
                    if idx == 2:
                        fpath = f"_csv/sim/sim_{r}_{disp}.csv"
                    elif r == "11":
                        fpath = f"_csv/sim/sim_{r}_{disp}_300.csv"
                    else:
                        fpath = f"_csv/sim/sim_{r}_{disp}_200.csv"
                    
                    # . check file exists
                    if not os.path.exists(fpath):
                        raise FileNotFoundError(f"Missing: {fpath}")
                    
                    df = pd.read_csv(fpath)

                    # . select data
                    df_r = df[
                        (df["X"].between(0.10 * EDGE[0], 0.90 * EDGE[0])) &
                        (df["Y"].between(0.10 * EDGE[1], 0.90 * EDGE[1])) &
                        (df["Z"].between(0.10 * EDGE[2], 0.90 * EDGE[2]))
                    ]

                    col = f"sig_{c}"
                    region_data[c].append(df_r.loc[df_r[col].between(-50, 50), col].mean())

                pair_data.append(region_data[c])
                
                if r == rs[0]:
                    av[0, :] = region_data[c]
                if r == rs[1]:
                    av[1, :] = region_data[c]

            av_vals = np.mean(av, axis=0)
            
            r0_trends.append(pair_data[0])
            r1_trends.append(pair_data[1])
            combo_trends.append(pair_data[2])
            avg_trends.append(av_vals)
            r0_clrs.append(sns.color_palette("tab20")[int(rs[0])])
            r1_clrs.append(sns.color_palette("tab20")[int(rs[1])])
            
        except (FileNotFoundError, Exception) as e:
            print(f"Skipping pair ({r0}, {r1}): {e}")
    
    # . plot R1s (top row)
    ax = axes[0]
    for trend, clr in zip(r1_trends, r1_clrs):
        ax.plot(disps, trend, color=clr, lw=2.5, alpha=1.0)
    ax.set_xlim(-1, 21)
    ax.set_xticks(np.arange(0, 30, 10))
    # ax.set_ylim(-1.2, 1.2)
    # ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_ylim(-.25, .25)
    ax.set_yticks(np.arange(-0.2, 0.3, 0.1))
    ax.set_xticklabels([])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)
    
    # . plot combos and averages (middle row)
    ax = axes[1]
    for jj, combo in enumerate(combo_trends):
        ax.plot(disps, combo, color=sns.color_palette("Grays", 9)[jj], lw=2.5, alpha=1.0)
    # for avg in avg_trends:
    #     ax.plot(disps, avg, linestyle="--", color="gray", lw=1, alpha=1.0)
    ax.set_xlim(-1, 21)
    ax.set_xticks(np.arange(0, 30, 10))
    # ax.set_ylim(-1.2, 1.2)
    # ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_ylim(-.25, .25)
    ax.set_yticks(np.arange(-0.2, 0.3, 0.1))
    ax.set_xticklabels([])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)
    
    # . plot R0s (bottom row)
    ax = axes[2]
    for trend, clr in zip(r0_trends, r0_clrs):
        ax.plot(disps, trend, color=clr, lw=2.5, alpha=1.0)
    ax.set_xlim(-1, 21)
    ax.set_xticks(np.arange(0, 30, 10))
    # ax.set_ylim(-1.2, 1.2)
    # ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_ylim(-.25, .25)
    ax.set_yticks(np.arange(-0.2, 0.3, 0.1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1, zorder=0)

    plt.tight_layout()
    plt.savefig(f"_png/PUB_{c}_TREND.png")

def _orientComp(pairs):
    # . fig setup: 3x1 grid, 50mm width
    mm_to_inch = 1 / 25.4
    fig, axes = plt.subplots(3, 1, figsize=(60 * mm_to_inch, 100 * mm_to_inch), dpi=300)
    
    # . data containers
    r0_sph, r1_sph, combo_sph = [], [], []
    r0_azi, r1_azi, combo_azi = [], [], []
    r0_ele, r1_ele, combo_ele = [], [], []
    
    y_mid = EDGE[1] / 2

    def _process_tile(tile_id):
        df = pd.read_csv(f"_csv/tile_{tile_id}_w.csv")
        y_vals = df["Centroid"].map(lambda x: ast.literal_eval(x)[1])
        def get_valid(col):
            mask = df[col].abs() <= 35
            return df[col][mask], y_vals[mask]
        return get_valid("Sph_[DEG]"), get_valid("Azi_[DEG]"), get_valid("Ele_[DEG]")

    for (r0, r1) in pairs:
        try:
            data0, data1 = _process_tile(r0), _process_tile(r1)
            lists = [(r0_sph, combo_sph, r1_sph), 
                     (r0_azi, combo_azi, r1_azi), 
                     (r0_ele, combo_ele, r1_ele)]
            for i, (r0_l, c_l, r1_l) in enumerate(lists):
                (v0, y0), (v1, y1) = data0[i], data1[i]
                r0_l.extend(v0.values); r1_l.extend(v1.values)
                c_l.extend(v1[y1 < y_mid].values); c_l.extend(v0[y0 >= y_mid].values)
        except Exception as e: print(f"Error: {e}")
    
    clrs = ["#BEA8FF", "#D3D3D3", "#FBBB60"] 
    data_map = [[r0_sph, combo_sph, r1_sph],
                [r0_azi, combo_azi, r1_azi],
                [r0_ele, combo_ele, r1_ele]]
    
    for row, ax in enumerate(axes):
        row_data = data_map[row]
        
        # . 1. plot boxplots
        for col in range(3):
            bp = ax.boxplot([row_data[col]], positions=[col+1], widths=0.5, 
                            patch_artist=True, showfliers=False, vert=False,
                            whiskerprops=dict(color="black", linewidth=0.7),
                            capprops=dict(color="black", linewidth=0.7),
                            medianprops=dict(color="black", linestyle="-", linewidth=1.1))
            bp['boxes'][0].set(facecolor=clrs[col], alpha=0.9, linewidth=0.6)

        # . 2. pairwise significance brackets
        # . x_start is where the brackets begin (just past the data range)
        x_start = 15
        bracket_step = 5 # . how far apart each bracket sits
        significant_count = 0
        
        for i, j in combinations(range(3), 2):
            if len(row_data[i]) > 1 and len(row_data[j]) > 1:
                _, p = ttest_ind(row_data[i], row_data[j], equal_var=False)
                
                if p < 0.05:
                    significant_count += 1
                    # . determine the x-position for this specific bracket
                    x_pos = x_start + (significant_count * bracket_step)
                    y1, y2 = i + 1, j + 1
                    
                    # . draw the bracket (vertical line with small horizontal tips)
                    ax.plot([x_pos, x_pos+1, x_pos+1, x_pos], [y1, y1, y2, y2], 
                            color="black", linewidth=0.6)
                    
                    # . add the star
                    ax.text(x_pos + 2, (y1 + y2) / 2, "*", 
                            ha='left', va='center', fontsize=8, fontweight='bold')

        # . 3. styling
        ax.set_yticks([])
        ax.set_xlim(-31, 31) # . extended for brackets
        ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
        ax.tick_params(axis='both', labelsize=10, length=2)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.5, alpha=0.3)
        
        for spine in ['right', 'top']: ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        
        if row < 2: ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig("_png/PUB_ORIENT.png")

# . Main
def main():

    # . stress curve
    rs = [
        [0, 9],
        [3, 12],
        [6, 15],
        [1, 10],
        [4, 13],
        [7, 16],
        [2, 11],
        [5, 14],
        [8, 17],
    ]
    
    _stressCompTrend(rs, "yz")
    _stressCompRows(rs, "yz")
    # _orientComp(rs)

# . Inititate
if __name__ == "__main__":
    main()