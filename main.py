"""
simulate responce of Turbie in terms of displacements
plot time history (one case) & std dev vs. w/s
assumptions & commentaries in README file
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import re
import turbie_mod 

# FOLDERS & CASE TO BE PLOTTED
OUT_ROOT = Path("outputs")
FOLDER_WIND = Path("inputs/wind_files")
CT_PATH = "inputs/turbie_inputs/CT.txt"
PARAM_PATH = "inputs/turbie_inputs/turbie_parameters.txt"

TI_LIST = [0.05, 0.10, 0.15]
TI_TO_PLOT = 0.10           # TI per il time-history da mostrare
CASE_INDEX_TO_PLOT = 2      # indice del file (0-based) in quella TI
KIND_CT = "linear"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Parameters Turbie 
rawP, P = turbie_mod.load_parameters_from_listfile(PARAM_PATH)
M, C, K = turbie_mod.build_matrices(P)

# Ct(U)
Ct_func = turbie_mod.build_ct_function(CT_PATH, kind=KIND_CT)

# wind speed files names
def _extract_u_from_name(p: Path) -> float:
    # atteso: wind_12_ms_TI_0.10.txt
    m = re.search(r'wind_(\d+)_ms', p.name)
    return float(m.group(1)) if m else float('inf')


# simulations loop over TI and wind speed cases
all_stats = {}  
time_history_fig = None 
time_history_info = None  

for TI in TI_LIST:
    print(f"\n=== TI={TI:.2f} ===")
    files = turbie_mod.list_wind_cases(FOLDER_WIND, TI)
    files = sorted(files, key=_extract_u_from_name)

    rows = []
    out_dir = OUT_ROOT / f"TI_{TI:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # index of case to be plotted
    if abs(TI - TI_TO_PLOT) < 1e-12:
        if len(files) == 0:
            print(f"Warning: no files TI={TI:.2f}, can't plot time-history.")
            valid_index_to_plot = None
        else:
            valid_index_to_plot = max(0, min(CASE_INDEX_TO_PLOT, len(files)-1))
    else:
        valid_index_to_plot = None

    for idx, fpath in enumerate(files):
        Ut = np.loadtxt(fpath, comments="#", skiprows=1, usecols=(0, 1))
        U_mean = float(np.mean(Ut[:, 1]))

        res = turbie_mod.simulate_turbie_case(
            M, C, K, Ut=Ut,
            Ct_func=Ct_func,
            rho=P.rho, D=P.D_rotor,
            y0=None, rtol=1e-4, atol=1e-7,
            ct_mode="mean"
        )

        # saving time history
        safe_um = int(round(U_mean))
        out_file = out_dir / f"response_U{safe_um}_TI_{TI:.2f}.txt"
        meta = dict(TI=TI, U_mean=res.get("U_mean", U_mean),
                    Ct_used=res.get("Ct_used", None),
                    D=P.D_rotor, rho=P.rho,
                    Ct_path=CT_PATH, param_path=PARAM_PATH,
                    wind_file=fpath.name, ct_mode=res["ct_mode"])
        turbie_mod.save_time_history(out_file, res, meta=meta)

        # stats
        st = turbie_mod.compute_stats(res)
        rows.append(dict(TI=TI, U_mean=U_mean, **st))

        # plotting time history
        if valid_index_to_plot is not None and idx == valid_index_to_plot:
            title = f"Wind & Displacements | U≈{U_mean:.1f} m/s | TI={TI:.2f}"
            time_history_fig = turbie_mod.plot_case_wind_and_displacements(res, title=title)

            # saving plot  in the upper level folder
            png_path = OUT_ROOT / f"time_history_U{safe_um}_TI_{TI:.2f}.png"
            time_history_fig.savefig(png_path, dpi=180, bbox_inches="tight")
            print(f"[PLOT] Time-history pronto e salvato in: {png_path}")

            # show the figure
            try:
                time_history_fig.canvas.draw()
                time_history_fig.canvas.flush_events()
            except Exception:
                pass

            print(f"[PLOT] Time-history visibile: file='{fpath.name}', idx={idx}, U_mean={U_mean:.2f}, TI={TI:.2f}")
            print(f"[PLOT] Salvato PNG: {png_path}")
            time_history_info = (TI, idx, fpath.name, U_mean)

    # saving stats sorted
    rows_sorted = sorted(rows, key=lambda r: r['U_mean'])
    stats_file = OUT_ROOT / f"stats_TI_{TI:.2f}.txt"
    turbie_mod.save_stats_per_TI(stats_file, rows_sorted)
    all_stats[TI] = rows_sorted

# plotting mean ± std dev 
fig, ax = plt.subplots(figsize=(10, 6))

colors = cm.viridis(np.linspace(0.2, 0.9, len(TI_LIST)))
markers = ['o', 's', '^', 'D', 'P']  # marker per DOF (blade/tower)

for i, TI in enumerate(TI_LIST):
    if TI not in all_stats:
        continue

    rows = all_stats[TI]
    U = np.array([r['U_mean'] for r in rows], dtype=float)
    mean_b = np.array([r['mean_blade'] for r in rows], dtype=float)
    std_b  = np.array([r['std_blade']  for r in rows], dtype=float)
    mean_t = np.array([r['mean_tower'] for r in rows], dtype=float)
    std_t  = np.array([r['std_tower']  for r in rows], dtype=float)

    col = colors[i]

    # blade collective
    ax.errorbar(
        U, mean_b, yerr=std_b,
        fmt='o-', color=col, capsize=4, elinewidth=1.5, capthick=1.2,
        markersize=6, markeredgecolor='white', markeredgewidth=1.0,
        label=f"Blade | TI={TI:.2f}"
    )

    # tower  =hub + nacelle + tower
    ax.errorbar(
        U, mean_t, yerr=std_t,
        fmt='s--', color=col, capsize=4, elinewidth=1.3, capthick=1.1,
        markersize=5, markerfacecolor='none', markeredgewidth=1.2,
        label=f"Tower | TI={TI:.2f}"
    )

ax.set_xlabel("U_mean [m/s]", fontsize=11)
ax.set_ylabel("Displacement [m]", fontsize=11)
ax.set_title("Mean ± Std dev of displacements vs U_mean (Blade & Tower, per TI)", fontsize=13)
ax.grid(True, alpha=0.4)
ax.legend(ncols=2, fontsize=9, frameon=False)
fig.tight_layout()

out_png = OUT_ROOT / "mean_std_whiskers_vs_U_mean_all_TI.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Plot mean ± std (whiskers) salvato: {out_png}")

plt.show()