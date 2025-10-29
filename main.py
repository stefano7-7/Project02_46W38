import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import turbie_mod

# inputs
folderName = Path(r"C:\Users\stefano\Project02_46W38\inputs\wind_files")
Ct_path = os.path.join("inputs", "turbie_inputs", "CT.txt")
param_path = os.path.join("inputs", "turbie_inputs", "turbie_parameters.txt")
TI_list = [0.05, 0.10, 0.15]

# output folders
out_root = Path("outputs")
out_root.mkdir(exist_ok=True, parents=True)

# build mass, stiffness and damping matrices
raw, P = turbie_mod.load_parameters_from_listfile(param_path)
M, C, K = turbie_mod.build_matrices(P)
turbie_mod.analyze_and_print_system(M, C, K, raw=raw)

# loop over TI and w/s files
all_stats = {}  # TI -> list of rows
for TI in TI_list:
    print(f"\n=== TI={TI:.2f} ===")
    files = turbie_mod.list_wind_cases(folderName, TI)
    rows = []

    for fpath in files:
        name = fpath.name
        try:
            u_nom = float(name.split("_")[1])  # "wind_6_ms_TI_0.1.txt" -> "6"
        except Exception:
            u_nom = np.nan

        Ut = np.loadtxt(fpath, comments="#", skiprows=1)   # t, U(t)
        U_mean = float(np.mean(Ut[:, 1]))

        # simulation
        res = turbie_mod.simulate_turbie_case(M, C, K,
                                              Ut=Ut,
                                              Ct_path=Ct_path, kind='linear',
                                              rho=P.rho, D=P.D_rotor,
                                              y0=None)

        # time-history saved
        out_dir = out_root / f"TI_{TI:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"response_U{int(round(U_mean))}_TI_{TI:.2f}.txt"
        meta = dict(TI=TI, U_mean=U_mean, D=P.D_rotor, rho=P.rho)
        turbie_mod.save_time_history(out_file, res, meta=meta)

        # statistics
        st = turbie_mod.compute_stats(res)
        row = dict(TI=TI, U_mean=U_mean, **st)
        rows.append(row)

        # Plot of one single case
        if TI == 0.10 and files.index(fpath) == 0:
            fig = turbie_mod.plot_case_wind_and_displacements(
                res, title=f"Wind & Displacements | U≈{U_mean:.1f} m/s | TI={TI:.2f}"
            )
            fig.savefig(out_dir / f"case_plot_U{int(round(U_mean))}_TI_{TI:.2f}.png",
                        dpi=180, bbox_inches="tight")
            plt.close(fig)

    # save statistics by TI
    turbie_mod.save_stats_per_TI(out_root / f"stats_TI_{TI:.2f}.txt", rows)
    all_stats[TI] = rows

    # plot mean/std vs U per TI
    fig1, fig2 = turbie_mod.plot_stats_vs_wind(TI, rows)
    fig1.savefig(out_root / f"means_vs_wind_TI_{TI:.2f}.png", dpi=180, bbox_inches="tight")
    fig2.savefig(out_root / f"stds_vs_wind_TI_{TI:.2f}.png",   dpi=180, bbox_inches="tight")
    plt.close(fig1); plt.close(fig2)

print("\n✅ Simulations completed and results saved in:", out_root.resolve())


