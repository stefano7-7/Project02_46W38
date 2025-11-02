# turbie_mod.py — CLEAN & FAST
"""
DATACLASSES & PARAMATERS LOADING
SYSTEM DYNAMIC EQUATION SOLVED FOR A TIME-MARCHING RESPONSE
SAVING DATA & PLOTTING FUNCTIONS
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# =========DATACLASSES & PARAMATERS LOADING===========
@dataclass
class RawListParams:
    mb: float; mn: float; mh: float; mt: float
    c1: float; c2: float
    k1: float; k2: float
    fb: float; ft: float
    drb: float; drt: float
    Dr: float; rho: float

@dataclass
class TurbieParams:
    m1: float; m2: float
    c1: float; c2: float
    k1: float; k2: float
    D_rotor: float; rho: float

def load_parameters_from_listfile(path: str) -> Tuple[RawListParams, TurbieParams]:
    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.append(float(s.split("#", 1)[0].split()[0]))

    mb, mn, mh, mt, c1, c2, k1, k2, fb, ft, drb, drt, Dr, rho = vals[:14]
    raw = RawListParams(mb, mn, mh, mt, c1, c2, k1, k2, fb, ft, drb, drt, Dr, rho)
    params = TurbieParams(
        m1=3.0*mb, m2=mh+mn+mt,
        c1=c1, c2=c2, k1=k1, k2=k2,
        D_rotor=Dr, rho=rho
    )
    return raw, params

def build_matrices(p: TurbieParams):
    """
    assemblying matrices
    """
    M = np.array([[p.m1, 0.0],[0.0, p.m2]], dtype=float)
    C = np.array([[ p.c1,    -p.c1],
                  [-p.c1, p.c1+p.c2]], dtype=float)
    K = np.array([[ p.k1,    -p.k1],
                  [-p.k1, p.k1+p.k2]], dtype=float)
    return M, C, K

def read_wind_file(base_folder: Path, wind_speed: int | float, TI: float):
    """"
    reads input files of wind speed time series
    """
    base_folder = Path(base_folder)
    fileNameWS = f"wind_{int(wind_speed)}_ms_TI_{TI:.1f}.txt"
    found = list(base_folder.rglob(fileNameWS))
    if not found:
        raise FileNotFoundError(f"Non trovato: {fileNameWS} in {base_folder}")
    data = np.loadtxt(found[0], comments="#", skiprows=1)
    return data  # t[s], U[m/s]

def list_wind_cases(base_folder: Path, TI: float) -> List[Path]:
    base_folder = Path(base_folder)
    files = sorted(base_folder.rglob(f"wind_*_ms_TI_{TI:.1f}.txt"))
    if not files:
        raise FileNotFoundError(f"no file wind_*_ms_TI_{TI:.1f}.txt in {base_folder}")
    return files

# ==========SYSTEM DYNAMIC EQUATION SOLVED FOR A TIME-MARCHING RESPONSE===============
def build_ct_function(ct_path: str, kind: str = "linear"):
    """
    reads look-up table Ct vs. w/s
    returns interpolating function
    """
    data = np.loadtxt(ct_path, comments="#", skiprows=1)
    U_tab, Ct_tab = data[:, 0], data[:, 1]
    Ct_func = interp1d(
        U_tab, Ct_tab,
        kind=kind, bounds_error=False,
        fill_value=(Ct_tab[0], Ct_tab[-1])
    )
    return Ct_func

def make_thrust_function_fast(Ut: np.ndarray, rho: float, D: float, Ct_func):
    """
    calculate forcing by wind loads
    """
    t = Ut[:, 0]; U = Ut[:, 1]
    Arot = 0.25*np.pi*D**2
    T = 0.5 * rho * Arot * Ct_func(U) * U**2
    T_of_t = interp1d(t, T, kind='linear', bounds_error=False,
                      fill_value=(T[0], T[-1]))
    return T_of_t, (t[0], t[-1])

def make_rhs(M: np.ndarray, C: np.ndarray, K: np.ndarray, T_of_t):
    """
    y' = A y + B(t)
    """
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    Z = np.zeros_like(M); I = np.eye(n)
    A = np.vstack([np.hstack([Z, I]),
                   np.hstack([-Minv @ K, -Minv @ C])])

    B = np.zeros(2*n)
    force_dir = np.array([1.0, 0.0])   # forza solo sul DOF1

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        T = float(T_of_t(t))
        # B = [0; Minv @ (T * e1)]
        B[:n] = 0.0
        B[n:] = Minv @ (T * force_dir)
        return A @ y + B

    return rhs

def simulate_turbie_case(
    M: np.ndarray, C: np.ndarray, K: np.ndarray,
    Ut: np.ndarray,
    Ct_func=None, Ct_path: Optional[str]=None, kind: str='linear',
    rho: float=1.225, D: float=90.0,
    y0: Optional[np.ndarray]=None,
    rtol: float=1e-4, atol: float=1e-7,
    ct_mode: str = "mean"   # "mean" (Ct costante per file) oppure "instant"
):
    """
    integrates over time for each (wind speed - TI) input
    has the options
    - ct_mode="mean": Ct è valutato a partire da U_mean del file (Ct costante).
    - ct_mode="instant": Ct è valutato istantaneamente Ct(U(t)).

    Performance patches:
      - nessun max_step (lasciare adattività al solver)
      - np.interp per T(t)
      - pre-calcolo Minv @ e1 e riuso di buffer B
    """
    # --- Ct(U) ---
    if Ct_func is None:
        if Ct_path is None:
            raise ValueError("Provide Ct_func or Ct_path")
        Ct_func = build_ct_function(Ct_path, kind=kind)

    # --- wind data ---
    t_eval = np.asarray(Ut[:, 0], dtype=float)
    U_vec  = np.asarray(Ut[:, 1], dtype=float)
    t0, tf = float(t_eval[0]), float(t_eval[-1])

    # --- choice for Ct ---
    if ct_mode not in ("mean", "instant"):
        raise ValueError("ct_mode must be 'mean' or 'instant'")

    if ct_mode == "mean":
        U_mean = float(np.mean(U_vec))
        Ct_used = float(Ct_func(U_mean))   # Ct costant
        Ct_vec = None                     
    else:
        Ct_used = None
        Ct_vec = np.asarray(Ct_func(U_vec))  # Ct(t)

    # --- forcing sampled as T_samp ---
    Arot = 0.25 * np.pi * D**2
    if ct_mode == "mean":
        T_samp = 0.5 * rho * Arot * Ct_used * U_vec**2
    else:
        T_samp = 0.5 * rho * Arot * Ct_vec * U_vec**2

    # --- initial state y0 ---
    n = M.shape[0]
    if y0 is None:
        y0 = np.zeros(2*n, dtype=float)

    # --- dynamic equation ---
    Minv = np.linalg.inv(M)
    Z = np.zeros_like(M)
    I = np.eye(n)
    A = np.vstack([
        np.hstack([Z, I]),
        np.hstack([-Minv @ K, -Minv @ C])
    ])

    e1 = np.zeros(n)
    e1[0] = 1.0
    Minv_e1 = Minv @ e1
    B = np.zeros(2*n, dtype=float)

    # --- np.interp for forcing T(t) ---
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        T = np.interp(t, t_eval, T_samp)
        B[:n] = 0.0
        B[n:] = T * Minv_e1
        return A @ y + B

    # --- integration ---
    sol = solve_ivp(rhs, (t0, tf), y0,
                    method='RK45', t_eval=t_eval,
                    rtol=rtol, atol=atol)

    if not sol.success:
        raise RuntimeError(f"solve_ivp non riuscito: {sol.message}")

    out = dict(
        t = sol.t,
        x1 = sol.y[0, :],
        x2 = sol.y[1, :],
        v1 = sol.y[2, :],
        v2 = sol.y[3, :],
        U  = U_vec
    )
    # info  for savingsalvataggi
    if ct_mode == "mean":
        out["U_mean"] = U_mean
        out["Ct_used"] = Ct_used
    else:
        out["Ct_series"] = Ct_vec
    out["ct_mode"] = ct_mode
    return out

# ==========SAVING DATA & PLOTTING FUNCTIONS===============
def save_time_history(out_path: Path, result: dict, meta: Optional[dict] = None):
    """
    time histories saved in output folder 
    """
    out_path = Path(out_path)

    t  = np.asarray(result.get('t', []), dtype=float)
    U  = np.asarray(result.get('U', []), dtype=float)
    x1 = np.asarray(result.get('x1', []), dtype=float)
    x2 = np.asarray(result.get('x2', []), dtype=float)
    v1 = np.asarray(result.get('v1', []), dtype=float)
    v2 = np.asarray(result.get('v2', []), dtype=float)

    # Sampling rate
    if len(t) > 1:
        dt = np.median(np.diff(t))
        fs = 1.0 / dt if dt > 0 else np.nan
    else:
        fs = np.nan

    # Header
    header_lines = []
    if meta:
        parts = []
        for k, v in meta.items():
            try:
                v_float = float(v)
                parts.append(f"{k}={v_float:.2f}")
            except Exception:
                parts.append(f"{k}={v}")  # mantiene stringhe o altri tipi
        header_lines.append(" | ".join(parts))

    header_lines.append(f"Sampling_rate_Hz={fs:.2f}")
    header_lines.append("t[s]\tU[m/s]\tx_blade[m]\tx_tower[m]\tv_blade[m/s]\tv_tower[m/s]")
    header = "\n".join(header_lines)

    # Format 
    arr = np.column_stack([t, U, x1, x2, v1, v2])
    np.savetxt(out_path, arr, header=header, comments='', fmt="%.2f")

    print(f"Saved '{out_path.name} ' (fs = {fs:.2f} Hz)")


def compute_stats(result: Dict):
    """
    calculate mean & std dev of responce displacements
    """
    x1 = np.asarray(result['x1']); x2 = np.asarray(result['x2'])
    return {
        'mean_blade': float(np.mean(x1)),
        'std_blade' : float(np.std(x1, ddof=0)),
        'mean_tower': float(np.mean(x2)),
        'std_tower' : float(np.std(x2, ddof=0)),
    }

def save_stats_per_TI(out_path: Path, rows: List[Dict]):
    """
    saves the statistics
    """
    out_path = Path(out_path)
    header = "TI\tU_mean[m/s]\tmean_blade[m]\tstd_blade[m]\tmean_tower[m]\tstd_tower[m]"
    arr = np.array([[r['TI'], r['U_mean'], r['mean_blade'], r['std_blade'],
                     r['mean_tower'], r['std_tower']] for r in rows], dtype=float)
    np.savetxt(out_path, arr, header=header, comments='', fmt="%.4e")



def plot_stats_vs_wind(TI: float, rows: List[Dict]):
    """
    Plot the means and standard deviations of the blade and tower displacements 
    for the wind speeds of each TI category
    """
    # --- Sort data by mean wind speed ---
    rows = sorted(rows, key=lambda r: r['U_mean'])
    U = np.array([r['U_mean'] for r in rows])
    mean_b = np.array([r['mean_blade'] for r in rows])
    mean_t = np.array([r['mean_tower'] for r in rows])
    std_b  = np.array([r['std_blade']  for r in rows])
    std_t  = np.array([r['std_tower']  for r in rows])

    # === FIGURE 1: Means vs wind speed ===
    fig1, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.plot(U, mean_b, 'o-', color='tab:blue', label='Blade')
    ax1.plot(U, mean_t, 's--', color='tab:orange', label='Tower')
    ax1.set_xlabel("Mean wind speed [m/s]")
    ax1.set_ylabel("Mean displacement [m]")
    ax1.set_title(f"Mean displacements vs wind | TI = {TI:.2f}")
    ax1.grid(True, alpha=0.4)
    ax1.legend()
    fig1.tight_layout()

    # === FIGURE 2: Standard deviations vs wind speed ===
    fig2, ax2 = plt.subplots(figsize=(7.5, 4))
    ax2.plot(U, std_b, 'o-', color='tab:blue', label='Blade')
    ax2.plot(U, std_t, 's--', color='tab:orange', label='Tower')
    ax2.set_xlabel("Mean wind speed [m/s]")
    ax2.set_ylabel("Standard deviation [m]")
    ax2.set_title(f"Standard deviations vs wind | TI = {TI:.2f}")
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    fig2.tight_layout()

    return fig1, fig2


def plot_case_wind_and_displacements(result: dict, title: str = ""):
    """
    plot w/S & displacements over time 
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    t = result['t']; U = result['U']
    x1 = result['x1']; x2 = result['x2']

    # sampling rate 
    dt = np.median(np.diff(t))
    fs = 1.0 / dt if dt > 0 else np.nan

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                         gridspec_kw=dict(hspace=0.08))

    # wind speed
    ws_color = 'tab:blue'
    ax_top.plot(t, U, color=ws_color, linewidth=1.8, alpha=0.85, label="U(t) [m/s]")
    ax_top.set_ylabel("Wind speed [m/s]", color=ws_color)
    ax_top.tick_params(axis='y', colors=ws_color)
    ax_top.spines['right'].set_color(ws_color)
    ax_top.grid(True, alpha=0.35)
    ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # BOTTOM: displacements
    ax_bot.plot(t, x1, color='tab:orange', linewidth=1.8, label="x_blade [m]")
    ax_bot.plot(t, x2, color='tab:green', linewidth=1.6, linestyle='--', label="x_tower [m]")
    ax_bot.set_xlabel("Time [s]")
    ax_bot.set_ylabel("Displacement [m]")
    ax_bot.grid(True, alpha=0.35)
    ax_bot.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_bot.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax_bot.legend(loc='upper right', fontsize=9)

    full_title = (f"{title} | Sampling rate ≈ {fs:.2f} Hz") if title else (f"Sampling rate ≈ {fs:.2f} Hz")
    fig.suptitle(full_title, y=0.995, fontsize=12.5)

    fig.tight_layout()
    return fig

