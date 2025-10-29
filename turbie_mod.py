import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from dataclasses import dataclass
from pathlib import Path
from scipy.integrate import solve_ivp

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def get_Ct(path, ws, kind='linear', plot=True, show_arrows=True, ax=None):
    """
    reads the look-up table, 
    creates interpolating func calculating the thrust coeff for a given w/s
    plot
    """
    # Caricamento robusto (salta 1 riga di header se presente)
    data = np.loadtxt(path, comments="#", skiprows=1)
    U_tab, Ct_tab = data[:, 0], data[:, 1]

    # Interpolante con extrapolation clamp (estende i bordi)
    get_Ct = interp1d(
        U_tab, Ct_tab,
        kind=kind,
        bounds_error=False,
        fill_value=(Ct_tab[0], Ct_tab[-1])
    )

    Ct_at_U = float(get_Ct(ws))

    if plot:
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
            created_fig = True

        # Curva “lisciata” per il plot
        U_fine = np.linspace(U_tab.min(), U_tab.max(), 300)
        Ct_fine = get_Ct(U_fine)

        ax.plot(U_tab, Ct_tab, 'o', label='Look-up table')
        ax.plot(U_fine, Ct_fine, '-', linewidth=2, label=f'Interp: {kind}')
        ax.plot([ws], [Ct_at_U], 's', markersize=7, label=rf'calculated: $U={ws:.2f}$ m/s')

        ax.set_xlabel("wind speed [m/s]")
        ax.set_ylabel(r"thrust coeff $C_T$ [-]")
        ax.set_title(r"$C_T(U)$ and current point")
        ax.grid(True, alpha=0.5)

        if show_arrows:
            # vertical arrow
            y_min, y_max = ax.get_ylim()
            ax.annotate(
                '', xy=(ws, Ct_at_U), xytext=(ws, y_min),
                arrowprops=dict(arrowstyle='->', lw=1.5)
            )
            ax.text(ws, y_min, rf"$U={ws:.2f}$ m/s",
                    ha='center', va='top', rotation=90, fontsize=9, color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.6))

            # horizontal arrow
            x_min, x_max = ax.get_xlim()
            ax.annotate(
                '', xy=(ws, Ct_at_U), xytext=(x_min, Ct_at_U),
                arrowprops=dict(arrowstyle='->', lw=1.5)
            )
            ax.text(x_min, Ct_at_U, rf"$C_T={Ct_at_U:.3f}$",
                    ha='left', va='bottom', fontsize=9, color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.6))

        ax.legend(loc='best')
        plt.tight_layout()
        if created_fig:
            plt.show()

    return Ct_at_U

# building matrices
# ---------- Dataclasses ----------
@dataclass
class RawListParams:
    mb: float     # blade (singola pala) [kg]
    mn: float     # nacelle [kg]
    mh: float     # hub [kg]
    mt: float     # tower [kg]
    c1: float     # N/(m/s)
    c2: float     # N/(m/s)
    k1: float     # N/m
    k2: float     # N/m
    fb: float     # Hz (target mode, ad es. "blade-like")
    ft: float     # Hz (target mode, ad es. "tower-like")
    drb: float    # damping ratio blade-like [-]
    drt: float    # damping ratio tower-like [-]
    Dr: float     # rotor diameter [m]
    rho: float    # kg/m^3

@dataclass
class TurbieParams:
    m1: float
    m2: float
    c1: float
    c2: float
    k1: float
    k2: float
    D_rotor: float
    rho: float

# ---------- Parser ----------
def load_parameters_from_listfile(path: str) -> tuple[RawListParams, TurbieParams]:
    """
    reads input file where the pattern is: value # variable name [unit] 
    returns raw (RawListParams) and params (elaborated turbine parameters like m1=3*mb; m2=mn+mh+mt; etc.
    """
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # prendi il primo token numerico prima del commento
            token = s.split("#", 1)[0].strip().split()[0]
            vals.append(float(token))

    expected = 14
    if len(vals) < expected:
        raise ValueError(f"Attesi {expected} valori, trovati {len(vals)} in {path}")

    mb, mn, mh, mt, c1, c2, k1, k2, fb, ft, drb, drt, Dr, rho = vals[:expected]

    raw = RawListParams(mb, mn, mh, mt, c1, c2, k1, k2, fb, ft, drb, drt, Dr, rho)

    # 2DOF Turbie model:
    # DOF1 = "rotore+blades" -> mass ~3*mb 
    # DOF2 = "nacelle+hub+tower" -> combined mass mn+mh+mt
    m1 = 3.0 * mb 
    m2 = mn + mh + mt

    params = TurbieParams(
        m1=m1, m2=m2,
        c1=c1, c2=c2,
        k1=k1, k2=k2,
        D_rotor=Dr, rho=rho
    )
    return raw, params

# ---------- Matrices ----------
def build_matrices(p: TurbieParams):
    """
    2DOF:
      M = diag(m1, m2)
      C = [[ c1,     -c1    ],
           [ -c1,  c1 + c2 ]]
      K = [[ k1,     -k1    ],
           [ -k1,  k1 + k2 ]]
    """
    M = np.array([[p.m1, 0.0],
                  [0.0,  p.m2]], dtype=float)

    C = np.array([[ p.c1,     -p.c1],
                  [-p.c1, p.c1 + p.c2]], dtype=float)

    K = np.array([[ p.k1,     -p.k1],
                  [-p.k1, p.k1 + p.k2]], dtype=float)
    return M, C, K

def build_state_matrix(M: np.ndarray, C: np.ndarray, K: np.ndarray):
    Minv = np.linalg.inv(M)
    Z = np.zeros_like(M)
    I = np.eye(M.shape[0])
    A = np.vstack([
        np.hstack([Z, I]),
        np.hstack([-Minv @ K, -Minv @ C])
    ])
    return A

# ---------- check of the 2DOF system----------
def analyze_and_print_system(M: np.ndarray, C: np.ndarray, K: np.ndarray, raw: RawListParams | None = None):
    """
    print matrices, eigenvaleus, resonance freq [Hz] and ζ from A
    """
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    Z = np.zeros_like(M); I = np.eye(n)
    A = np.vstack([np.hstack([Z, I]), np.hstack([-Minv @ K, -Minv @ C])])

    eigs = np.linalg.eigvals(A)
    eigs_pos = np.array([lam for lam in eigs if np.imag(lam) >= 0], dtype=complex)
    sigma = np.real(eigs_pos); omega = np.imag(eigs_pos)
    fn = omega / (2.0 * np.pi)
    zeta = np.array([(-s)/np.hypot(s, w) if (s != 0 or w != 0) else np.nan for s, w in zip(sigma, omega)])

    order = np.argsort(fn)
    fn = fn[order]; zeta = zeta[order]

    print("=== Matrici ===")
    print("M=\n", M, "\nC=\n", C, "\nK=\n", K)
    print("\n=== Modi (da A) ===")
    for i, (f, z) in enumerate(zip(fn, zeta), 1):
        print(f"Mode {i}: f_n = {f:.4f} Hz, zeta = {z:.4f}")

    if raw is not None:
        targets = [("fb", raw.fb, "drb", raw.drb),
                   ("ft", raw.ft, "drt", raw.drt)]
        print("\n=== Confronto target (file) vs calcolato ===")
        for i, (fname, fval, dname, dval) in enumerate(targets, 1):
            if i <= len(fn):
                print(f"{fname}: target={fval:.4f} Hz, calc={fn[i-1]:.4f} Hz | {dname}: target={dval:.4f}, calc={zeta[i-1]:.4f}")
            else:
                print(f"{fname}/{dname}: target fornito ma modi calcolati insufficienti.")


def read_wind_file(base_folder: Path, wind_speed: int | float, TI: float):
    """
    reads wind_{wind_speed}_ms_TI_{TI:.1f}.txt
    returns array N x 2: [time[s], U(t) m/s]
    """
    base_folder = Path(base_folder)
    fileNameWS = f"wind_{int(wind_speed)}_ms_TI_{TI:.1f}.txt"
    found_files = list(base_folder.rglob(fileNameWS))
    if not found_files:
        raise FileNotFoundError(f"Non trovato: {fileNameWS} in {base_folder}")
    filePath = found_files[0]
    data = np.loadtxt(filePath, comments="#", skiprows=1)
    return data  # columns: t [s], U [m/s]

def list_wind_cases(base_folder: Path, TI: float) -> list[Path]:
    """
    list of ws files 'wind_*_ms_TI_{TI:.1f}.txt' 
    """
    base_folder = Path(base_folder)
    files = sorted(base_folder.rglob(f"wind_*_ms_TI_{TI:.1f}.txt"))
    if not files:
        raise FileNotFoundError(f"Nessun file wind_*_ms_TI_{TI:.1f}.txt trovato in {base_folder}")
    return files

def make_thrust_function(Ut: np.ndarray, rho: float, D: float, Ct_lookup_func):
    """
    calculates thrust F(t) = [T(t), 0]^T dove T(t) = 0.5 * rho * A * Ct(U(t)) * U(t)^2
    Ut: array Nx2 with time and U(t)
    Ct_lookup_func: funzione Ct(U) -> float
    """
    t_vec = Ut[:, 0]
    U_vec = Ut[:, 1]
    A = 0.25 * np.pi * D**2

    # ilinear interp of U(t) 
    U_of_t = interp1d(t_vec, U_vec, kind='linear', bounds_error=False,
                      fill_value=(U_vec[0], U_vec[-1]))

    def F_of_t(t: float) -> np.ndarray:
        U = float(U_of_t(t))
        Ct = float(Ct_lookup_func(U))
        T = 0.5 * rho * A * Ct * U**2
        # Assunzione: la forza aerodinamica agisce sul DOF1 (blades/rotore); DOF2 non forzato direttamente
        return np.array([T, 0.0], dtype=float)

    return F_of_t, (t_vec[0], t_vec[-1])


# === ADD: costruzione RHS y' = A y + B(t) per solve_ivp ===
def make_rhs(M: np.ndarray, C: np.ndarray, K: np.ndarray, F_of_t):
    """
    Restituisce f(t, y) compatibile con solve_ivp.
    Stato y = [x1, x2, v1, v2]^T (N=2 DOF)
    """
    n = M.shape[0]
    Minv = np.linalg.inv(M)
    Z = np.zeros_like(M)
    I = np.eye(n)

    A = np.vstack([np.hstack([Z, I]),
                   np.hstack([-Minv @ K, -Minv @ C])])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        # B(t) = [0; Minv * F(t)]
        F = F_of_t(t)
        B_top = np.zeros(n)
        B_bot = Minv @ F
        B = np.concatenate([B_top, B_bot])
        return A @ y + B

    return rhs


# === ADD: integrazione di un singolo caso ===
def simulate_turbie_case(M: np.ndarray, C: np.ndarray, K: np.ndarray,
                         Ut: np.ndarray,
                         Ct_path: str, kind: str,
                         rho: float, D: float,
                         y0: np.ndarray | None = None,
                         rtol: float = 1e-6, atol: float = 1e-9):
    """
    Integra un caso con forzante da vento (Ut: [t, U]) e curva Ct(U) presa da file.
    Ritorna dict con t, x1, x2, v1, v2, U.
    """
    # prepara Ct(U) come funzione (riuso di get_Ct ma senza plotting)
    data = np.loadtxt(Ct_path, comments="#", skiprows=1)
    U_tab, Ct_tab = data[:, 0], data[:, 1]
    Ct_func = interp1d(U_tab, Ct_tab, kind=kind, bounds_error=False,
                       fill_value=(Ct_tab[0], Ct_tab[-1]))

    # F(t)
    F_of_t, (t0, tf) = make_thrust_function(Ut, rho=rho, D=D, Ct_lookup_func=Ct_func)

    # RHS
    rhs = make_rhs(M, C, K, F_of_t)

    # tempi di integrazione: uso la stessa griglia del vento (così è facile salvare)
    t_eval = Ut[:, 0]
    if y0 is None:
        y0 = np.zeros(2* M.shape[0])  # [x1, x2, v1, v2] = 0

    sol = solve_ivp(rhs, (t0, tf), y0, method='RK45', t_eval=t_eval,
                    rtol=rtol, atol=atol, vectorized=False)

    if not sol.success:
        raise RuntimeError(f"solve_ivp non è riuscito: {sol.message}")

    x1 = sol.y[0, :]
    x2 = sol.y[1, :]
    v1 = sol.y[2, :]
    v2 = sol.y[3, :]
    U  = Ut[:, 1]

    return dict(t=sol.t, x1=x1, x2=x2, v1=v1, v2=v2, U=U)


# === ADD: salvatore ===
def save_time_history(out_path: Path, result: dict, meta: dict | None = None):
    """
    Salva su txt: t, U, x1, x2, v1, v2 (+ header).
    """
    out_path = Path(out_path)
    arr = np.column_stack([result['t'], result['U'], result['x1'], result['x2'], result['v1'], result['v2']])
    header = "t[s]\tU[m/s]\tx_blade[m]\tx_tower[m]\tv_blade[m/s]\tv_tower[m/s]"
    if meta:
        meta_str = " | ".join([f"{k}={v}" for k, v in meta.items()])
        header = meta_str + "\n" + header
    np.savetxt(out_path, arr, header=header, comments='', fmt="%.8e")


# === ADD: statistiche ===
def compute_stats(result: dict):
    """
    Ritorna mean/std delle displacement x1 (blade) e x2 (tower).
    """
    x1 = np.asarray(result['x1'])
    x2 = np.asarray(result['x2'])
    stats = {
        'mean_blade': float(np.mean(x1)),
        'std_blade' : float(np.std(x1, ddof=0)),
        'mean_tower': float(np.mean(x2)),
        'std_tower' : float(np.std(x2, ddof=0)),
    }
    return stats


# === ADD: plot comparativo per un caso (vento + spostamenti) ===
def plot_case_wind_and_displacements(result: dict, title: str = ""):
    """
    Due assi y: sinistra per x1,x2; destra per U(t).
    """
    t = result['t']; U = result['U']
    x1 = result['x1']; x2 = result['x2']
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ln1 = ax1.plot(t, x1, label="x_blade [m]", linewidth=1.5)
    ln2 = ax1.plot(t, x2, label="x_tower [m]", linewidth=1.2, linestyle='--')
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("displacement [m]")
    ax1.grid(True, alpha=0.4)

    ax2 = ax1.twinx()
    ln3 = ax2.plot(t, U, label="U(t) [m/s]", alpha=0.7)
    ax2.set_ylabel("wind speed [m/s]")

    # legende combinate
    lines = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    if title:
        ax1.set_title(title)
    fig.tight_layout()
    return fig


# === ADD: salvataggio stats per TI ===
def save_stats_per_TI(out_path: Path, rows: list[dict]):
    """
    rows: lista di dict con chiavi: TI, U_mean, mean_blade, std_blade, mean_tower, std_tower
    """
    out_path = Path(out_path)
    header = "TI\tU_mean[m/s]\tmean_blade[m]\tstd_blade[m]\tmean_tower[m]\tstd_tower[m]"
    arr = np.array([[r['TI'], r['U_mean'], r['mean_blade'], r['std_blade'], r['mean_tower'], r['std_tower']]
                    for r in rows], dtype=float)
    np.savetxt(out_path, arr, header=header, comments='', fmt="%.8e")


# === ADD: plot mean/std vs vento per una TI ===
def plot_stats_vs_wind(TI: float, rows: list[dict]):
    """
    Due grafici: (1) mean vs U_mean, (2) std vs U_mean
    """
    U = np.array([r['U_mean'] for r in rows])
    mean_b = np.array([r['mean_blade'] for r in rows])
    mean_t = np.array([r['mean_tower'] for r in rows])
    std_b  = np.array([r['std_blade'] for r in rows])
    std_t  = np.array([r['std_tower'] for r in rows])

    # means
    fig1, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.plot(U, mean_b, 'o-', label='mean blade [m]')
    ax1.plot(U, mean_t, 's--', label='mean tower [m]')
    ax1.set_xlabel("U_mean [m/s]")
    ax1.set_ylabel("mean displacement [m]")
    ax1.set_title(f"Means vs wind | TI={TI:.2f}")
    ax1.grid(True, alpha=0.4)
    ax1.legend()
    fig1.tight_layout()

    # std
    fig2, ax2 = plt.subplots(figsize=(7.5, 4))
    ax2.plot(U, std_b, 'o-', label='std blade [m]')
    ax2.plot(U, std_t, 's--', label='std tower [m]')
    ax2.set_xlabel("U_mean [m/s]")
    ax2.set_ylabel("std displacement [m]")
    ax2.set_title(f"Standard deviations vs wind | TI={TI:.2f}")
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    fig2.tight_layout()
    return fig1, fig2
