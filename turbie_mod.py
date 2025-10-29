import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from dataclasses import dataclass

def Ct_look_up_and_interp(path, kind='linear', plot=True):
    """
    Legge il file C_T.txt, crea la funzione interpolante e (opzionalmente) plottala.
    """
    # --- Caricamento dati ---
    data = np.loadtxt(path, comments="#", skiprows=1)
    U_tab, Ct_tab = data[:, 0], data[:, 1]

    # --- Interpolazione ---
    get_Ct = interp1d(
        U_tab, Ct_tab,
        kind=kind,
        bounds_error=False,
        fill_value=(Ct_tab[0], Ct_tab[-1])
    )

    if plot:
        # more points for plotting
        U_fine = np.linspace(U_tab.min(), U_tab.max(), 200)
        Ct_fine = get_Ct(U_fine)

        plt.figure(figsize=(7, 4))
        plt.plot(U_tab, Ct_tab, 'o', label='Look-up table')
        plt.plot(U_fine, Ct_fine, '-', label=f'Interpolation {kind}', linewidth=2)
        plt.xlabel("Wind speed U [m/s]")
        plt.ylabel("Thrust coefficient $C_T$ [-]")
        plt.title(f"{kind.capitalize()} interpolation of $C_T(U)$")
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return get_Ct, U_tab, Ct_tab

fullpath = os.path.join("inputs", "turbie_inputs", "CT.txt")

# -------------plotting---------------
get_Ct, U_tab, Ct_tab = Ct_look_up_and_interp(fullpath, kind='linear', plot=True)

# ------------get_Ct function tested on a test w/s-------------------
U_test = 10.5
print(f"C_T({U_test:.1f} m/s) = {get_Ct(U_test):.3f}")

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

# -------read parameters and build matrices using the functions------------
param_path = os.path.join("inputs", "turbie_inputs", "turbie_parameters.txt")  
raw, P = load_parameters_from_listfile(param_path)
M, C, K = build_matrices(P)

# print the matrices and checks comparing with fb/ft, drb/drt from input file
analyze_and_print_system(M, C, K, raw=raw)


# base = os.path.join("inputs", "wind_files")

# # Controlla che la cartella esista
# if not os.path.isdir(base):
#     raise FileNotFoundError(f"Cartella non trovata: {base}")

# # Scorre tutte le sottocartelle (es. wind_TI_0.05, wind_TI_0.10, ecc.)
# for subfolder in sorted(os.listdir(base)):
#     path = os.path.join(base, subfolder)
#     if os.path.isdir(path):   # solo cartelle
#         print(f"\n🔹 Cartella: {path}")
#         txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]
#         if not txt_files:
#             print("  (nessun file .txt trovato)")
#             continue

#         # Scorre tutti i file .txt dentro la sottocartella
#         for file in sorted(txt_files):
#             fullpath = os.path.join(path, file)
#             try:
#                 data = np.loadtxt(fullpath, comments="#", skiprows=1)
#                 t, u = data[:,0], data[:,1]
#                 print(f"  ✅ File: {file}  →  {data.shape[0]} righe, {data.shape[1] if data.ndim>1 else 1} colonne")
#             except Exception as e:
#                 print(f"  ⚠️ Errore nel leggere {file}: {e}")
