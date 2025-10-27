import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

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
        # Griglia fine per il plot
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

# Esegue la funzione e mostra il plot
get_Ct, U_tab, Ct_tab = Ct_look_up_and_interp(fullpath, kind='linear', plot=True)

# Ora puoi calcolare C_T per qualsiasi velocità del vento:
U_test = 10.5
print(f"C_T({U_test:.1f} m/s) = {get_Ct(U_test):.3f}")



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
