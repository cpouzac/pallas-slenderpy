"""Twin bundle: free vibration with energy transfer through a single spacer.

Reproduces the geometric layout of Figure 5 (left) of the Eurobios/RTE note
ESCB-2020-11-12: two parallel ACSR conductors held by a single mid-span
spacer. Cable 0 is released from a Gaussian initial deflection while
cable 1 is at rest; the spacer transfers energy between the two cables.

Run:
    python examples/bundle_figure5_twin.py

Produces a 2-subplot figure:
- left  : space-time map of cable 0 and cable 1 normal displacement
- right : RMS displacement of each cable over time (the spacer transfers
  energy: cable 1's RMS rises while cable 0's relaxes)
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy import simulation
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import twin_spacer


def _aster570():
    return Cable(
        mass=1.57,
        diameter=0.031,
        EA=3.76e07,
        length=200.0,
        tension=3.7e04,
        h=0.0,
    )


#########
# EXEMPLE TWIN BUNDLE
#########

# def run() -> None:
#     cable = _aster570()
#     bundle = Bundle(
#         cables=[cable, cable],
#         spacers=[twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
#                              is_2d = False, delta_n=0.0, delta_b=0.0, j_b=200, j_t=200, width=0.035, l0 = 0.4)],
#     )

#     ns = 81
#     los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
#     is_uniform = False
#     params = simulation.Parameters(
#         ns=ns,
#         t0=0.0,
#         tf=6.0,
#         dt=2.0e-3,
#         dr=2.0e-2,
#         los=los,
#         pp=False,
#     )

#     # Gaussian bump on cable 0, centred at s = 0.30 * L, amplitude 1 cm,
#     # standard deviation 5 % of L. Zero on cable 1.
#     s_grid = np.linspace(0.0, 1.0, ns)
#     s_centre = 0.30
#     sigma = 0.05
#     bump = 0.01 * np.exp(-((s_grid - s_centre) ** 2) / (2.0 * sigma**2))
#     # Force the boundary nodes to zero so the IC is compatible with the
#     # Dirichlet boundary condition of the solver.
#     bump[0] = 0.0
#     bump[-1] = 0.0
#     ic = InitialConditions(un=[bump, np.zeros(ns)])

#     res = bundle.solve(is_uniform, params, force=None, zt=0.0, initial_conditions=ic)

#     t_axis = np.asarray(res.lot())
#     un0 = res.data["un_0"].values  # shape (nt, n_los)
#     un1 = res.data["un_1"].values

#     rms0 = np.sqrt(np.mean(un0**2, axis=1))
#     rms1 = np.sqrt(np.mean(un1**2, axis=1))

#     fig, (ax_map, ax_rms) = plt.subplots(1, 2, figsize=(12, 5))

#     s_phys = np.array(los) * cable.length
#     #extent = (s_phys[0], s_phys[-1], t_axis[-1], t_axis[0])
#     vmax = max(np.max(np.abs(un0)), np.max(np.abs(un1)))
#     ax_map.imshow(
#         np.concatenate([un0, un1], axis=1),
#         aspect="auto",
#         extent=(s_phys[0], 2.0 * s_phys[-1] - s_phys[0], t_axis[-1], t_axis[0]),
#         cmap="RdBu_r",
#         vmin=-vmax,
#         vmax=vmax,
#     )
#     ax_map.axvline(s_phys[-1], color="k", lw=1.2)
#     ax_map.set_xlabel("s [m]   (cable 0  |  cable 1)")
#     ax_map.set_ylabel("time [s]")
#     ax_map.set_title("normal displacement, cable 0 (left) | cable 1 (right)")

#     ax_rms.plot(t_axis, rms0 * 1e3, label="cable 0 (excited)", color="C0")
#     ax_rms.plot(t_axis, rms1 * 1e3, label="cable 1 (passive)", color="C1")
#     ax_rms.set_xlabel("time [s]")
#     ax_rms.set_ylabel("RMS normal displacement [mm]")
#     ax_rms.set_title("Energy transfer through the central spacer")
#     ax_rms.legend()
#     ax_rms.grid(True, alpha=0.3)

#     fig.suptitle(
#         "Twin bundle, mid-span spacer, Gaussian IC on cable 0 - "
#         "Figure 5 (left) of the note"
#     )
#     fig.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     run()









###########
# COMPARAISON MODELES ENTRETOISE 
###########

def run_simulation(mode="1D", width=0.035):
    """
    Exécute la simulation pour un mode donné ('None', '1D', ou '2D')
    et retourne l'axe temporel et les RMS des deux câbles.
    """
    cable = _aster570()
    
    # Configuration de l'entretoise selon le mode
    if mode == "None":
        spacers = [twin_spacer(s=0.5, k_n=0.0, c_n=0.0, k_b=0.0, c_b=0.0,
                               is_2d=False, delta_n=0.0, delta_b=0.0)]
    elif mode == "1D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=False, delta_n=0.0, delta_b=0.0)]
    elif mode == "2D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=True, delta_n=0.0, delta_b=0.0, 
                               j_b=200, j_t=200, width=width, l0=0.4)]
    
    bundle = Bundle(cables=[cable, cable], spacers=spacers)

    ns = 81
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    # On active le maillage non-uniforme uniquement pour le 2D
    is_uniform = False if mode == "2D" else True 

    params = simulation.Parameters(
        ns=ns, t0=0.0, tf=6.0, dt=2.0e-3, dr=2.0e-2, los=los, pp=False,
    )

    s_grid = np.linspace(0.0, 1.0, ns)
    s_centre = 0.30
    sigma = 0.05
    # CORRECTION : Ajout des symboles de multiplication manquants
    bump = 0.01 * np.exp(-((s_grid - s_centre) ** 2) / (2.0 * sigma**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    
    ic = InitialConditions(un=[bump, np.zeros(ns)])
    res = bundle.solve(is_uniform, params, force=None, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    un0 = res.data["un_0"].values
    un1 = res.data["un_1"].values

    rms0 = np.sqrt(np.mean(un0**2, axis=1)) * 1e3 # Conversion en mm
    rms1 = np.sqrt(np.mean(un1**2, axis=1)) * 1e3 # Conversion en mm

    return t_axis, rms0, rms1

def comparative_study():
    # Figure légèrement plus large pour faire de la place à la légende
    plt.figure(figsize=(12, 6))
    
    # 1. Test sans entretoise
    print("Simulation: Sans entretoise...")
    t, r0_none, r1_none = run_simulation(mode="None")
    plt.plot(t, r0_none, color='grey', linestyle=':', linewidth=2, label="Câble 0 (Sans entretoise)")
    plt.plot(t, r1_none, color='black', linestyle='--', linewidth=2, label="Câble 1 (Sans entretoise)")
    
    # 2. Test 1D
    print("Simulation: 1D...")
    t, r0_1d, r1_1d = run_simulation(mode="1D")
    plt.plot(t, r0_1d, color='blue', linestyle='--', alpha=0.7, label="Câble 0 (1D)")
    plt.plot(t, r1_1d, color='blue', linestyle='-', alpha=0.7, label="Câble 1 (1D)")
    
    # 3. Test 2D - Impact de la largeur (width)
    widths_to_test = [0.01, 0.05, 0.20] # 1cm, 5cm, 20cm
    colors = ['r', 'g', 'm']
    
    for w, col in zip(widths_to_test, colors):
        print(f"Simulation: 2D avec width = {w}m ...")
        t, r0_2d, r1_2d = run_simulation(mode="2D", width=w)
        # Câble 0 en pointillés
        plt.plot(t, r0_2d, color=col, linestyle='--', alpha=0.7, label=f"Câble 0 (2D, width={w}m)")
        # Câble 1 en trait plein
        plt.plot(t, r1_2d, color=col, linestyle='-', label=f"Câble 1 (2D, width={w}m)")

    plt.xlabel("Temps [s]")
    plt.ylabel("RMS Déplacement Normal [mm]")
    plt.title("Transfert d'énergie : Câble 0 (excité) et Câble 1 (passif)")
    plt.grid(True, alpha=0.3)
    
    # Placement de la légende à l'extérieur (à droite)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    comparative_study()






#########
# ETUDE DE CONVERGENCE EN MAILLAGE
#########

# Maillage uniforme avec entretoise 2d, width = ds

# def run_convergence_sim(ns, tf=1.0, dt=5.0e-4):
#     """
#     Fait tourner le solveur pour un nombre de noeuds `ns`.
#     La largeur de l'entretoise est strictement forcée à 1 maille (ds).
#     Retourne le vecteur complet du déplacement (bords inclus) à t=tf.
#     """
#     cable = _aster570()
#     L = cable.length
    
#     # 1. Forçage de la largeur de l'entretoise à EXACTEMENT 1 maille
#     ds_phys = L / (ns - 1)
    
#     bundle = Bundle(
#         cables=[cable, cable],
#         spacers=[twin_spacer(
#             s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
#             is_2d=True, delta_n=0.0, delta_b=0.0,
#             j_b=200, j_t=200, width=ds_phys, l0=0.40 # Largeur glissante
#         )],
#     )

#     los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    
#     # On sort uniquement le dernier pas de temps (dr = tf)
#     params = simulation.Parameters(
#         ns=ns, t0=0.0, tf=tf, dt=dt, dr=tf, los=los, pp=False,
#     )

#     s_grid = np.linspace(0.0, 1.0, ns)
#     bump = 0.01 * np.exp(-((s_grid - 0.30) ** 2) / (2.0 * 0.05**2))
#     bump[0] = 0.0
#     bump[-1] = 0.0
    
#     ic = InitialConditions(un=[bump, np.zeros(ns)])
    
#     # On garantit au solveur que le maillage est uniforme
#     res = bundle.solve(is_uniform=True, parameters=params, force=None, zt=0.0, initial_conditions=ic)

#     # Extraction du câble 1 (passif) à l'instant final
#     un1_interior = res.data["un_1"].values[-1, :]
    
#     # Reconstitution du vecteur complet (avec les zéros aux appuis)
#     un1_full = np.concatenate(([0.0], un1_interior, [0.0]))
#     return un1_full

# def spatial_convergence_study():
#     # Grilles emboîtées : ns = 2^k * 10 + 1 (pour tomber juste sur les mêmes noeuds physiques)
#     ns_list = [41, 81, 161, 321]
#     ns_ref = 641
#     L = 200.0
    
#     print(f"Calcul de la solution de référence (ns={ns_ref})...")
#     u_ref = run_convergence_sim(ns_ref)
    
#     errors = []
#     ds_values = []
    
#     for ns in ns_list:
#         print(f"Calcul pour ns={ns}...")
#         u_coarse = run_convergence_sim(ns)
        
#         ds = L / (ns - 1)
#         ds_values.append(ds)
        
#         # Le ratio d'emboîtement (ex: entre 641 et 81, le ratio est (640/80) = 8)
#         step = (ns_ref - 1) // (ns - 1)
        
#         # Sous-échantillonnage de la référence pour comparer sur les mêmes noeuds physiques
#         u_ref_downsampled = u_ref[::step]
        
#         # Erreur Norme L2 (Moyenne quadratique de l'écart spatial)
#         l2_error = np.sqrt(np.mean((u_coarse - u_ref_downsampled)**2))
#         errors.append(l2_error)
        
#         print(f"  -> ds = {ds:.4f} m | L2 Error = {l2_error:.4e}")

#     pente, intercept = np.polyfit(np.log(ds_values), np.log(errors), 1)
#     print(f"Pente calculée par régression : {pente:.3f}")

#     # -------- TRACÉ LOG-LOG --------
#     plt.figure(figsize=(8, 6))
    
#     # MODIFICATION : Intégration de la pente dans la légende
#     plt.loglog(ds_values, errors, 'bo-', 
#                label=f'Erreur du modèle (Pente mesurée : {pente:.2f})', 
#                linewidth=2)
    
#     # Tracer la pente théorique d'ordre 2 (O(ds^2))
#     # On aligne la pente théorique sur le dernier point pour faciliter la lecture visuelle
#     ds_array = np.array(ds_values)
#     ref_slope = (ds_array / ds_array[-1])**2 * errors[-1]
#     plt.loglog(ds_values, ref_slope, 'k--', label='Pente théorique (Ordre 2)')
    
#     plt.xlabel('Taille de la maille $ds$ [m]')
#     plt.ylabel('Erreur Norme $L_2$ par rapport à la référence [m]')
#     plt.title('Étude de convergence spatiale (Largeur entretoise = $ds$)')
#     plt.grid(True, which="both", ls="--", alpha=0.5)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     spatial_convergence_study()




# Maillage 1D uniforme et maillage 2D avec entretoise de largeur fixe

# def run_convergence_sim(mode, ns, tf=1.0, dt=5.0e-4, fixed_width=0.1):
#     """
#     Exécute le solveur pour un mode donné ('1D' ou '2D').
#     En 2D, la largeur est fixée, ce qui force un maillage irrégulier.
#     """
#     cable = _aster570()
    
#     if mode == "1D":
#         spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
#                                is_2d=False, delta_n=0.0, delta_b=0.0)]
#         is_uniform = True
#     elif mode == "2D":
#         spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
#                                is_2d=True, delta_n=0.0, delta_b=0.0,
#                                j_b=200, j_t=200, width=fixed_width, l0=0.40)]
#         is_uniform = False

#     bundle = Bundle(cables=[cable, cable], spacers=spacers)
#     los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    
#     params = simulation.Parameters(
#         ns=ns, t0=0.0, tf=tf, dt=dt, dr=tf, los=los, pp=False,
#     )

#     s_grid = np.linspace(0.0, 1.0, ns)
#     bump = 0.01 * np.exp(-((s_grid - 0.30) ** 2) / (2.0 * 0.05**2))
#     bump[0] = 0.0
#     bump[-1] = 0.0
#     ic = InitialConditions(un=[bump, np.zeros(ns)])
    
#     res = bundle.solve(is_uniform=is_uniform, parameters=params, force=None, zt=0.0, initial_conditions=ic)

#     # Récupération du câble 1 au dernier pas de temps
#     un1_interior = res.data["un_1"].values[-1, :]
#     un1_full = np.concatenate(([0.0], un1_interior, [0.0]))
#     return un1_full


# def order_of_convergence_study():
#     ns_list = [41, 81, 161, 321]
#     ns_ref = 641
#     L = 200.0
    
#     # ---------------------------------------------------------
#     # 1. RÉFÉRENCES (Solution très fine)
#     # ---------------------------------------------------------
#     print(f"Calcul des références (ns={ns_ref})...")
#     u_ref_1d = run_convergence_sim("1D", ns_ref)
#     u_ref_2d = run_convergence_sim("2D", ns_ref, fixed_width=0.1)
    
#     err_1d, err_2d, ds_values = [], [], []
    
#     # ---------------------------------------------------------
#     # 2. BOUCLE SUR LES MAILLAGES GROSSIERS
#     # ---------------------------------------------------------
#     for ns in ns_list:
#         print(f"Calcul pour ns={ns}...")
#         u_coarse_1d = run_convergence_sim("1D", ns)
#         u_coarse_2d = run_convergence_sim("2D", ns, fixed_width=0.1)
        
#         ds = L / (ns - 1)
#         ds_values.append(ds)
#         step = (ns_ref - 1) // (ns - 1)
        
#         # Sous-échantillonnage pour comparer noeud à noeud
#         u_ref_1d_down = u_ref_1d[::step]
#         u_ref_2d_down = u_ref_2d[::step]
        
#         # Erreur Norme L2
#         err_1d.append(np.sqrt(np.mean((u_coarse_1d - u_ref_1d_down)**2)))
#         err_2d.append(np.sqrt(np.mean((u_coarse_2d - u_ref_2d_down)**2)))
        
#     pente_1d, intercept_1d = np.polyfit(np.log(ds_values), np.log(err_1d), 1)
#     pente_2d, intercept_2d = np.polyfit(np.log(ds_values), np.log(err_2d), 1)
    
#     print(f"Pente 1D calculée : {pente_1d:.3f}")
#     print(f"Pente 2D calculée : {pente_2d:.3f}")

#     # ---------------------------------------------------------
#     # 3. TRACÉ LOG-LOG DES ORDRES DE CONVERGENCE
#     # ---------------------------------------------------------
#     plt.figure(figsize=(9, 7))
#     ds_arr = np.array(ds_values)
    
#     # MODIFICATION : Trace des erreurs avec les pentes intégrées
#     # plt.loglog(ds_arr, err_1d, 'bo-', 
#     #            label=f'Modèle 1D (Maillage uniforme) - Pente : {pente_1d:.2f}', 
#     #            markersize=6)
#     plt.loglog(ds_arr, err_2d, 'ro-', 
#                label=f'Modèle 2D (Maillage modifié) - Pente : {pente_2d:.2f}', 
#                markersize=6)

#     # Trace des pentes théoriques (Ordre 1 et Ordre 2) calées sur le dernier point
#     slope_O2 = (ds_arr / ds_arr[-1])**2 * err_1d[-1]
#     slope_O1 = (ds_arr / ds_arr[-1])**1 * err_2d[-1]
    
#     # plt.loglog(ds_arr, slope_O2, 'b--', alpha=0.6, label='Pente théorique ordre 2')
#     plt.loglog(ds_arr, slope_O1, 'r--', alpha=0.6, label='Pente théorique ordre 1')
    
#     plt.xlabel('Taille maille $\Delta s$ [m]')
#     plt.ylabel('Erreur Norme $L_2$ [m]')
#     plt.title("Convergence spatiale")
#     plt.grid(True, which="both", ls=":", alpha=0.7)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     order_of_convergence_study()