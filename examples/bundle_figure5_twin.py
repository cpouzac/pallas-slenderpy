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

def run() -> None:
    cable = _aster570()
    bundle = Bundle(
        cables=[cable, cable],
        spacers=[twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                             is_2d = False, delta_n=0.0, delta_b=0.0, j_b=200, j_t=200, width=0.035, l0 = 0.4)],
    )

    ns = 81
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    is_uniform = False
    params = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=6.0,
        dt=2.0e-3,
        dr=2.0e-2,
        los=los,
        pp=False,
    )

    # Gaussian bump on cable 0, centred at s = 0.30 * L, amplitude 1 cm,
    # standard deviation 5 % of L. Zero on cable 1.
    s_grid = np.linspace(0.0, 1.0, ns)
    s_centre = 0.30
    sigma = 0.05
    bump = 0.01 * np.exp(-((s_grid - s_centre) ** 2) / (2.0 * sigma**2))
    # Force the boundary nodes to zero so the IC is compatible with the
    # Dirichlet boundary condition of the solver.
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns)])

    res = bundle.solve(is_uniform, params, force=None, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    un0 = res.data["un_0"].values  # shape (nt, n_los)
    un1 = res.data["un_1"].values

    rms0 = np.sqrt(np.mean(un0**2, axis=1))
    rms1 = np.sqrt(np.mean(un1**2, axis=1))

    fig, (ax_map, ax_rms) = plt.subplots(1, 2, figsize=(12, 5))

    s_phys = np.array(los) * cable.length
    #extent = (s_phys[0], s_phys[-1], t_axis[-1], t_axis[0])
    vmax = max(np.max(np.abs(un0)), np.max(np.abs(un1)))
    ax_map.imshow(
        np.concatenate([un0, un1], axis=1),
        aspect="auto",
        extent=(s_phys[0], 2.0 * s_phys[-1] - s_phys[0], t_axis[-1], t_axis[0]),
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_map.axvline(s_phys[-1], color="k", lw=1.2)
    ax_map.set_xlabel("s [m]   (cable 0  |  cable 1)")
    ax_map.set_ylabel("time [s]")
    ax_map.set_title("normal displacement, cable 0 (left) | cable 1 (right)")

    ax_rms.plot(t_axis, rms0 * 1e3, label="cable 0 (excited)", color="C0")
    ax_rms.plot(t_axis, rms1 * 1e3, label="cable 1 (passive)", color="C1")
    ax_rms.set_xlabel("time [s]")
    ax_rms.set_ylabel("RMS normal displacement [mm]")
    ax_rms.set_title("Energy transfer through the central spacer")
    ax_rms.legend()
    ax_rms.grid(True, alpha=0.3)

    fig.suptitle(
        "Twin bundle, mid-span spacer, Gaussian IC on cable 0 - "
        "Figure 5 (left) of the note"
    )
    fig.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     run()









###########
# COMPARAISON MODELES ENTRETOISE 
###########

def run_simulation(mode="1D", width=0.035):
    """
    Simulation for a spacer model ('None', '1D', ou '2D')

    """
    cable = _aster570()

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
    is_uniform = False if mode == "2D" else True 

    params = simulation.Parameters(
        ns=ns, t0=0.0, tf=6.0, dt=2.0e-3, dr=2.0e-2, los=los, pp=False,
    )

    s_grid = np.linspace(0.0, 1.0, ns)
    s_centre = 0.30
    sigma = 0.05
    bump = 0.01 * np.exp(-((s_grid - s_centre) ** 2) / (2.0 * sigma**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    
    ic = InitialConditions(un=[bump, np.zeros(ns)])
    res = bundle.solve(is_uniform, params, force=None, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    un0 = res.data["un_0"].values
    un1 = res.data["un_1"].values

    rms0 = np.sqrt(np.mean(un0**2, axis=1)) * 1e3 
    rms1 = np.sqrt(np.mean(un1**2, axis=1)) * 1e3 

    return t_axis, rms0, rms1

def comparative_study():
  
    plt.figure(figsize=(12, 6))
    
    # No spacer
    print("Simulation: no spacer...")
    t, r0_none, r1_none = run_simulation(mode="None")
    # plt.plot(t, r0_none, color='grey', linestyle=':', linewidth=2, label="Cable 0 (No spacer)")
    plt.plot(t, r1_none, color='black', linestyle='--', linewidth=2, label="Cable 1 (No spacer)")
    
    # 1D spacer
    print("Simulation: 1D...")
    t, r0_1d, r1_1d = run_simulation(mode="1D")
    # plt.plot(t, r0_1d, color='blue', linestyle='--', alpha=0.7, label="Cable 0 (1D)")
    plt.plot(t, r1_1d, color='blue', linestyle='-', alpha=0.7, label="Cable 1 (1D)")
    
    # 2D spacer
    widths_to_test = [0.01, 0.05, 0.20] # 1cm, 5cm, 20cm
    colors = ['r', 'g', 'm']
    
    for w, col in zip(widths_to_test, colors):
        print(f"Simulation: 2D width = {w}m ...")
        t, r0_2d, r1_2d = run_simulation(mode="2D", width=w)
        # Cable 0
        # plt.plot(t, r0_2d, color=col, linestyle='--', alpha=0.7, label=f"Cable 0 (2D, width={w}m)")
        # Cable 1 
        plt.plot(t, r1_2d, color=col, linestyle='-', label=f"Cable 1 (2D, width={w}m)")

    plt.xlabel("Time [s]")
    plt.ylabel("RMS normal displacement [mm]")
    plt.title("Energy transfer: cable 1 (passive) through cable 0 (excited)")

    plt.grid(True, alpha=0.3)
 
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    comparative_study()






#########
# ETUDE DE CONVERGENCE EN MAILLAGE
#########

# Uniform mesh, 2D spacer: width forced to width = ds
def run_convergence_sim(ns, tf=1.0, dt=5.0e-4):
    """
    Displacement, boundaries included, at t=tf.
    """
    cable = _aster570()
    L = cable.length
    
    # Spacer width
    ds_phys = L / (ns - 1)
    
    bundle = Bundle(
        cables=[cable, cable],
        spacers=[twin_spacer(
            s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
            is_2d=True, delta_n=0.0, delta_b=0.0,
            j_b=200, j_t=200, width=ds_phys, l0=0.40 
        )],
    )

    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    
    # Last time step
    params = simulation.Parameters(
        ns=ns, t0=0.0, tf=tf, dt=dt, dr=tf, los=los, pp=False,
    )

    s_grid = np.linspace(0.0, 1.0, ns)
    bump = 0.01 * np.exp(-((s_grid - 0.30) ** 2) / (2.0 * 0.05**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    
    ic = InitialConditions(un=[bump, np.zeros(ns)])
    
    res = bundle.solve(is_uniform=True, parameters=params, force=None, zt=0.0, initial_conditions=ic)

    # Cable 1
    un1_interior = res.data["un_1"].values[-1, :]
    
    # Full vector
    un1_full = np.concatenate(([0.0], un1_interior, [0.0]))
    return un1_full

def spatial_convergence_study():
    ns_list = [41, 81, 161, 321]
    ns_ref = 641
    L = 200.0
    
    print(f"Calcul de la solution de référence (ns={ns_ref})...")
    u_ref = run_convergence_sim(ns_ref)
    
    errors = []
    ds_values = []
    
    for ns in ns_list:
        print(f"Calcul pour ns={ns}...")
        u_coarse = run_convergence_sim(ns)
        
        ds = L / (ns - 1)
        ds_values.append(ds)
        
        # LRatio computation
        step = (ns_ref - 1) // (ns - 1)
        
        u_ref_downsampled = u_ref[::step]
        
        # Error norm L2 
        l2_error = np.sqrt(np.mean((u_coarse - u_ref_downsampled)**2))
        errors.append(l2_error)
        
        print(f"  -> ds = {ds:.4f} m | L2 Error = {l2_error:.4e}")

    pente, intercept = np.polyfit(np.log(ds_values), np.log(errors), 1)
    print(f"Slope computed by regression: {pente:.3f}")

    plt.figure(figsize=(8, 6))
    
    plt.loglog(ds_values, errors, 'bo-', 
               label=f'Error of the model (Measured slope: {pente:.2f})', 
               linewidth=2)
    
    ds_array = np.array(ds_values)
    ref_slope = (ds_array / ds_array[-1])**2 * errors[-1]
    plt.loglog(ds_values, ref_slope, 'k--', label='Theoritical slope (Order 2)')
    
    plt.xlabel('Mesh size $ds$ [m]')
    plt.ylabel('Error norm $L_2$ with reference mesh [m]')
    plt.title('Spatial convergence study (Spacer width = $ds$)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     spatial_convergence_study()




# Maillage 1D uniforme et maillage 2D avec entretoise de largeur fixe

def run_convergence_sim(mode, ns, tf=1.0, dt=5.0e-4, fixed_width=0.1):
    """
    1D convergence study withe uniform mesh.
    In 2D, the width is constant, forcing a non-uniform mesh.
    """
    cable = _aster570()
    
    if mode == "1D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=False, delta_n=0.0, delta_b=0.0)]
        is_uniform = True
    elif mode == "2D":
        spacers = [twin_spacer(s=0.5, k_n=5.0e4, c_n=200.0, k_b=5.0e4, c_b=200.0,
                               is_2d=True, delta_n=0.0, delta_b=0.0,
                               j_b=200, j_t=200, width=fixed_width, l0=0.40)]
        is_uniform = False

    bundle = Bundle(cables=[cable, cable], spacers=spacers)
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()
    
    params = simulation.Parameters(
        ns=ns, t0=0.0, tf=tf, dt=dt, dr=tf, los=los, pp=False,
    )

    s_grid = np.linspace(0.0, 1.0, ns)
    bump = 0.01 * np.exp(-((s_grid - 0.30) ** 2) / (2.0 * 0.05**2))
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns)])
    
    res = bundle.solve(is_uniform=is_uniform, parameters=params, force=None, zt=0.0, initial_conditions=ic)

    # Cable 1 last time step
    un1_interior = res.data["un_1"].values[-1, :]
    un1_full = np.concatenate(([0.0], un1_interior, [0.0]))
    return un1_full


def order_of_convergence_study():
    ns_list = [41, 81, 161, 321]
    ns_ref = 641
    L = 200.0

    # Reference solution
    print(f"Calcul des références (ns={ns_ref})...")
    u_ref_1d = run_convergence_sim("1D", ns_ref)
    u_ref_2d = run_convergence_sim("2D", ns_ref, fixed_width=0.1)
    
    err_1d, err_2d, ds_values = [], [], []
    
    # Coarse meshes
    for ns in ns_list:
        print(f"Calcul pour ns={ns}...")
        u_coarse_1d = run_convergence_sim("1D", ns)
        u_coarse_2d = run_convergence_sim("2D", ns, fixed_width=0.1)
        
        ds = L / (ns - 1)
        ds_values.append(ds)
        step = (ns_ref - 1) // (ns - 1)
        
        u_ref_1d_down = u_ref_1d[::step]
        u_ref_2d_down = u_ref_2d[::step]
        
        # Error norm L2
        err_1d.append(np.sqrt(np.mean((u_coarse_1d - u_ref_1d_down)**2)))
        err_2d.append(np.sqrt(np.mean((u_coarse_2d - u_ref_2d_down)**2)))
        
    pente_1d, intercept_1d = np.polyfit(np.log(ds_values), np.log(err_1d), 1)
    pente_2d, intercept_2d = np.polyfit(np.log(ds_values), np.log(err_2d), 1)
    
    print(f"1D computed slope : {pente_1d:.3f}")
    print(f"2D computed slope : {pente_2d:.3f}")


    plt.figure(figsize=(9, 7))
    ds_arr = np.array(ds_values)
    
    plt.loglog(ds_arr, err_2d, 'ro-', 
               label=f'2D model (modified mesh) - Slope : {pente_2d:.2f}', 
               markersize=6)

    # Trace des pentes théoriques (Ordre 1 et Ordre 2) calées sur le dernier point
    slope_O2 = (ds_arr / ds_arr[-1])**2 * err_1d[-1]
    slope_O1 = (ds_arr / ds_arr[-1])**1 * err_2d[-1]
    
    # plt.loglog(ds_arr, slope_O2, 'b--', alpha=0.6, label='Theoritical slope order 2')
    plt.loglog(ds_arr, slope_O1, 'r--', alpha=0.6, label='Theoritical slope order 1')
    
    plt.xlabel('Mesh size $\Delta s$ [m]')
    plt.ylabel('Error norm $L_2$ [m]')
    plt.title("Spatial convergence")
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     order_of_convergence_study()