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
import os
import time
from scipy import signal
from pathlib import Path
from scipy.fft import fft, fftfreq
from slenderpy import simulation
from slenderpy.force.wind import ConstantWind
from slenderpy.wake import aero_force_bundle
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import twin_spacer, Spacer, SpacerEdge
from slenderpy.cable.frequency import _ip_frequencies, _op_frequencies


def _cable_0():
    return Cable(
        mass=1.25,
        diameter=3.e-2,
        EA=2.9e07,
        spanlength=45.0,
        tension=1.58e04,
        h=0.0,
    )

def _cable_1():
    return Cable(
        mass=1.25,
        diameter=3.e-2,
        EA=2.9e07,
        spanlength=45.0,
        tension=1.58e04,
        h=0.0,
    )



#########
# EXEMPLE P62 SNEGOVSKIY 3 SUB-SPANS
#########

def run() -> None:
    cable0 = _cable_0()
    cable1 = _cable_1()

    k_n = 0.0
    c_n = 0.0
    k_b = 0.0
    c_b = 0.0
    is_2d = False
    delta_n = 0.0
    delta_b = 0.0
    j_b = 0.0
    j_t = 0.0
    width = 0.065
    l0 = 10*3.e-2



    bundle = Bundle(
        cables=[cable0, cable1],
        spacers=[],
    )

    geometry = [(0.0, -0.2), (l0, -0.2-2.5*3.e-2)]

    ns = 45
    is_uniform = False
    s = np.linspace(0.0, 1.0, ns)
    ds = np.diff(s)

    # Per-cable non-dimensionalisation. All cables share L, m, g (enforced
    # at construction), so tAd, uAd are identical
    L = bundle.cables[0].length
    m = bundle.cables[0].mass
    g = bundle.cables[0].g
    tAd = float(np.sqrt(L / g))
    uAd = L / tAd

    # Spacer pre-processing: precompute interior-node index and non-dim
    # coefficients per edge.
    spacer_data, ds, ns = bundle._prepare_spacers(ds, L, m, g, tAd, is_uniform)
    
    los = np.linspace(0.0, 1.0, ns)[1:-1].tolist()

    params = simulation.Parameters(
        ns=ns,
        t0=0.0,
        tf=100.0,
        dt=0.003,
        dr=0.003,
        los=los,
        pp=False,
    )

    my_wind = ConstantWind(speed=15.0)

    force_callback = aero_force_bundle(
    bundle_geometry = geometry, 
    diameter=0.03, 
    wind=my_wind,         
    aero_model="price",     
    air_density=1.204
)

    ic = InitialConditions(un=[np.zeros(ns), np.zeros(ns)])

    res, _, _, _ = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=(force_callback,), zt=0.0, initial_conditions=ic)

    filename = "Exemple_Snegovskiy_p58"
    save_dir = "./results"

    t_axis = np.asarray(res.lot())

    # Récupération de l'axe temporel
    y1_full = res.data["ub_0"].values
    z1_full = res.data["un_0"].values
    y2_full = res.data["ub_1"].values
    z2_full = res.data["un_1"].values

    # Trouver l'indice du milieu de la portée (s = 0.5)
    los_arr = np.array(los)
    iMidspan = np.abs(los_arr - 0.5).argmin()
    print("milieu = ", los[iMidspan]*45)

    # Extraction des signaux temporels au milieu de la portée
    y1 = y1_full[:, iMidspan]
    z1 = z1_full[:, iMidspan]
    y2 = y2_full[:, iMidspan]
    z2 = z2_full[:, iMidspan]

    # # Constantes récupérées
    # diameter = cable0.diameter
    y_offset_cable1 = geometry[0][0]
    z_offset_cable1 = geometry[0][1] 

    y_offset_cable2 = geometry[1][0]
    z_offset_cable2 = geometry[1][1] 

    # =========================================================
    # FIGURE 1 : Orbites (Trajectories) - Centrées sur les données
    # =========================================================
    fig1 = plt.figure(figsize=[10, 6])
    fig1.suptitle(filename, fontsize=16)
    ax = fig1.add_subplot(111)

    y1_norm = y1 + y_offset_cable1
    z1_norm = z1 + z_offset_cable1
    y2_norm = y2 + y_offset_cable2
    z2_norm = z2 + z_offset_cable2
    
    # Tracé propre
    ax.plot(y1_norm, z1_norm, 'r-', linewidth=1.0, label="cable 1 (windward)")
    ax.plot(y2_norm, z2_norm, 'k-', linewidth=1.0, label="cable 2 (leeward)")

    ax.set_xlabel("y (Out-of-plane)")
    ax.set_ylabel("z (In-plane)")
    ax.set_title("Subspan middle point Orbits")
    
    # On utilise grid et axis('equal') pour ne pas déformer les orbites elliptiques
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.axis('equal') 
    ax.legend(loc='best')

    # Texte d'information (placé proprement dans un encadré)
    info_str = (f"wind speed: {my_wind.speed:.2f} m/s\n"
                f"tension: {cable0.tension} N\n"
                f"EA: {cable0.EA} N\n")
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.02, info_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, f"{filename}_orbits.svg"), format="svg", bbox_inches="tight", dpi=300)

    # =========================================================
    # FIGURE 2 : Évolutions temporelles (Y, Z et profil de vent)
    # =========================================================

    fig2 = plt.figure(figsize=[14, 10])
    
    # Déplacement Y
    ax1 = fig2.add_subplot(311)
    ax1.plot(t_axis, y1, 'r-', label="cable 1")
    ax1.plot(t_axis, y2, 'b-', label="cable 2")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("y [m] (Out-of-plane)")
    ax1.grid(True)
    ax1.legend()

    # Déplacement Z
    ax2 = fig2.add_subplot(312)
    ax2.plot(t_axis, z1, 'r-', label="cable 1")
    ax2.plot(t_axis, z2, 'b-', label="cable 2")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("z [m] (In-plane)")
    ax2.grid(True)
    ax2.legend()

    # Profil de Vent
    ax3 = fig2.add_subplot(313)
    wind_profile = np.full_like(t_axis, my_wind.speed) 
    ax3.plot(t_axis, wind_profile, 'm.-')
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("wind speed [m/s]")
    ax3.grid(True)

    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"{filename}_time_series.svg"), format="svg", bbox_inches="tight", dpi=300)


    # =========================================================
    # FIGURE 3 : Analyse Spectrale (FFT)
    # =========================================================
    f_theorique_y = _op_frequencies(cable0.length, cable0.tension, cable0.h, cable0.mass, n=1)[0]
    f_theorique_z = _ip_frequencies(cable0.length, cable0.tension, cable0.h, cable0.mass, cable0.EA, n=1)[0]

    print(f"Théorie Y (Out-of-plane) : {f_theorique_y:.3f} Hz")
    print(f"Théorie Z (In-plane)     : {f_theorique_z:.3f} Hz")

    dt_out = float(np.median(np.diff(t_axis)))
    n_pts = len(t_axis)
    freq = np.fft.rfftfreq(n_pts, d=dt_out)

    # Normalisation pour obtenir l'amplitude crête approximative en mètres
    scale_fft = 2.0 / n_pts

    spectres = [
        (
            "Y (Out-of-plane)",
            "Displacement Y [m]",
            f_theorique_y,
            np.abs(np.fft.rfft(y1 - np.mean(y1))) * scale_fft,
            np.abs(np.fft.rfft(y2 - np.mean(y2))) * scale_fft,
        ),
        (
            "Z (In-plane)",
            "Displacement Z [m]",
            f_theorique_z,
            np.abs(np.fft.rfft(z1 - np.mean(z1))) * scale_fft,
            np.abs(np.fft.rfft(z2 - np.mean(z2))) * scale_fft,
        ),
    ]

    fig3, axes_fft = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig3.suptitle(f"{filename} - FFT", fontsize=16)

    for ax_fft, (comp_label, y_label, f_theo, spec1, spec2) in zip(axes_fft, spectres):
        ax_fft.plot(freq, spec1, "r-", linewidth=1.0, label="cable 1 (windward)")
        ax_fft.plot(freq, spec2, "b-", linewidth=1.0, label="cable 2 (leeward)")
        ax_fft.axvline(
            f_theo,
            color="g",
            linestyle="--",
            linewidth=1.2,
            label=f"Théorie mode 1 ({f_theo:.3f} Hz)",
        )
        ax_fft.set_ylabel(f"|FFT({comp_label})| [m]")
        ax_fft.set_title(f"Spectre {y_label}")
        ax_fft.grid(True, linestyle=":", alpha=0.7)
        ax_fft.legend(loc="best")

        print(f"\nPics dominants {comp_label} (théorie mode 1 = {f_theo:.3f} Hz) :")
        for spec, color, cable_label in (
            (spec1, "r", "cable 1"),
            (spec2, "b", "cable 2"),
        ):
            peaks, _ = signal.find_peaks(spec)
            if peaks.size > 0:
                order = np.flip(np.argsort(spec[peaks]))[:5]
                top_peaks = peaks[order]
                peaks_str = ", ".join(
                    f"{freq[p]:.3f} Hz (amp≈{spec[p]:.3e} m)" for p in top_peaks
                )
                print(f"  {cable_label} : {peaks_str}")

                for p in top_peaks:
                    ax_fft.annotate(
                        f"{freq[p]:.2f} Hz",
                        xy=(freq[p], spec[p]),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=color,
                    )
            else:
                print(f"  {cable_label} : aucun pic détecté")

    axes_fft[-1].set_xlabel("Frequency [Hz]")
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, f"{filename}_fft.svg"), format="svg", bbox_inches="tight", dpi=300)

    plt.show()

if __name__ == "__main__":
    run()



