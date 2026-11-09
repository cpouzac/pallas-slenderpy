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
from slenderpy.force.wind import ConstantWind
from slenderpy.wake import aero_force_bundle
from slenderpy.cable.bundle.bundle import Bundle, InitialConditions
from slenderpy.cable.bundle.cable import Cable
from slenderpy.cable.bundle.spacer import twin_spacer, Spacer, SpacerEdge


def _cable_0():
    return Cable(
        mass=1.25,
        diameter=2.54e-2,
        EA=2.9e07,
        length=240.0,
        tension=1.9e04,
        h=0.0,
    )

def _cable_1():
    return Cable(
        mass=1.25,
        diameter=2.54e-2,
        EA=2.9e07,
        length=240.0,
        tension=1.91e04,
        h=0.0,
    )



#########
# EXEMPLE P62 SNEGOVSKIY 3 SUB-SPANS
#########

def run() -> None:
    cable0 = _cable_0()
    cable1 = _cable_1()

    k_n = 1e9
    c_n = 0.0
    k_b = 1e9
    c_b = 0.0
    is_2d = True
    delta_n = 0.0
    delta_b = 0.0
    j_b = 0.0
    j_t = 0.0
    width = 0.065
    l0 = 15*2.54e-2


    spacer0 = Spacer(
        s=1./3,
        edges=(SpacerEdge(i=0, j=1, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, j_b = j_b, j_t = j_t, 
                  width = width, l0 = l0), ), 
                  is_2d = is_2d, width = width)

    spacer1 = Spacer(
        s=2./3,
        edges=(SpacerEdge(i=0, j=1, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, j_b = j_b, j_t = j_t, 
                  width = width, l0 = l0), ), 
                  is_2d = is_2d, width = width)

    bundle = Bundle(
        cables=[cable0, cable1],
        spacers=[spacer0, spacer1],
    )

    geometry = [(0.0, 0.0), (l0, -2.5*2.54e-2)]

    ns = 321
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
        tf=300.0,
        dt=0.01,
        dr=0.01,
        los=los,
        pp=False,
    )

    my_wind = ConstantWind(speed=10.0)

    force_callback = aero_force_bundle(
    bundle_geometry = geometry, 
    diameter=0.0254, 
    wind=my_wind,         
    aero_model="price",     
    air_density=1.22
)

    ic = InitialConditions(un=[np.zeros(ns), np.zeros(ns)])

    res, _ = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=force_callback, zt=0.0, initial_conditions=ic)

    t_axis = np.asarray(res.lot())
    
    # Récupération du déplacement Y (binormal) pour le câble sous le vent (index 1)
    # ou un selon ce que vous préférez tracer
    y_leeward = res.data["ub_1"].values  # shape (nt, n_los)

    # Création des indices des sous-portées (milieux)
    los_arr = np.array(los)
    subspan_targets = [1/6.0, 3/6.0, 5/6.0]  # milieux des 3 sous-portées de 80m
    subspans = [np.abs(los_arr - target).argmin() for target in subspan_targets]

    # Tracé direct style "autre code"
    fig1 = plt.figure(figsize=[14, 8])
    ax = fig1.add_subplot(111)
    fig1.suptitle("Y-component motions in the leeward subconductors, middle of subspans 1 and 3", fontsize=12)
    
    colors = ['r', 'b', 'k']
    for i in range(len(subspans)):
        label = f"subspan {i+1}"
        node_idx = subspans[i]
        y_signal = y_leeward[:, node_idx]
        ax.plot(t_axis, y_signal, "-", linewidth=1.5, color=colors[i], label=label)
        
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Y displacement [m]")
    ax.grid(True)
    ax.legend()
    fig1.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()



