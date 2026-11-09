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
        h=-2.5*2.54e-2,
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
        tf=6.0,
        dt=2.0e-3,
        dr=2.0e-2,
        los=los,
        pp=False,
    )

    # Gaussian bump on cable 0, centred at s = 0.30 * L, amplitude 1 cm,
    # standard deviation 5 % of L. Zero on cable 1.
    s_centre = 0.30
    sigma = 0.05
    bump = 0.01 * np.exp(-((np.concatenate((np.array([0.0]), np.cumsum(ds))) - s_centre) ** 2) / (2.0 * sigma**2))
    # Force the boundary nodes to zero so the IC is compatible with the
    # Dirichlet boundary condition of the solver.
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns)])

    res, _ = bundle.solve(is_uniform, spacer_data, ds, ns, params, force=None, zt=0.0, initial_conditions=ic)

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


if __name__ == "__main__":
    run()



