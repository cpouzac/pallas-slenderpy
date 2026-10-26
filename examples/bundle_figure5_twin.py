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


def run() -> None:
    cable = _aster570()
    bundle = Bundle(
        cables=[cable, cable],
        spacers=[twin_spacer(s=0.5, k_perp=5.0e4, c_perp=200.0)],
    )

    ns = 81
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
    s_grid = np.linspace(0.0, 1.0, ns)
    s_centre = 0.30
    sigma = 0.05
    bump = 0.01 * np.exp(-((s_grid - s_centre) ** 2) / (2.0 * sigma**2))
    # Force the boundary nodes to zero so the IC is compatible with the
    # Dirichlet boundary condition of the solver.
    bump[0] = 0.0
    bump[-1] = 0.0
    ic = InitialConditions(un=[bump, np.zeros(ns)])

    res = bundle.solve(params, force=None, zt=0.0, initial_conditions=ic)

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
