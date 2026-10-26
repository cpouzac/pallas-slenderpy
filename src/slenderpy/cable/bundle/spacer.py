"""Spacer descriptors for bundled conductors.

A bundle of N sub-conductors is held in a fixed geometric arrangement by
spacers installed at discrete locations along the span. Each spacer is
described by its curvilinear location s in ]0, 1[ and a set of edges, where
each edge couples one pair of sub-conductors via a linear spring-dashpot in
the plane perpendicular to the span (the normal-binormal plane of the local
triad). Rotational and torsional stiffnesses are omitted; see
doc/bundle.rst for the rationale and references.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SpacerEdge:
    """One sub-conductor pair linked by a spacer.

    Parameters
    ----------
    i, j : int
        Sub-conductor indices (0-based) of the two endpoints. Must differ.
    k_perp : float
        Translational stiffness (N/m), isotropic in the (n, b) plane.
    c_perp : float
        Translational damping (N.s/m), isotropic in the (n, b) plane.
    delta_n, delta_b : float
        Rest offset (m) between sub-conductor i and sub-conductor j in the
        normal and binormal directions. For most bundles the static
        equilibrium catenaries already encode the geometric offset, so the
        common case is delta_n = delta_b = 0 (the spacer exerts no force at
        rest because the state is the offset from static equilibrium).
    """

    i: int
    j: int
    k_perp: float
    c_perp: float
    delta_n: float = 0.0
    delta_b: float = 0.0

    def __post_init__(self) -> None:
        if self.i == self.j:
            raise ValueError("SpacerEdge endpoints i and j must differ")
        if self.i < 0 or self.j < 0:
            raise ValueError("SpacerEdge indices must be non-negative")
        if self.k_perp < 0.0:
            raise ValueError("SpacerEdge k_perp must be non-negative")
        if self.c_perp < 0.0:
            raise ValueError("SpacerEdge c_perp must be non-negative")


@dataclass(frozen=True)
class Spacer:
    """A spacer at a single curvilinear location coupling sub-conductors.

    Parameters
    ----------
    s : float
        Curvilinear abscissa in ]0, 1[ along the normalized span.
    edges : tuple of SpacerEdge
        Pairs of sub-conductors linked by this spacer. A star spacer with a
        central hub is encoded as a set of edges from the hub-conductor to
        each leaf; a ring spacer is a cycle.
    """

    s: float
    edges: Tuple[SpacerEdge, ...]

    def __post_init__(self) -> None:
        if not 0.0 < self.s < 1.0:
            raise ValueError("Spacer s must be in ]0, 1[")
        if len(self.edges) == 0:
            raise ValueError("Spacer must have at least one edge")


def twin_spacer(s: float, k_perp: float, c_perp: float) -> Spacer:
    """Build the common two-sub-conductor spacer (edge 0-1, zero rest offset).

    Parameters
    ----------
    s : float
        Curvilinear location in ]0, 1[.
    k_perp : float
        Translational stiffness (N/m).
    c_perp : float
        Translational damping (N.s/m).

    Returns
    -------
    Spacer
    """
    return Spacer(s=s, edges=(SpacerEdge(i=0, j=1, k_perp=k_perp, c_perp=c_perp),))
