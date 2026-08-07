"""Pure polygon builders for the SG13G2 octagonal spiral inductor.

Each function takes explicit coordinates in the technology's user units (um) and 
returns a polygon with no knowledge of layers, cells, or PCell parameters, so 
they can be unit.
"""

import math

from .compat import pya
from .tech import round_point_to_grid, round_to_grid


def make_45_bridge(tech: "pya.Technology", w: float, l: float, s: float, add_vias: bool = False) -> "pya.DPolygon":
    """A 45-degree crossover bridge connecting two parallel traces of width `w`
    separated by `s`, spanning length `l`. With `add_vias`, the two end pads
    are extended by `w` so an underpass via array fits under them.
    """
    x, y = -(2 * w + s) / 2, -l / 2
    p1 = pya.DPoint(*round_point_to_grid(tech, x, y))
    x += w
    p2 = pya.DPoint(*round_point_to_grid(tech, x, y))
    y += s / (1 + math.sqrt(2))
    p3 = pya.DPoint(*round_point_to_grid(tech, x, y))
    x += w + s
    y += w + s
    p4 = pya.DPoint(x, y)
    y += (w + s) / (1 + math.sqrt(2))
    p5 = pya.DPoint(*round_point_to_grid(tech, x, y))
    x -= w
    p6 = pya.DPoint(*round_point_to_grid(tech, x, y))
    y -= s / (1 + math.sqrt(2))
    p7 = pya.DPoint(*round_point_to_grid(tech, x, y))
    x -= w + s
    y -= w + s
    p8 = pya.DPoint(*round_point_to_grid(tech, x, y))

    if add_vias:
        p1.y -= w
        p2.y -= w
        p5.y += w
        p6.y += w

    return pya.DPolygon([p1, p2, p3, p4, p5, p6, p7, p8])


def oct_fill(r: float) -> "pya.DPolygon":
    """A regular octagon of outer radius `r`, centred at the origin."""
    a = r / (1 + math.sqrt(2))
    return pya.DPolygon([
        pya.DPoint(-a, -r), pya.DPoint(a, -r),
        pya.DPoint(r, -a), pya.DPoint(r, a),
        pya.DPoint(a, r), pya.DPoint(-a, r),
        pya.DPoint(-r, a), pya.DPoint(-r, -a),
    ])


def oct_segment(tech: "pya.Technology", w: float, r: float, e: float) -> "pya.DPolygon":
    """One 45-degree wedge of an octagonal turn at outer radius `r`, with a
    gap of width `e` at the horizontal axis for bridge/underpass access.
    """
    a = round_to_grid(tech, (2 * r) / (math.sqrt(2) + 2))
    r = round_to_grid(tech, r)
    # Inner corner length must be floored instead of rounded.
    c = round_to_grid(tech, (2 * r) / (math.sqrt(2) + 2) + w / (math.sqrt(2) + 1), floor=True)

    x, y = r, e / 2
    p1 = pya.DPoint(x, y)
    y = r - a
    p2 = pya.DPoint(x, y)
    x -= a
    y += a
    p3 = pya.DPoint(x, y)
    x = e / 2
    p4 = pya.DPoint(x, y)
    y -= w
    p5 = pya.DPoint(x, y)
    x = r - c
    p6 = pya.DPoint(x, y)
    x = r - w
    y = r - c
    p7 = pya.DPoint(x, y)
    y = e / 2
    p8 = pya.DPoint(x, y)

    return pya.DPolygon([p1, p2, p3, p4, p5, p6, p7, p8])
