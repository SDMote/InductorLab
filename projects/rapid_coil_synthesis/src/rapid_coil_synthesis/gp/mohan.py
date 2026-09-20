"""
Monomial coefficients for inductance expression.

Taken from 'Simple Accurate Expressions for Planar Spiral Inductances' by
Mohan et al.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class InductanceCoefficients:
  """Monomial coefficients for the Mohan inductance expression

      L_nH = beta * d_out^a1 * w^a2 * d_avg^a3 * n^a4 * s^a5

  with every length in microns and L in nanohenries. 
  """
  beta: float
  a1:   float   # exponent of d_out
  a2:   float   # exponent of w
  a3:   float   # exponent of d_avg
  a4:   float   # exponent of n
  a5:   float   # exponent of s

@dataclass(frozen=True)
class ShapeParams:
  """Geometry and inductance-model constants for one spiral shape."""
  sides:       int                    # number of straight segments per turn
  inductance:  InductanceCoefficients

SQUARE = ShapeParams(
  sides=4,
  inductance=InductanceCoefficients(
    beta=1.62e-3, a1=-1.21, a2=-0.147, a3=2.4, a4=1.78, a5=-0.030
  ),
)
HEXAGONAL = ShapeParams(
  sides=6,
  inductance=InductanceCoefficients(
    beta=1.28e-3, a1=-1.24, a2=-0.174, a3=2.47, a4=1.77, a5=-0.049
  ),
)
OCTAGONAL = ShapeParams(
  sides=8,
  inductance=InductanceCoefficients(
    beta=1.33e-3, a1=-1.21, a2=-0.163, a3=2.43, a4=1.75, a5=-0.049
  ),
)

# Single source of truth for "what shapes exist" -- add a shape by adding one
# entry here; nothing else needs to change to make hershenson.py accept it.
SHAPES: dict[str, ShapeParams] = {
  "square":  SQUARE,
  "hexagon": HEXAGONAL,
  "octagon": OCTAGONAL,
}