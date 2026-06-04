"""
Geometric program for an inductor inside an LC resonator tank.

Exposes build_model(pdk, ...) -> gpkit.Model.
Minimizes 1/R_tank (maximizes tank resistance) subject to a resonance frequency
constraint, minimum tank Q, series resistance bound, and PDK dimensional
constraints.  An additional capacitance C_ad is accepted as a parameter.
"""
