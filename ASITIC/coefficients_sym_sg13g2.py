# SG13G2 SYMMETRIC octagonal inductance coefficients -- TRUE BACKSIDE-GROUND fixture.
# L_nH = BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5   (microns, nH)
#
# The five EXPONENTS are Mohan/Hershenson's published, field-solver-validated values
# ("Simple accurate expressions for planar spiral inductances", 19000-coil fit; see the
# Hershenson GP paper eq. 2). They fit our Palace diffpec (PEC-ground backside, 1-port
# differential Z_diff at 0.1-0.3 GHz) LHS to 7.8% RMS with FLAT residuals across turn count
# -- refitting all five buys only 0.5% and risks overfitting ~90 noisy points.
#
# Only BETA is re-fit here: the 180 um backside ground images the coil and knocks L down by a
# uniform ~27% (bias flat across n/w/s/d_out), so Mohan's BETA=1.66e-3 -> 1.222e-3. That single
# geometry-independent scale is ALL the backside fixture changes. Fit: RMS 7.8%, N=90 Palace coils
# (scripts/.out/lhs_diffpec). Supersedes the _2.4ghz (ASITIC), _palace (d_avg-only) and _rapidfem
# (d_avg-only) fits -- all discarded. Use with spiral_cturn.design (full d_out form; a1!=0).

BETA = 1.222000e-03
A1   = -1.330000        # d_out
A2   = -0.125000        # w
A3   = 2.500000         # d_avg
A4   = 1.830000         # n
A5   = -0.022000        # s


def inductance_nH(d_out, w, s, n):
    d_avg = d_out - n * w - (n - 1) * s   # n traces, (n-1) gaps per side
    return BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5
