# SG13G2 SYMMETRIC octagonal inductance coefficients (sympoly, built-geometry fit)
# L_nH = BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5  (microns, nH)
# Fit: RMSE=5.37%  N=12581

BETA = 8.613900e-04
A1   = -1.217735
A2   = -0.141603
A3   = 2.482231
A4   = 1.864156
A5   = -0.038759


def inductance_nH(d_out, w, s, n):
    d_avg = d_out - n * (w + s)
    return BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5
