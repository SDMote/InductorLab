# SG13G2 SYMMETRIC octagonal inductance coefficients (sympoly, built-geometry fit)
# L_nH = BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5  (microns, nH)
# Fit: RMSE=2.10%  N=1191

BETA = 1.202073e-03
A1   = -1.023469
A2   = -0.130204
A3   = 2.226724
A4   = 1.765854
A5   = -0.030117


def inductance_nH(d_out, w, s, n):
    d_avg = d_out - n * (w + s)
    return BETA * d_out**A1 * w**A2 * d_avg**A3 * n**A4 * s**A5
