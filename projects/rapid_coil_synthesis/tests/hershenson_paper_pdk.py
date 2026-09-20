"""
Fake PDK matching Hershenson's published GP-inductor paper process constants.

Only t_ox, t_m, sigma_m, w_min and s_min come from the paper. eps_r_ox is a
generic SiO2 value (not stated in the paper) -- it still feeds C_s via k_3
regardless of pgs. eps_r_sub/sigma_sub/t_sub are inert placeholders: with
pgs=True, InductanceTargetMaxQ overrides R_p/C_p directly and never touches
the substrate branch. The coil-to-underpass via/oxide gap (top_vias[0])
is NOT given in the paper either -- 0.85um was calibrated by sweeping
InductanceTargetMaxQ.t_ox_tm1tm2 against Table 1's L1/L2 (n, w, d_out).
"""

from rapid_coil_synthesis.pdk.base import MetalLayer, ProcessParams, ViaLayer

_T_OX = 5.2e-6           # oxide thickness, substrate to top metal [m]
_T_OX_TM1TM2 = 0.85e-6   # coil-to-underpass oxide gap [m] -- calibrated, see above
_T_M = 0.9e-6            # top metal thickness [m]
_SIGMA_M = 3e5 * 100     # 3e5 (ohm*cm)^-1 -> S/m

HERSHENSON_PAPER = ProcessParams(
    name="hershenson_paper",
    top_metals=(
        MetalLayer(
            name="M_top",
            sigma=_SIGMA_M,
            thickness=_T_M,
            w_min=1.9e-6,
            s_min=1.9e-6,
        ),
    ),
    top_vias=(
        ViaLayer(
            name="underpass_via",
            sigma=_SIGMA_M,
            thickness=_T_OX_TM1TM2,
            width=1.9e-6,
            space=1.9e-6,
            enc_lower=0.0,
            enc_upper=0.0,
        ),
    ),
    eps_r_ox=3.9,     # generic SiO2, not stated in the paper
    t_ox=_T_OX,
    eps_r_sub=11.9,   # inert placeholder under pgs=True
    sigma_sub=2.0,    # inert placeholder under pgs=True
    t_sub=200e-6,     # inert placeholder under pgs=True
)
