"""
IHP SG13G2 process parameters.

Values ported from inductor_lab.pdk.sg13g2 -- see that module for the
original process-spec / layout-rule citations behind each number.
"""

from rapid_coil_synthesis.pdk.base import MetalLayer, ProcessParams, ViaLayer

# Effective substrate thickness for each IHP back-lapping option [m].
# The nominal thickness (dict key, in µm) is the total wafer thickness after
# lapping; t_sub is ~20 µm less, accounting for the BEOL stack height above
# the silicon surface.
# Usage: from dataclasses import replace
#        pdk = replace(SG13G2, t_sub=BACKLAPPING[200])
BACKLAPPING: dict[int, float] = {
     75:  55e-6,
    100:  80e-6,
    150: 130e-6,
    200: 180e-6,
    250: 230e-6,
    300: 280e-6,
}

SG13G2 = ProcessParams(
    name="sg13g2",
    top_metals=(
        MetalLayer(
            # Process spec §2.13 / §2.16: RSTM2 = 11 mΩ/sq, TTM2 = 3000 nm
            name="TopMetal2",
            sigma=30.3e6,
            thickness=3.0e-6,
            w_min=2.0e-6,
            s_min=2.0e-6,
        ),
        MetalLayer(
            # Process spec §2.13 / §2.16: RSTM1 = 18 mΩ/sq, TTM1 = 2000 nm
            name="TopMetal1",
            sigma=27.8e6,
            thickness=2.0e-6,
            w_min=1.64e-6,
            s_min=1.64e-6,
        ),
    ),
    top_vias=(
        ViaLayer(
            # Process spec §2.14: RTV2 = 1.1 Ω/via, TILTM2 = 2800 nm
            name="TopVia2",
            sigma=3.143e6,
            thickness=2.8e-6,
            width=0.90e-6,
            space=1.06e-6,
            enc_lower=0.50e-6,
            enc_upper=0.50e-6,
        ),
        ViaLayer(
            # Process spec §2.14: RTV1 = 2.2 Ω/via, TILDTM1 = 850 nm
            name="TopVia1",
            sigma=2.19e6,
            thickness=850e-9,
            width=0.42e-6,
            space=0.42e-6,
            enc_lower=0.10e-6,
            enc_upper=0.42e-6,
        ),
    ),
    eps_r_ox=4.1,
    t_ox=11.2303e-6,
    eps_r_sub=11.9,
    sigma_sub=2.0,  # 1 / (50e-2 Ω·m)
    t_sub=BACKLAPPING[200],
)
