"""
IHP SG13G2 process parameters.

Frozen dataclasses with all technology constants needed to build the inductor
equivalent-circuit model (conductivities, oxide/substrate thicknesses, via
geometry) and to formulate GP dimensional constraints (DRC rules).

New processes add a sibling file that provides the same field names so that
GP problem code never needs to change.

Sources:
  - IHP SG13G2 Open Source Process Specification Rev. 1.2 (2023-12-20)
  - IHP SG13G2 Open Source Layout Rules Rev. 0.4 (2024-12-19)
All values in SI units (metres, S/m, ...).
"""

from dataclasses import dataclass

from inductor_lab.pdk.base import TODO, MetalLayer, ViaLayer

# Effective substrate thickness for each IHP back-lapping option [m].
# The nominal thickness (dict key, in µm) is the total wafer thickness after
# lapping; t_sub is ~20 µm less, accounting for the BEOL stack height above
# the silicon surface.  Source: IHP MPW service options; 200 µm value
# confirmed against notebook (GeometricProgramming.ipynb).
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


@dataclass(frozen=True)
class SG13G2Params:
    """Technology constants for IHP SG13G2 0.13 µm BiCMOS.

    top_metals and top_vias are ordered top-to-bottom so that index 0 is
    always the topmost layer (i.e. the preferred inductor conductor).
    """

    # Ordered top-to-bottom: (TopMetal2, TopMetal1)
    top_metals: tuple[MetalLayer, ...]

    # Ordered top-to-bottom: (TopVia2 between TM2/TM1, TopVia1 between TM1/Metal5)
    top_vias: tuple[ViaLayer, ...]

    # ── Oxide dielectric (BEOL stack) ────────────────────────────────────────
    # Process spec §1.1: ε_R = 4.1 ± 0.1 for all BEOL oxide layers
    eps_r_ox: float  # relative permittivity (dimensionless)
    t_ox: float      # total oxide thickness, substrate surface to TM2 [m]

    # ── Substrate ─────────────────────────────────────────────────────────────
    # Process spec §1.1: ε_R = 11.9
    # Process spec §2.13: RSBLK = 50 Ω·cm (target) → σ = 1/(50e-2) = 2 S/m
    # t_sub depends on the back-lapping option ordered from IHP; use BACKLAPPING
    # dict and dataclasses.replace() to set it:
    #   from dataclasses import replace
    #   pdk = replace(SG13G2, t_sub=BACKLAPPING[200])
    eps_r_sub: float  # relative permittivity (dimensionless)
    sigma_sub: float  # conductivity [S/m]
    t_sub: float      # thickness [m] — must be set via BACKLAPPING


SG13G2 = SG13G2Params(
    top_metals=(
        MetalLayer(
            # Process spec §2.13 / §2.16: RSTM2 = 11 mΩ/sq, TTM2 = 3000 nm
            # σ = 1 / (11e-3 Ω/sq × 3e-6 m)
            name="TopMetal2",
            sigma=30.3e6,
            thickness=3.0e-6,
            # Layout rules §5.25: TM2.a, TM2.b
            w_min=2.0e-6,
            s_min=2.0e-6,
        ),
        MetalLayer(
            # Process spec §2.13 / §2.16: RSTM1 = 18 mΩ/sq, TTM1 = 2000 nm
            # σ = 1 / (18e-3 Ω/sq × 2e-6 m)
            name="TopMetal1",
            sigma=27.8e6,
            thickness=2.0e-6,
            # Layout rules §5.22: TM1.a, TM1.b
            w_min=1.64e-6,
            s_min=1.64e-6,
        ),
    ),
    top_vias=(
        ViaLayer(
            # Process spec §2.14: RTV2 = 1.1 Ω/via, TILTM2 = 2800 nm
            # Layout rules §5.24: TV2.a (width, fixed), TV2.b (space)
            # σ = TILTM2 / (RTV2 × width²) = 2.8e-6 / (1.1 × 0.9e-6²)
            name="TopVia2",
            sigma=3.143e6,
            thickness=2.8e-6,
            width=0.90e-6,
            space=1.06e-6,
            enc_lower=0.50e-6,  # TV2.c: min TopMetal1 enclosure
            enc_upper=0.50e-6,  # TV2.d: min TopMetal2 enclosure
        ),
        ViaLayer(
            # Process spec §2.14: RTV1 = 2.2 Ω/via, TILDTM1 = 850 nm
            # Layout rules §5.21: TV1.a (width, fixed), TV1.b (space)
            # σ = TILDTM1 / (RTV1 × width²) = 850e-9 / (2.2 × 0.42e-6²)
            name="TopVia1",
            sigma=2.19e6,
            thickness=850e-9,
            width=0.42e-6,
            space=0.42e-6,
            enc_lower=0.10e-6,  # TV1.c: min Metal5 enclosure
            enc_upper=0.42e-6,  # TV1.d: min TopMetal1 enclosure
        ),
    ),
    eps_r_ox=4.1,
    t_ox=11.2303e-6,
    eps_r_sub=11.9,
    sigma_sub=2.0,  # 1 / (50e-2 Ω·m)
    t_sub=BACKLAPPING[200],
)
