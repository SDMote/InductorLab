"""
GlobalFoundries GF180MCU process parameters (5LM + TM11kA configuration).

Frozen dataclasses with all technology constants needed to build the inductor
equivalent-circuit model and to formulate GP dimensional constraints (DRC rules).

Target stack: 5LM + TM11kA (IEEE SSCS Chipathon 2025/2026 configuration).
In GF180MCU 5LM, Metal5 IS the MetalTop — there is no separate thick-metal
layer.  The stack is M1–M4 (thin, 0.55 µm) + Metal5/TM (11kA, 1.1925 µm).
  top_metals[0]  = Metal5 / MetalTop (TM11kA)  — primary inductor conductor
  top_metals[1]  = Metal4               — cross-under conductor
  top_vias[0]    = Via4                 — Metal5↔Metal4

MIM option B (between Metal4 and Metal5) is available but the oxide thickness
used here (IMD4 = 1.15 µm) is for the non-MIM region; the inductor spiral
does not require a MIM capacitor beneath it.

New processes add a sibling file that provides the same field names so that
GP problem code never needs to change.

Sources:
  - GF180MCU DRM (Physical Verification Design Manual), tables in
    docs/physical_verification/design_manual/tables_clear/
  - GF180MCU Interconnect Application Note (IA Specs):
    docs/analog/layout/inter_specs/tables_clear/ and
    docs/analog/layout/inter_specs/images/2_cross_section_38.png
    (section 3.38: 1P5M TM11kA without MIM)
  - GF180MCU Electrical Specifications:
    docs/analog/spice/elec_specs/tables_clear/
All values in SI units (metres, S/m, ...).
"""

from dataclasses import dataclass

from inductor_lab.pdk.base import TODO, MetalLayer, ViaLayer


@dataclass(frozen=True)
class GF180MCUParams:
    """Technology constants for GF180MCU 0.18 µm HV CMOS (5LM+TM11kA).

    top_metals and top_vias are ordered top-to-bottom so that index 0 is
    always the topmost layer (i.e. the preferred inductor conductor).
    """

    # Ordered top-to-bottom: (Metal5/MetalTop TM11kA, Metal4)
    top_metals: tuple[MetalLayer, ...]

    # Ordered top-to-bottom: (Via4 between Metal5/Metal4)
    top_vias: tuple[ViaLayer, ...]

    # ── Oxide dielectric (BEOL stack) ────────────────────────────────────────
    # IA specs (4_General_Dielectric): all IMD layers have ε_r = 4.0
    # t_ox = oxide from Si surface to bottom of Metal5/TM (cross-section 38):
    #   ILD (0.86) + 4 × via_height (4 × 0.60) = 3.26 µm
    #   via_height = IMD_thickness − M_thickness = 1.15 − 0.55 = 0.60 µm
    eps_r_ox: float  # relative permittivity (dimensionless)
    t_ox: float      # total oxide thickness, Si surface to TM bottom [m]

    # ── Substrate ─────────────────────────────────────────────────────────────
    # NONE of these are stated in the GF180MCU open docs. eps_r_sub is the standard
    # silicon value; sigma_sub and t_sub are ASSUMPTIONS (see the instantiation below).
    eps_r_sub: float  # relative permittivity (dimensionless)
    sigma_sub: float  # conductivity [S/m]
    t_sub: float      # thickness [m]


GF180MCU = GF180MCUParams(
    top_metals=(
        MetalLayer(
            # Elec specs (5_General_Specification1): Rs_TM11 typ = 40 mΩ/sq
            # Cross-section 38: t_TM11 = 1.1925 µm
            #   (stack: 375Å IMPTi + 300Å TiN + 11000Å Al + 250Å TiN)
            # σ = 1 / (40e-3 Ω/sq × 1.1925e-6 m)
            name="MetalTop_TM11",
            sigma=21.0e6,
            thickness=1.1925e-6,
            # DRM §24 (MetalTop): MT.1, MT.2a for 9kA/11kA options
            w_min=0.44e-6,
            s_min=0.46e-6,
        ),
        MetalLayer(
            # Elec specs (5_General_Specification1): Rs_M2-5 typ = 90 mΩ/sq
            # IA specs (3_General_conductor): t_M4 = 0.55 µm (typical)
            # σ = 1 / (90e-3 Ω/sq × 0.55e-6 m)
            name="Metal4",
            sigma=20.2e6,
            thickness=0.55e-6,
            # DRM §22 (Metaln): Mn.1, Mn.2a for n = 2-5
            w_min=0.28e-6,
            s_min=0.28e-6,
        ),
    ),
    top_vias=(
        ViaLayer(
            # Elec specs (5_General_Specification2): R_Via4 typ = 4.5 Ω/via
            # DRM §23 (Vian): Vn.1 width = 0.26 µm (fixed min & max)
            # Via height from cross-section 38:
            #   IMD4(1.15 µm) − M4_thickness(0.55 µm) = 0.60 µm
            # σ = via_height / (R_via × width²) = 0.60e-6 / (4.5 × 0.26e-6²)
            name="Via4",
            sigma=1.97e6,
            thickness=0.60e-6,
            width=0.26e-6,
            space=0.26e-6,
            enc_lower=0.01e-6,  # Vn.3: min Metal4 overlap of Via4
            enc_upper=0.01e-6,  # Vn.4: min Metal5 overlap of Via4
        ),
    ),
    eps_r_ox=4.0,
    t_ox=3.26e-6,   # ILD(0.86) + 4 × 0.60 µm  (cross-section 38)

    eps_r_sub=11.9,   # standard silicon; not stated in GF180MCU docs
    # ASSUMPTIONS -- NOT in any GF180MCU open doc (confirmed absent from the PDK docs,
    # SPICE/LVS decks, and the SSCS chipathon resources). Confirm with GF/wafer.space
    # before trusting L/SRF. Standard bulk mixed-signal CMOS p-type start material is
    # ~8-15 ohm.cm; 10 ohm.cm -> sigma = 1/(10e-2 ohm.m) = 10 S/m. This puts the substrate
    # dielectric-relaxation corner f_tau = sigma/(2*pi*eps_r*eps_0) ~ 15 GHz (vs SG13G2's
    # 3 GHz), so at 2.4-5 GHz designs sit BELOW the crossover -> use the frequency-dependent
    # Cp (fringing.Cp_yue / fit_cp_at_freq), NOT the high-frequency series-limit fit.
    sigma_sub=10.0,   # ASSUMED ~10 ohm.cm bulk p-substrate (see note above)
    t_sub=725e-6,     # ASSUMED un-thinned 200 mm wafer; set to the back-lapped value if thinned
)
