"""Registers the InductorLab PCells with KLayout.

One InductorPCell instance is registered per entry in tech._TECH_LAYERS, each
in its own pya.Library locked to that technology (Library.technology) --
KLayout only offers a library's PCells to layouts using a matching
technology, so the Inductor PCell that shows up in the GUI (or that
layout.create_cell resolves) always matches whichever PDK is selected.

    layout.create_cell("Inductor", "InductorLab_sg13g2", {"target_l_nh": 5.0})
"""

from .compat import pya
from .inductor_pcell import InductorPCell
from .tech import supported_technologies, tech_layers_for


class InductorLabLibrary(pya.Library):
    # pya.Library.__init__ just allocates an empty library; the vendored PDK's
    # own native library classes (e.g. SG13G2_ViaLib) skip calling it too.
    def __init__(self, tech_name: str) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.description: str = f"InductorLab GP-synthesized spiral inductors ({tech_name})"
        self.layout().register_pcell("Inductor", InductorPCell(tech_layers_for(tech_name)))
        self.register(f"InductorLab_{tech_name}")
        self.technology: str = tech_name


for _tech_name in supported_technologies():
    InductorLabLibrary(_tech_name)
