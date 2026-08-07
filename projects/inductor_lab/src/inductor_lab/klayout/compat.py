"""Compatibility shim so PCell code runs both inside the KLayout GUI and
headless.

Inside the KLayout application, the layout API is exposed as `pya`. Outside
it (scripts, notebooks, the `klayout` pip package), the identical API is
exposed as `klayout.db`. Every module in this package imports `pya` from
here so it works in both contexts.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # klayout's `pya` compat package ships no type stubs, but `klayout.db`
    # (same API) does -- type-check against that one.
    import klayout.db as pya
else:
    try:
        import pya
    except ImportError:
        import klayout.db as pya

__all__ = ["pya"]
