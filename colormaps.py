"""Re-export shim — TAT colormaps now live in the shared ``tat_palettes`` package.

The Triple-A-Tropics color palettes (rainbow_ir, dvorak, tat_neon, wv_tat,
ir_gray and the enhancement helpers) were extracted VERBATIM into the canonical,
pip-installable ``tat_palettes`` package, which lives in the main Pages repo
under ``palette/``. It is the single source of truth, imported by BOTH this
satellite backend and the main repo's HAFS model pipeline, so a color edit
happens in ONE place and propagates to every consumer on the next deploy.

This module stays only as a thin re-export so existing call sites keep working
unchanged — both ``from colormaps import get_enhancement`` and
``import colormaps; colormaps.ENHANCEMENTS`` resolve to the package symbols.

DO NOT edit palettes here. Edit ``palette/tat_palettes/__init__.py`` in the main
Triple-A-Tropics repo and run ``python -m tat_palettes`` (zero-drift self-test).
See that repo's ``palette/README.md``.
"""

from tat_palettes import *  # noqa: F401,F403  (re-export the whole public API)

# Explicit re-exports so static analysis and ``colormaps.X`` access stay obvious.
from tat_palettes import (  # noqa: F401
    ENHANCEMENTS,
    DEFAULT_ENHANCEMENT,
    get_enhancement,
    enhancement_norm,
    list_enhancements_for_domain,
    normalize_visible,
    RAINBOW_IR_CMAP,
    DVORAK_BD_CMAP,
    TAT_NEON_CMAP,
    WV_TAT_CMAP,
    IR_GRAY_CMAP,
)
