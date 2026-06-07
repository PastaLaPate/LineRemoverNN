"""
Thin wrapper around the compiled C++ extension.
Falls back gracefully if the extension wasn't built.
"""

import numpy as np

try:
    from lineremovernn._lineremovernn_ext import (  # noqa: F401
        remove_lines_opencv,
        render_page_cairo,
    )

    HAS_EXT = True
except ImportError:
    HAS_EXT = False

    def render_page_cairo(
        width: int, height: int, line_density: float = 0.1
    ) -> np.ndarray:  # type: ignore[misc]
        raise RuntimeError("C++ extension not built. Run: ./dev-install.sh")

    def remove_lines_opencv(img: np.ndarray) -> None:  # type: ignore[misc]
        raise RuntimeError("C++ extension not built. Run: ./dev-install.sh")
