"""Supplementary Figure S6 -- Mouse hypothalamus Sankey plots.

Generate one Sankey plot for each of the five comparison methods, mapping the
46 curated cell types to each method's inferred clusters. The rendering code is
shared with Figure 4d so cluster extraction and colours stay consistent.

    python figS6.py --data-dir /path/to/mouse_h --out-dir /path/to/output
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig4d_sankey import main as render_sankeys


COMPARISON_METHODS = "scvi,scgnn,adclust,scace,scdac"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--methods" or arg.startswith("--methods=") for arg in argv):
        argv.extend(["--methods", COMPARISON_METHODS])
    return render_sankeys(argv)


if __name__ == "__main__":
    raise SystemExit(main())
