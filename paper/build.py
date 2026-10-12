import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import re
s = open(PAPER / "latex" / "main_src.tex").read()
s = re.sub(r"\\input\{(gen/[^}]+)\}", lambda m: open(PAPER / "latex" / m.group(1)).read().strip(), s)
open(PAPER / "latex" / "main.tex", "w").write(s)
