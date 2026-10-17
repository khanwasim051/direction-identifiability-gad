r"""Inlines every \input{...} of main_src.tex into a single main.tex (journals require one file; \input inside
tabular environments is also fragile)."""
import re, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex"))
def inline(path):
    txt = open(path).read()
    def rep(m):
        f = m.group(1); f = f if f.endswith(".tex") else f + ".tex"
        return inline(f).rstrip("\n")
    return re.sub(r"\\input\{([^}]+)\}", rep, txt)
open("main.tex", "w").write(inline("main_src.tex"))
print("main.tex written")
