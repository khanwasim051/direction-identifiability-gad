"""Clones the official baseline code at the exact commits used in the paper into WORK/official.
GDN:   https://github.com/kaize0409/Meta-GDN_AnomalyDetection          @ 426516b
BWGNN: https://github.com/squareRoot3/Rethinking-Anomaly-Detection     @ de0631f
(BWGNN is run with bwgnn_package/, which downloads and checks its model file itself; it is cloned here for reference.)"""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import OFFICIAL  # noqa: E402
import subprocess

REPOS = {"Meta-GDN_AnomalyDetection": ("https://github.com/kaize0409/Meta-GDN_AnomalyDetection", "426516bf4356ce9ff7077149584732c5dd8f7e60"),
         "Rethinking-Anomaly-Detection": ("https://github.com/squareRoot3/Rethinking-Anomaly-Detection", "de0631f039bbd19c1890b483cc01f1007f596af7")}

for name, (url, commit) in REPOS.items():
    dest = OFFICIAL / name
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    print(name, "at", subprocess.run(["git", "-C", str(dest), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
