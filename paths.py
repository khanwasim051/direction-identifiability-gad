"""Repository-relative paths. Override the data and work folders with GAD_DATA / GAD_WORK."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GAD_DATA", REPO / "data"))          # downloaded benchmark files
WORK = Path(os.environ.get("GAD_WORK", REPO / "work"))          # caches and baseline working copies
PKG = REPO / "coregad_msda"                                      # original CoRe-GAD-MSDA release
BENCH = REPO / "benchmarks"
STUDY = REPO / "direction_study"
PAPER = REPO / "paper"
RES = REPO / "results"
RES_BENCH, RES_STUDY, RES_PAPER = RES / "benchmarks", RES / "direction_study", RES / "paper"
for _p in (WORK, RES_BENCH, RES_STUDY, RES_PAPER):
    _p.mkdir(parents=True, exist_ok=True)
