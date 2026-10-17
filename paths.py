"""Repository-relative paths. Override the data and work folders with GAD_DATA / GAD_WORK."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GAD_DATA", REPO / "data"))          # downloaded benchmark files
WORK = Path(os.environ.get("GAD_WORK", REPO / "work"))          # statistics cache, labels, official baseline code
PKG = REPO / "coregad_msda"                                      # base detector (CoRe-GAD-MSDA release)
GADLIB = REPO / "gad"                                            # loaders and exact statistics
FDC = REPO / "fdc"                                               # method and experiment scripts
CACHE = WORK / "cache"                                           # per-graph statistics and development labels
SEALED = WORK / "sealed_labels"                                  # held-out labels, stored separately
OFFICIAL = WORK / "official"                                     # official GDN / BWGNN repositories
RES = REPO / "results"
RES_R1, RES_DEV, RES_R23, RES_HO, RES_TAB = (RES / "round1", RES / "official_dev", RES / "rounds23",
                                             RES / "heldout", RES / "tables")
for _p in (WORK, CACHE, SEALED, OFFICIAL, RES_R1, RES_DEV, RES_DEV / "bwgnn", RES_R23, RES_HO, RES_HO / "bwgnn", RES_TAB):
    _p.mkdir(parents=True, exist_ok=True)
