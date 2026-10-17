"""CONFIRMATORY_PROTOCOL.md: frozen FDC (W1) and FDC-R (V0) on the held-out graphs. Output: results/heldout/final_T2.csv
Refuses to run unless every file recorded in protocols/METHOD_FREEZE.json is unchanged: the frozen method files are
kept verbatim in protocols/frozen/; fdc/fsdc.py, fdc/step2.py and fdc/step3.py differ from them only in file paths
(check with: diff protocols/frozen/step3.py fdc/step3.py)."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import PKG, FDC, SEALED, REPO, RES_HO  # noqa: E402
import sys, os, csv, json, hashlib, numpy as np

PROT = REPO / "protocols"
FREEZE = json.load(open(PROT / "METHOD_FREEZE.json"))
for f, h in FREEZE["sha256"].items():
    name = os.path.basename(f)
    if name == "run_experiment.py":
        p = PKG / "core_multiscale_directional" / name
    elif name.endswith(".md"):
        p = PROT / name
    else:
        p = PROT / "frozen" / name
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, f"frozen file changed: {p}"
print("freeze record verified:", len(FREEZE["sha256"]), "files unchanged")

sys.path.insert(0, str(FDC))
import fsdc as F, step2 as S2, step3 as S3

OUT = str(RES_HO)
T2 = ["CS", "Photo", "Books", "Enron"]

rows = []
for name in (sys.argv[1:] or T2):
    ga, y = S2.attribute_stats(name)
    assert (np.load(f"{SEALED}/labels_{name}.npy").astype(int) == y).all()
    for seed in F.SEEDS:
        z = np.load(f"{F.CACHE}/{name}__{seed}.npz")
        for k in F.cells(y):
            sh = F.shots_for(y, seed, k)
            sc = S3.final_scores(name, z["stats"], z["ranks"], z["mask"], z["counts"], ga, sh, k)
            for meth, key in (("FINAL_W1", "W1_kgated"), ("V0_M2", "V0_M2")):
                au, ap = F.metrics(y, sc[key], sh)
                rows.append(dict(dataset=name, seed=seed, k=k, method=meth, auroc=au, ap=ap))
    print("done", name, flush=True)
with open(f"{OUT}/final_T2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
