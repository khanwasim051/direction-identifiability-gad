"""Run the official AnomalyGFM zero-shot script (zero_shot/run_abnormal.py), unmodified, trained on Facebook.
Only the unused imports dgl and torch_sparse are stubbed (benchmarks/stubs). Logs go to results/benchmarks/anomalygfm_logs;
we report the abnormal-prototype score at the final epoch (300), fixed before running.
Usage: python benchmarks/run_anomalygfm.py Disney Amazon Reddit Weibo Tolokers Amazon-all"""
import os, subprocess, sys
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, BENCH, RES_BENCH  # noqa: E402
script = DATA / "mala-lab_AnomalyGFM" / "zero_shot" / "run_abnormal.py"
logs = RES_BENCH / "anomalygfm_logs"; logs.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(BENCH / "stubs"), str(script.parent)]), OMP_NUM_THREADS="1")
for target in sys.argv[1:]:
    for seed in (11, 23, 37, 53, 71):
        with open(logs / f"{target}_{seed}.log", "w") as fh:
            subprocess.run([sys.executable, "-W", "ignore", str(script), "--dataset_train", "Facebook_svd",
                            "--dataset_test", f"{target}_svd", "--seed", str(seed)], cwd=WORK / "agfm_work", env=env, stdout=fh, stderr=subprocess.STDOUT)
        print("done", target, seed, flush=True)
