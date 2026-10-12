"""Download every public benchmark file used in the study into this folder (or $GAD_DATA).

Requires git. Datasets are not redistributed in this repository; they are fetched from the
original public repositories listed below. Run from the repository root:
    python data/download_data.py
"""
import shutil, subprocess, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, PKG  # noqa: E402

REPOS = {
    "yixinliu233_ARC": "https://github.com/yixinliu233/ARC",            # most .mat benchmark files
    "mala-lab_UNPrompt": "https://github.com/mala-lab/UNPrompt",        # Disney, Weibo (UNPrompt file), UNPrompt code
    "mala-lab_AnomalyGFM": "https://github.com/mala-lab/AnomalyGFM",    # AnomalyGFM code
    "YingtongDou_CARE-GNN": "https://github.com/YingtongDou/CARE-GNN",  # Amazon.mat, YelpChi.mat (Amazon-all, YelpChi-all)
    "pygod-data": "https://github.com/pygod-team/data",                 # Books, Enron, Reddit (.pt)
}

def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        if not (DATA / name).exists():
            run(["git", "clone", "--depth", "1", url, name], cwd=DATA)
    raw = DATA / "raw"; raw.mkdir(exist_ok=True)
    for z in ("Amazon.zip", "YelpChi.zip"):
        zipfile.ZipFile(DATA / "YingtongDou_CARE-GNN" / "data" / z).extractall(raw)
    zipfile.ZipFile(DATA / "mala-lab_UNPrompt" / "Datasets" / "weibo.mat.zip").extractall(raw / "unprompt_weibo")
    pg = DATA / "pygod"; pg.mkdir(exist_ok=True)
    for z in ("books.pt.zip", "enron.pt.zip"):
        zipfile.ZipFile(DATA / "pygod-data" / z).extractall(pg)
    # data expected by the original CoRe-GAD-MSDA release (coregad_msda/)
    am = PKG / "condcal_amazon" / "data"; (am / "amazon_raw").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "pygod-data" / "reddit.pt.zip", am / "reddit.pt.zip")
    zipfile.ZipFile(am / "reddit.pt.zip").extract("reddit.pt", am)
    shutil.copy(raw / "Amazon.mat", am / "amazon_raw" / "Amazon.mat")
    run([sys.executable, str(PKG / "condcal_cora" / "download_data.py")])
    print("All datasets are in", DATA)
    print("Facebook files for coregad_msda/facebook_evaluation: copy", DATA / "yixinliu233_ARC/dataset/Facebook.mat",
          "to coregad_msda/facebook_evaluation/data/Facebook.mat (see coregad_msda/DATASETS.md for the ConAD file).")

if __name__ == "__main__":
    main()
