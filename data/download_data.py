"""Download every public benchmark file used in the study into this folder (or $GAD_DATA).

Requires git. Datasets are not redistributed in this repository; they are fetched from the
original public repositories listed below. Run from the repository root:
    python data/download_data.py
"""
import hashlib, shutil, subprocess, sys, zipfile
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

# Commits from which the files used in the paper were obtained (Appendix C)
COMMITS = {
    "yixinliu233_ARC": "f8be54f37710d8dc6449193b7fbc2260d433df34",
    "mala-lab_UNPrompt": "df1ddffac6262e5a9ebe4bb08909cc5c875a63db",
    "mala-lab_AnomalyGFM": "bf07e59af65eac521aeb6a19632f8fd7f99661bf",
    "YingtongDou_CARE-GNN": "a64ff7523e187a24251f7ca88435d2c9d8f7dcd9",
    "pygod-data": "77976df4129f5f753907beea4945bd3e21d0251f",
}
# SHA-256 of every benchmark file used in the paper (paths relative to the data folder)
SHA256 = {
    "mala-lab_UNPrompt/Datasets/Disney.mat": "468810c22dd3eb2a260bf0bfb522881cdca92d9e6e989db9b34537e623c1bdab",
    "pygod/books.pt": "b23320b9c0c654bc3ac88ca999f28fc52de101582c73ec3730dc29aeaf475972",
    "pygod/enron.pt": "1c1f7659b6d3e4c51c548b3ffaf7a6b3429f73d02bcd4410e2efc2ea42da3e0a",
    "raw/Amazon.mat": "4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63",
    "raw/YelpChi.mat": "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42",
    "raw/unprompt_weibo/weibo.mat": "63c1ffdfc3b40c05010f30dc3599e8322dc202b879382a5d6e3529f46b0744e7",
    "yixinliu233_ARC/dataset/ACM.mat": "31833bd23c8fa8d3aa3358876ebea43cf77d03fb4068f2a3b73b3d7dda2b6e98",
    "yixinliu233_ARC/dataset/Amazon.mat": "6677b4f17f594c4169df3717fc9da9c6b06d1dfcc28e5f0d42beedf4c7843dd3",
    "yixinliu233_ARC/dataset/BlogCatalog.mat": "65a532eee6903842d684772847f3771a4f16225841ca891af7f3f8b72e68ce5a",
    "yixinliu233_ARC/dataset/Facebook.mat": "606d8cce2f9a9d2829244f5691d1abf0c3a5e0f28a2ade25a7ca7df304184163",
    "yixinliu233_ARC/dataset/Flickr.mat": "350986debdbf452359a5edf510649188341de3d0bbaa17e3362e5643d256c88f",
    "yixinliu233_ARC/dataset/Reddit.mat": "533a739caa38939dff51f1902e6705fbb4e727badf1d408fc0d1b45467cf4fd3",
    "yixinliu233_ARC/dataset/YelpChi.mat": "61eae7fd894b54840eb7038d52bbf9e3802704ccd16a74526dd2fff3cbff72ca",
    "yixinliu233_ARC/dataset/citeseer.mat": "9a5dd61bb1840f0ec448b24bd9bfce44a5a922f5dc9060846221b36c02f41a2e",
    "yixinliu233_ARC/dataset/cora.mat": "1f67271a4355d88efbc11fb4d67419c2e83f8f0f8bc5c5703d4cedb8f2fcca37",
    "yixinliu233_ARC/dataset/cs.mat": "1d3eb800cc0d1c0fbbb3658cc2393693fca6172cf48cfb94115397b3f65e1989",
    "yixinliu233_ARC/dataset/photo.mat": "561a379de758693f6169e9b3263b9b491e181e18c4c3056a509d05700cda6408",
    "yixinliu233_ARC/dataset/pubmed.mat": "630b73bbe79261d13ec71cede9b71c124da5be39d48817f829fe6ae4102b51e3",
    "yixinliu233_ARC/dataset/questions.mat": "549173adf44d795b539e219296bb68895e416e84868ac1abc0f512a2c29a5e97",
    "yixinliu233_ARC/dataset/tolokers.mat": "6f9e98216758bbce3a438469007a9adc151e7f22256f54ae18e43f06d4e1a830",
    "yixinliu233_ARC/dataset/weibo.mat": "b79248b7ea8329fbfffc6b0c97e73e5e930a79cb5ffa6d0e529117b9bae9b5da",
}


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        if not (DATA / name).exists():
            run(["git", "init", "-q", name], cwd=DATA)
            run(["git", "remote", "add", "origin", url], cwd=DATA / name)
            run(["git", "fetch", "-q", "--depth", "1", "origin", COMMITS[name]], cwd=DATA / name)
            run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=DATA / name)
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
    bad = [f for f, h in SHA256.items() if hashlib.sha256((DATA / f).read_bytes()).hexdigest() != h]
    if bad:
        sys.exit("SHA-256 mismatch (file differs from the one used in the paper): " + ", ".join(bad))
    print("All", len(SHA256), "benchmark files match the SHA-256 hashes used in the paper.")
    print("All datasets are in", DATA)
    print("Facebook files for coregad_msda/facebook_evaluation: copy", DATA / "yixinliu233_ARC/dataset/Facebook.mat",
          "to coregad_msda/facebook_evaluation/data/Facebook.mat (see coregad_msda/DATASETS.md for the ConAD file).")

if __name__ == "__main__":
    main()
