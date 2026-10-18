"""Appendix C table: benchmark file, source repository, commit, SHA-256 prefix, and injection structure."""
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from paths import REPO, DATA  # noqa: E402
import hashlib
SRC = {"yixinliu233_ARC": ("ARC", "f8be54f"), "mala-lab_UNPrompt": ("UNPrompt", "df1ddff"), "raw/unprompt": ("UNPrompt", "df1ddff"),
       "raw/Amazon": ("CARE-GNN", "a64ff75"), "raw/YelpChi": ("CARE-GNN", "a64ff75"), "pygod": ("PyGOD", "77976df")}
FILES = [("Cora", "yixinliu233_ARC/dataset/cora.mat", "5$\\times$15 / 75"), ("CiteSeer", "yixinliu233_ARC/dataset/citeseer.mat", "5$\\times$15 / 75"),
         ("PubMed", "yixinliu233_ARC/dataset/pubmed.mat", "20$\\times$15 / 300"), ("ACM", "yixinliu233_ARC/dataset/ACM.mat", "20$\\times$15 / --"),
         ("BlogCatalog", "yixinliu233_ARC/dataset/BlogCatalog.mat", "10$\\times$15 / 150"), ("Flickr", "yixinliu233_ARC/dataset/Flickr.mat", "15$\\times$15 / 225"),
         ("Facebook", "yixinliu233_ARC/dataset/Facebook.mat", "--$^{b}$"), ("Reddit", "yixinliu233_ARC/dataset/Reddit.mat", "--"),
         ("Amazon", "yixinliu233_ARC/dataset/Amazon.mat", "--"), ("Disney", "mala-lab_UNPrompt/Datasets/Disney.mat", "--"),
         ("Tolokers", "yixinliu233_ARC/dataset/tolokers.mat", "--"), ("Amazon-all", "raw/Amazon.mat", "--"), ("YelpChi-all", "raw/YelpChi.mat", "--"),
         ("Questions", "yixinliu233_ARC/dataset/questions.mat", "--"), ("YelpChi", "yixinliu233_ARC/dataset/YelpChi.mat", "--"),
         ("Weibo", "yixinliu233_ARC/dataset/weibo.mat", "--"), ("Coauthor-CS", "yixinliu233_ARC/dataset/cs.mat", "20$\\times$15 / 300"),
         ("Amazon-Photo", "yixinliu233_ARC/dataset/photo.mat", "15$\\times$15 / 225"), ("Books", "pygod/books.pt", "--"), ("Enron", "pygod/enron.pt", "--"),
         ("Weibo (UNPrompt)$^{a}$", "raw/unprompt_weibo/weibo.mat", "--")]
rows = []
for g, rel, inj in FILES:
    src = next(v for k, v in SRC.items() if rel.startswith(k))
    h = hashlib.sha256((DATA / rel).read_bytes()).hexdigest()[:12]
    name = rel.split("/")[-1].replace("_", "\\_")
    rows.append(f"{g} & \\texttt{{{name}}} & {src[0]} & \\texttt{{{src[1]}}} & \\texttt{{{h}}} & {inj} \\\\")
open(REPO / "paper" / "latex" / "gen" / "tab_files.tex", "w").write("\n".join(rows))
