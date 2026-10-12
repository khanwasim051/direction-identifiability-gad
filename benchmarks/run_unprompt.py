"""Official UNPrompt (mala-lab/UNPrompt, train.py) run on CPU: trained on Facebook, zero-shot on targets.
Changes vs official: device=cpu; DGL stubbed (used only for seeding); reduced SVD (identical leading
components, needed for memory on large graphs); 5 seeds; extended target list. Hyperparameters = run.sh."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA, WORK, PKG, BENCH, STUDY, PAPER, RES_BENCH, RES_STUDY, RES_PAPER  # noqa: E402
import sys, types, os, random, csv, time
sys.modules["dgl"] = types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda s: None))
REPO = str(DATA / "mala-lab_UNPrompt")
sys.path.insert(0, REPO)
os.chdir(WORK / "unp_work")
import numpy as np, torch, torch.nn as nn
import utils as U
def x_svd(data, out_dim):
    Uu, S, _ = torch.linalg.svd(data, full_matrices=False)
    return torch.mm(Uu[:, :out_dim], torch.diag(S[:out_dim]))
U.x_svd = x_svd
from model import Model, SimplePrompt, GPFplusAtt, Projection
import pretrain
from pretrain import traingrace
from utils import load_mat, normalize_adj, sparse_mx_to_torch_sparse_tensor, completionloss, completionsim, evaluate
import scipy.sparse as sp
device = torch.device("cpu")
class A: pass
args = A(); args.lr=1e-3; args.weight_decay=0.0; args.embedding_dim=128; args.edge_drop_prob=0.2
args.feat_drop_prob=0.3; args.lamda=0.2; args.epochs=900; args.unifeat=8; args.numprompts=10

def loaddata(dataset):
    adj, features, ano_label, _, _ = load_mat(dataset)
    adj = adj.astype(np.float32)
    features = torch.FloatTensor(np.asarray(features.todense()))
    features = x_svd(features, args.unifeat)
    features = nn.BatchNorm1d(features.shape[1], affine=False)(features)
    if (adj.diagonal() > 0).all():
        won = adj; woself = adj - sp.eye(adj.shape[0])
    else:
        won = adj + sp.eye(adj.shape[0]); woself = adj
    withloop = sparse_mx_to_torch_sparse_tensor(normalize_adj(won))
    woself = sparse_mx_to_torch_sparse_tensor(normalize_adj(woself))
    return won, withloop, woself, features.detach(), torch.FloatTensor(ano_label)

targets = sys.argv[1].split(",")
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [11, 23, 37, 53, 71]
rows = []
for seed in seeds:
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    t0 = time.perf_counter()
    won, wl, wo, feat, lab = loaddata("Facebook")
    model = Model(args.unifeat, args.embedding_dim, "prelu")
    traingrace(model, [won], [wl], [feat], args, device)
    model.eval()
    prompts = GPFplusAtt(args.unifeat, args.numprompts); proj = Projection(args.embedding_dim)
    opt = torch.optim.Adam(list(prompts.parameters()) + list(proj.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        prompts.train(); proj.train(); opt.zero_grad()
        mf = prompts.add(feat)
        loss = completionloss(proj(model(mf, wo)), proj(model(mf, None)), lab)
        loss.backward(); opt.step()
    train_s = time.perf_counter() - t0
    prompts.eval(); proj.eval()
    for ds in targets:
        t1 = time.perf_counter()
        with torch.no_grad():
            _, _, wo_t, f_t, lab_t = loaddata(ds)
            mf = prompts.add(f_t)
            score = completionsim(proj(model(mf, None)), proj(model(mf, wo_t)))
        auc, ap = evaluate(score.numpy() if torch.is_tensor(score) else score, lab_t.numpy())
        rows.append(dict(method="UNPrompt", dataset=ds, seed=seed, auroc=float(auc), ap=float(ap), train_seconds=train_s, infer_seconds=time.perf_counter() - t1))
        print(f"seed={seed} Facebook->{ds} AUROC={auc:.4f} AP={ap:.4f}", flush=True)
out = f"{RES_BENCH}/unp_out_{sys.argv[3] if len(sys.argv) > 3 else 'main'}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
