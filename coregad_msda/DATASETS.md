# Dataset acquisition and integrity

Raw datasets are excluded from the release archive. The following files reproduce the exact graph versions used in the experiments.

## Automatically downloaded datasets

Run:

```bash
python condcal_cora/download_data.py
python condcal_amazon/prepare_data.py
```

The citation loader downloads the `allx`, `tx`, `graph`, and `test.index` Planetoid files for Cora, CiteSeer, and PubMed from the original Planetoid repository. Document-class labels are not downloaded or used for anomaly scoring.

| Dataset | Release file | Source | SHA-256 |
|---|---|---|---|
| Reddit | `condcal_amazon/data/reddit.pt.zip` | PyGOD data repository | `276432d36219349b7ad98e974649f73c94e124e11f5282f53b1ec41086c1e18a` |
| Amazon Fraud | `condcal_amazon/data/FraudAmazon.zip` | DGL dataset server | `c1b7a1f8cd8621467d5121b690fef61a7d398e4f582232bb3e8a154424f7a5b9` |

The Amazon experiment uses only the U-P-U relation and evaluates nodes 3305–11943, matching the conventional CARE-GNN/PyGOD mask.

## Facebook artifacts

Retrieve the matrices from the official TAM and ConAD repositories and place them in `facebook_evaluation/data/`.

| Role | Required filename | Expected nodes/anomalies | SHA-256 |
|---|---|---:|---|
| Primary TAM-distributed artifact | `Facebook.mat` | 1,081/25 | `606d8cce2f9a9d2829244f5691d1abf0c3a5e0f28a2ade25a7ca7df304184163` |
| ConAD version-sensitivity artifact | `fb_conad_full.mat` | 1,086/27 | `d7041572dc1d08701984460d2d7498f1b436ab5b3207bb8d758aaf7b4f404904` |

The primary artifact contains 27,552 undirected edges and 576 features. Its observed 25 anomalies differ from the 27 anomalies stated in the TAM README; the manuscript and package report the contents of the actual file.

## AnomalyGFM-aligned files

The optional AnomalyGFM reproduction requires `Facebook_svd.mat` and `reddit_svd.mat` from the official AnomalyGFM repository. Place both files in `external_data/` or set `COREGAD_EXTERNAL_DATA` to their directory. AnomalyGFM uses Facebook source labels during source training; target labels remain evaluation-only.

## Excluded dataset

Weibo was evaluated during an earlier diagnostic stage but was excluded before the final manuscript comparison. The public reproduction protocol, manuscript tables, score export, and final claims do not include Weibo.
