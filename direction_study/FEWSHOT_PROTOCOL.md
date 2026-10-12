# Feasibility check (specified before running): few-shot direction selection
Written after the zero-shot rule study (freeze.json) and before any few-shot result was computed.
- For each graph and seed (standard graphs: 5 cached seeds; synthetic units: every run), draw k labeled
  anomalies and k labeled normal nodes uniformly at random (numpy default_rng(SeedSequence([seed, 9000+k]))),
  k in {5, 10}. Graphs with fewer than k+1 anomalies are skipped for that k.
- For each statistic, choose the orientation in {upper, lower, two-sided} whose evidence has the highest AUROC
  on these 2k labeled nodes (ties: keep the frozen direction). Everything else is the frozen pipeline.
- Evaluate AUROC and AP on all remaining nodes; the frozen detector is evaluated on the same remaining nodes.
- No other choices; no tuning. All graphs (dev, T1, T2) are reported, since nothing is selected.
