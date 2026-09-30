"""
Supplementary analysis for: "Evaluating XGBoost vs. LSTM for Engine RUL Prediction"
Reproduces every number added in the revised statistical / error-structure sections.

Usage (from the repository root, next to results/):
    python supplementary_analysis.py

Reads : results/{LSTM,XGB}_{FD001,FD003}/{predictions.csv, metrics_per_seed.csv, training_log.json}
Writes: results/comparison/supplementary_results.json  (+ printed summary)

Conventions
- Standard deviations use ddof=1 (sample SD) everywhere.
- Seed-level tests: two-sided Mann-Whitney U (10 vs 10 seeds), Holm correction over the six
  primary comparisons (RMSE, MAE, NASA x FD001, FD003); Welch t-test reported as a robustness check;
  Cliff's delta as effect size (positive = LSTM value larger).
- Engine-level tests use the seed-averaged prediction (y_pred_mean) of each of the 100 test engines.
  Paired bootstrap re-uses the exact RNG of 05_comparison.ipynb (RandomState(0), randint(0, n, n),
  10,000 resamples) so that the Delta-RMSE interval reproduces statistical_comparison.csv.
- NASA score: s = exp(-d/13) - 1 (d < 0), exp(d/10) - 1 (d >= 0), d = predicted - true.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RES = "results"
DATASETS = ["FD001", "FD003"]
MODELS = ["LSTM", "XGB"]
PRIMARY = ["primary_rmse", "primary_mae", "primary_nasa_score"]
TOL = ["primary_within_5", "primary_within_10", "primary_within_15"]
N_SEEDS = 10
BINS = [(0, 40), (40, 80), (80, 126)]


def nasa(d):
    return np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b)))


def load(model, ds):
    pred = pd.read_csv(f"{RES}/{model}_{ds}/predictions.csv").sort_values("engine_id").reset_index(drop=True)
    met = pd.read_csv(f"{RES}/{model}_{ds}/metrics_per_seed.csv")
    return pred, met


def seed_preds(pred):
    return [pred[f"y_pred_seed{s}"].values for s in range(N_SEEDS)]


out = {"per_seed_summary": {}, "seed_tests": [], "engine_level": {}, "rul_bins": [], "nasa_decomposition": [],
       "engine_diagnostics": {}, "unclipped": {}, "training": {}}

# ------------------------------------------------------------------ per-seed summary + tests
prim_rows = []
for ds in DATASETS:
    (Lp, Lm), (Xp, Xm) = load("LSTM", ds), load("XGB", ds)
    out["per_seed_summary"][ds] = {
        name: {c: {"mean": float(m[c].mean()), "sd": float(m[c].std(ddof=1))} for c in PRIMARY + TOL}
        for name, m in (("LSTM", Lm), ("XGB", Xm))
    }
    for c in PRIMARY + TOL:
        a, b = Lm[c].values, Xm[c].values
        row = {
            "dataset": ds, "metric": c, "lstm_mean": float(a.mean()), "xgb_mean": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "mwu_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
            "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
            "cliffs_delta": cliffs_delta(a, b),
        }
        out["seed_tests"].append(row)
        if c in PRIMARY:
            prim_rows.append(row)
for key in ("mwu_p", "welch_p"):
    adj = holm([r[key] for r in prim_rows])
    for r, a in zip(prim_rows, adj):
        r[key.replace("_p", "_holm")] = float(a)

# ------------------------------------------------------------------ engine level + diagnostics
for ds in DATASETS:
    (Lp, _), (Xp, _) = load("LSTM", ds), load("XGB", ds)
    assert (Lp.engine_id.values == Xp.engine_id.values).all()
    yt = Lp.y_true_clipped.values
    eL, eX = Lp.y_pred_mean.values - yt, Xp.y_pred_mean.values - yt
    SL, SX = seed_preds(Lp), seed_preds(Xp)
    nL = np.mean([nasa(p - yt) for p in SL], axis=0)  # per-engine penalty, mean over seeds
    nX = np.mean([nasa(p - yt) for p in SX], axis=0)

    rng = np.random.RandomState(0)
    n = len(yt)
    d_rmse, d_mae, d_nasa = [], [], []
    for _ in range(10000):
        idx = rng.randint(0, n, size=n)
        d_rmse.append(np.sqrt(np.mean(eL[idx] ** 2)) - np.sqrt(np.mean(eX[idx] ** 2)))
        d_mae.append(np.mean(np.abs(eL[idx])) - np.mean(np.abs(eX[idx])))
        d_nasa.append(nL[idx].sum() - nX[idx].sum())
    ci = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
    out["engine_level"][ds] = {
        "d_rmse": float(np.sqrt(np.mean(eL ** 2)) - np.sqrt(np.mean(eX ** 2))), "d_rmse_ci": ci(d_rmse),
        "d_mae": float(np.mean(np.abs(eL)) - np.mean(np.abs(eX))), "d_mae_ci": ci(d_mae),
        "d_nasa": float(nL.sum() - nX.sum()), "d_nasa_ci": ci(d_nasa),
        "wilcoxon_abs_err_p": float(stats.wilcoxon(np.abs(eL), np.abs(eX)).pvalue),
        "wilcoxon_nasa_p": float(stats.wilcoxon(nL, nX).pvalue),
        "ensemble_rmse": {"LSTM": float(np.sqrt(np.mean(eL ** 2))), "XGB": float(np.sqrt(np.mean(eX ** 2)))},
    }

    # RUL-stage table (RMSE averaged over seeds; bias = mean error averaged over seeds)
    for lo, hi in BINS:
        mk = (yt >= lo) & (yt < hi)
        row = {"dataset": ds, "bin": f"[{lo},{min(hi, 125)}]" if hi > 125 else f"[{lo},{hi})", "n": int(mk.sum())}
        for name, S in (("LSTM", SL), ("XGB", SX)):
            row[f"{name}_rmse"] = float(np.mean([np.sqrt(np.mean((p - yt)[mk] ** 2)) for p in S]))
            row[f"{name}_bias"] = float(np.mean([np.mean((p - yt)[mk]) for p in S]))
        out["rul_bins"].append(row)

    # NASA decomposition
    worst = int(np.argmax(nL - nX))  # engine with the largest LSTM-minus-XGB penalty gap
    keep = np.ones(n, bool)
    keep[worst] = False
    for name, S, pen in (("LSTM", SL, nL), ("XGB", SX, nX)):
        late = float(np.mean([nasa(p - yt)[(p - yt) >= 0].sum() for p in S]))
        early = float(np.mean([nasa(p - yt)[(p - yt) < 0].sum() for p in S]))
        out["nasa_decomposition"].append({
            "dataset": ds, "model": name, "late": late, "early": early,
            "n_late_engines": float(np.mean([((p - yt) >= 0).sum() for p in S])),
            "top5_share": float(np.sort(pen)[-5:].sum() / pen.sum()),
            "total_without_worst_gap_engine": float(pen[keep].sum()),
            "worst_gap_engine_id": int(Lp.engine_id[worst]),
        })

    top5 = lambda e: set(np.argsort(-e)[:5])
    out["engine_diagnostics"][ds] = {
        "resid_corr": float(np.corrcoef(eL, eX)[0, 1]),
        "top5_overestimated_overlap": len(top5(eL) & top5(eX)),
        "worst_gap_engine": {"engine_id": int(Lp.engine_id[worst]), "true": float(yt[worst]),
                             "LSTM_pred": float(Lp.y_pred_mean[worst]), "XGB_pred": float(Xp.y_pred_mean[worst]),
                             "LSTM_penalty": float(nL[worst]), "XGB_penalty": float(nX[worst])},
        "resid_min_max": {"LSTM": [float(eL.min()), float(eL.max())], "XGB": [float(eX.min()), float(eX.max())]},
        "resid_mean": {"LSTM": float(eL.mean()), "XGB": float(eX.mean())},
        "n_within_-10_+15": {"LSTM": int(((eL >= -10) & (eL <= 15)).sum()), "XGB": int(((eX >= -10) & (eX <= 15)).sum())},
        "n_over_+30": {"LSTM": int((eL > 30).sum()), "XGB": int((eX > 30).sum())},
    }

    # training dynamics
    lg_l = json.load(open(f"{RES}/LSTM_{ds}/training_log.json"))
    lg_x = json.load(open(f"{RES}/XGB_{ds}/training_log.json"))
    at = lambda i: [g["val_rmse_history"][min(i, len(g["val_rmse_history"]) - 1)] for g in lg_x]
    sizes = [os.path.getsize(f"{RES}/XGB_{ds}/models/xgb_seed{s}.ubj") / 1024 for s in range(N_SEEDS)]
    out["training"][ds] = {
        "LSTM_best_epoch": [min(g["best_epoch"] for g in lg_l), max(g["best_epoch"] for g in lg_l)],
        "LSTM_epochs_run": [min(g["epochs_run"] for g in lg_l), max(g["epochs_run"] for g in lg_l)],
        "LSTM_val_ep0": [min(g["val_rmse_history"][0] for g in lg_l), max(g["val_rmse_history"][0] for g in lg_l)],
        "LSTM_val_ep2": [min(g["val_rmse_history"][2] for g in lg_l), max(g["val_rmse_history"][2] for g in lg_l)],
        "LSTM_best_val": [min(g["best_val_rmse"] for g in lg_l), max(g["best_val_rmse"] for g in lg_l)],
        "XGB_trees": [min(g["trees_used"] for g in lg_x), max(g["trees_used"] for g in lg_x)],
        "XGB_seeds_at_cap": int(sum(g["trees_used"] >= 995 for g in lg_x)),
        "XGB_mean_val_rmse_at_rounds": {str(r): float(np.mean(at(r - 1))) for r in (100, 200, 400, 800)},
        "XGB_size_kib_min_max": [min(sizes), max(sizes)],
    }

# ------------------------------------------------------------------ unclipped (secondary) target
sec = pd.read_csv(f"{RES}/comparison/master_comparison_secondary.csv")
out["unclipped"] = sec.round(3).to_dict(orient="records")

path = f"{RES}/comparison/supplementary_results.json"
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print(f"written {path}")
for r in out["seed_tests"]:
    if r["metric"] in PRIMARY:
        print(f"{r['dataset']} {r['metric']:<20} diff={r['diff']:+.3f}  MWU p={r['mwu_p']:.4f}  Holm={r['mwu_holm']:.4f}  cliff={r['cliffs_delta']:+.2f}")
for ds in DATASETS:
    e = out["engine_level"][ds]
    print(ds, "dRMSE", round(e["d_rmse"], 3), [round(x, 3) for x in e["d_rmse_ci"]], "| dMAE", round(e["d_mae"], 3),
          [round(x, 3) for x in e["d_mae_ci"]], "| dNASA", round(e["d_nasa"], 1), [round(x, 1) for x in e["d_nasa_ci"]])
