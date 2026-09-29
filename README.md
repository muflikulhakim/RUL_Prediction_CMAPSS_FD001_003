# RUL Prediction on NASA C-MAPSS: XGBoost vs. LSTM (FD001 & FD003)

**Status: PROTOCOL LOCKED (v1.0) — written before any experiment code was run.**
Everything in sections 1–9 is fixed *a priori*. Results are produced only by following this protocol.
Any change after the first test-set evaluation must be recorded in the Changelog (section 11)
and the affected results must be labelled *post-hoc*.

DOI: `10.5281/zenodo.23031016`

---

## 1. Research question and scope

Does a classical machine-learning model (XGBoost regression) or a deep-learning model (LSTM) give better
RUL prediction on C-MAPSS, and at what computational cost?

- Datasets: **FD001** (1 operating condition, 1 fault mode: HPC degradation) and
  **FD003** (1 operating condition, 2 fault modes: HPC + fan degradation).
- Models: **LSTM** (PyTorch) and **XGBoost** (`XGBRegressor`).
- Out of scope: FD002/FD004 (6 operating conditions), hyper-parameter optimisation, feature engineering,
  other architectures. These are stated as limitations, not hidden.

## 2. Experimental design (2 × 2)

| Notebook | Model | Dataset |
|----------|-------|---------|
| `01_LSTM_FD001.ipynb` | LSTM | FD001 |
| `02_LSTM_FD003.ipynb` | LSTM | FD003 |
| `03_XGB_FD001.ipynb`  | XGBoost | FD001 |
| `04_XGB_FD003.ipynb`  | XGBoost | FD003 |
| `05_comparison.ipynb` | analysis only (reads saved results, trains nothing) | both |

**Fairness rules**

1. The four notebooks have the **same section structure** (section 10). Only `DATASET_ID` / `MODEL`
   and the model-specific cells differ.
2. **Model hyper-parameters are identical across datasets** (FD001 vs FD003) for the same model.
3. **Data pipeline is identical across models**: same features, scaler, split, windows.
   Enforced by *data fingerprints* (SHA-256 of the arrays, section 5): LSTM and XGBoost notebooks on the
   same dataset must print identical fingerprints, otherwise the run is invalid.
4. Both models use the **same early-stopping rule on the same validation engines**.
5. The test set is used **once per seed, after training is finished**. No decision is made from test results.

## 3. Data

- Source: NASA Prognostics Center of Excellence, C-MAPSS Turbofan Engine Degradation Simulation
  (Saxena & Goebel, 2008). Files: `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`.
- Raw files are stored unmodified in `data/raw/`; SHA-256 checksums are recorded in `data/README.md`
  and verified at the start of every notebook.
- Expected sizes: 100 train engines and 100 test engines per subset.

## 4. Preprocessing (identical for all four notebooks)

| Step | Decision |
|------|----------|
| RUL label (train) | `RUL = max_cycle − cycle`, piece-wise clipped: `min(RUL, 125)` |
| RUL label (test) | `RUL_FD00X.txt` gives the true RUL at the last observed cycle |
| Features | 14 sensors: `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21` (same list for both subsets). Operating settings are dropped (single operating condition). |
| Feature-list check | A cell verifies that the 7 dropped sensors (`s1, s5, s6, s10, s16, s18, s19`) have (near-)zero variance in the training data and prints the evidence. If the check fails for a subset, the discrepancy is reported, the list is **not** changed silently. |
| Validation split | **By engine**, never by row: 80 train / 20 validation engines, `SPLIT_SEED = 42` (`RandomState(42).permutation`). The validation engine IDs and their hash are printed and must match across notebooks of the same dataset. |
| Scaling | `StandardScaler`, **fitted on the 80 training engines only**, then applied to validation and test. Same scaler for LSTM and XGBoost (trees are invariant to it; using one pipeline keeps the data identical). |
| Windowing | Window length **30** cycles, stride 1, never crossing engine boundaries; label = RUL at the last cycle of the window. |
| Test input | One window per test engine: its **last 30 cycles**. An assertion checks every test engine has ≥ 30 cycles; no padding is used. |
| XGBoost input | The same window flattened to 30 × 14 = 420 features (order fixed: time-major). |

## 5. Data fingerprints (validity check)

Each notebook computes and saves `fingerprint.json` containing SHA-256 hashes of:
`X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test_raw`, `y_test_clipped`, and the sorted validation engine IDs.
The comparison notebook refuses to compare two runs of the same dataset whose fingerprints differ.

## 6. Models and hyper-parameters (fixed, no tuning)

**LSTM**

| Parameter | Value |
|-----------|-------|
| Architecture | LSTM(input 14, hidden 64, 2 layers, inter-layer dropout 0.2) → last time-step → Linear(64→32) → ReLU → Linear(32→1) |
| Trainable parameters | 55,873 (asserted in the notebook) |
| Loss / optimiser | MSE / Adam, lr = 1e-3, weight decay = 0 |
| Batch size | 64 (training data shuffled) |
| Gradient clipping | max-norm 1.0 |
| Epochs | max 100, **early stopping** on validation RMSE, patience 10, best weights restored |

**XGBoost**

| Parameter | Value |
|-----------|-------|
| Objective | `reg:squarederror`, eval metric RMSE |
| Trees | `n_estimators = 1000` upper bound, **early stopping** on validation RMSE, `early_stopping_rounds = 30` |
| Learning rate / depth | 0.05 / 6 |
| Sampling | `subsample = 0.8`, `colsample_bytree = 0.8` |
| Other | `min_child_weight = 1`, `reg_lambda = 1`, `tree_method = "hist"`, `random_state = seed` |

These values are fixed by this protocol, not selected on the test set.
They were not tuned; both models are therefore "reasonable-default" baselines (see Limitations).

**Sanity-check rule.** If LSTM training loss does not decrease within the first epochs, that is treated as a
bug (data/shape/scaling), investigated on **training and validation data only**, and logged in the Changelog.
Hyper-parameters are not adjusted by looking at test results.

## 7. Repetitions and reproducibility

- **10 seeds** per (model, dataset): `SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
  (weight init, dropout, shuffling for LSTM; `random_state` for XGBoost).
  The validation split is fixed (`SPLIT_SEED = 42`) across seeds and models.
- PyTorch: `torch.manual_seed`, `cudnn.deterministic = True`, `cudnn.benchmark = False`;
  residual GPU non-determinism is acknowledged.
- Environment is logged to `env.json` (Python, PyTorch, XGBoost, scikit-learn, NumPy versions, GPU, CUDA, CPU model).
- Training on Google Colab, GPU T4.

## 8. Evaluation

Evaluation happens on the **100 test engines (last window)**, per seed.

**Primary target:** `RUL_FD00X.txt` clipped at 125 (same clipping as training, standard in the literature).
**Secondary target:** the official unclipped `RUL_FD00X.txt`. Both are reported; the primary one was chosen in advance.

| Metric | Definition |
|--------|------------|
| RMSE | `sqrt(mean((ŷ − y)²))` |
| MAE | `mean(|ŷ − y|)` |
| NASA score S | `Σ (exp(−d/13) − 1)` if `d < 0`, else `Σ (exp(d/10) − 1)`, with `d = ŷ − y` (late predictions penalised more) |
| Tolerance rate | share of engines with `|ŷ − y| ≤ 5 / 10 / 15` cycles |

Reported as **mean ± std over 10 seeds** (RMSE, MAE, S, tolerance rate); per-seed values and per-engine predictions
are saved. Validation metrics (best epoch / best iteration) are also saved.

**Statistical comparison (`05_comparison`)**: per-engine errors averaged over seeds; paired bootstrap
(10,000 resamples of the 100 engines) for the 95 % CI of `ΔRMSE = RMSE_LSTM − RMSE_XGB`; Wilcoxon signed-rank on
per-engine absolute errors. A difference is called "significant" only if the CI excludes 0.

## 9. Efficiency measurements

| Item | Protocol |
|------|----------|
| Model size | Native binary artefacts: LSTM `state_dict` (`.pth`) and XGBoost UBJSON (`.ubj`). Also report **LSTM parameters** and **XGBoost trees actually used** (`best_iteration + 1`) and total nodes. |
| Training time | Wall-clock per seed, plus epochs / trees actually used. |
| Inference latency | **CPU only**, 1 thread for both models, same Colab session; 50 warm-up calls, then 1,000 timed calls; report **median and P95**, for batch size 1 and for the full 100-engine test batch (per-sample). |
| Hardware | CPU model and Colab runtime type recorded. |

Latency is hardware-dependent; numbers indicate relative cost, not performance on a specific embedded target.

## 10. Notebook structure (identical in all four)

Each item below is one cell (or one markdown + one code cell):

| # | Section | Purpose |
|---|---------|---------|
| S0 | Title / protocol reference | Notebook ID, protocol version |
| S1 | Environment | Installs, imports, version logging → `env.json` |
| S2 | Drive & paths | Mount Drive, `BASE_DIR`, output folders |
| S3 | Configuration | `DATASET_ID`, `MODEL`, all constants from this README |
| S4 | Reproducibility | Seed helpers, determinism flags |
| S5 | Load raw data | Read files, verify checksums, shape assertions |
| S6 | RUL labels | Train RUL (clipped), test RUL (raw + clipped) |
| S7 | Feature check | Variance table, confirm 14-sensor list |
| S8 | Engine split | 80/20 by engine, print IDs + hash |
| S9 | Scaling | Fit on train engines only |
| S10 | Windowing | Train / val / test windows, shape assertions |
| S11 | Fingerprints | Hash arrays → `fingerprint.json` |
| S12 | Model definition | LSTM class or XGBoost factory; parameter assertion |
| S13 | Single-seed dry run | Loss curve / sanity check (train + val only) |
| S14 | Multi-seed training | Loop over `SEEDS`, save best model per seed |
| S15 | Test evaluation | Metrics (primary + secondary target), per-engine predictions |
| S16 | Efficiency | Size, parameters/trees, training time, CPU latency |
| S17 | Save results | `metrics_per_seed.csv`, `predictions.csv`, `summary.json`, `config.json` |
| S18 | Figures | Predicted vs. true, error distribution, learning curves (300 DPI) |

## 11. Deviation policy and Changelog

- Nothing in sections 1–9 changes after the first test evaluation, except through a Changelog entry stating
  *what*, *why*, and *whether test results had been seen*. Results affected are labelled post-hoc.
- Cells are re-run top to bottom before saving results ("Restart & Run All").

| Version | Date | Change | Test results seen? |
|---------|------|--------|--------------------|
| 1.0 | 2026-09-28 | Protocol created | No |

**Note on earlier exploratory runs.** An earlier exploratory experiment (single seed, different LSTM settings per
subset, no validation split) exists and is **discarded**. It is not reported as evidence, and none of its outputs are reused.
This study is a complete re-run under the protocol above.

## 12. Repository / Drive layout
RUL_Prediction_CMAPSS_FD001_003/
├── README.md # this protocol
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│ ├── README.md # download link + SHA-256 of raw files
│ └── raw/ # train/test/RUL for FD001 and FD003 (unmodified)
├── notebooks/
│ ├── 01_LSTM_FD001.ipynb
│ ├── 02_LSTM_FD003.ipynb
│ ├── 03_XGB_FD001.ipynb
│ ├── 04_XGB_FD003.ipynb
│ └── 05_comparison.ipynb
└── results/
├── LSTM_FD001/ ├── LSTM_FD003/ ├── XGB_FD001/ ├── XGB_FD003/
│ (each: env.json, fingerprint.json, config.json, metrics_per_seed.csv,
│ predictions.csv, summary.json, models/, figures/)
└── comparison/ # final tables + figures for the paper


## 13. Limitations (stated in advance)

- Only the two single-operating-condition subsets; results do not transfer automatically to FD002/FD004.
- No hyper-parameter search: fixed a-priori configurations for both models.
- Test evaluation uses only the last window of each test engine (standard C-MAPSS protocol).
- One fixed validation split; variance across splits is not estimated.
- Latency/size measured on a Colab CPU, not on target embedded hardware.

## 14. How to cite
Hakim, M. (2026). RUL prediction on NASA C-MAPSS: XGBoost vs. LSTM (FD001 & FD003).
Zenodo. https://doi.org/10.5281/zenodo.23031016

Dataset: A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set",
NASA Ames Prognostics Data Repository.

License: code MIT; derived results and figures CC BY 4.0; raw C-MAPSS data under NASA's terms.
