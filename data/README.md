# Data — NASA C-MAPSS (FD001, FD003)

## Source

NASA Prognostics Center of Excellence Data Repository — Turbofan Engine Degradation Simulation
Data Set (Saxena & Goebel, 2008). Download from the NASA Prognostics Data Repository / DASHlink
mirror and place the raw `.txt` files under `data/raw/` exactly as downloaded (no edits).

Citation:
```
A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set",
NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
```

## Files expected

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD003.txt
├── test_FD003.txt
└── RUL_FD003.txt
```

Column format (whitespace-separated, no header): `unit_number, time_in_cycles, op_setting_1..3, s1..s21`.

## Integrity — SHA-256 checksums

These are printed by cell **S5 ("Load raw data")** of each experiment notebook the first time it is run
(`01_LSTM_FD001`, `02_LSTM_FD003`, `03_XGB_FD001`, `04_XGB_FD003`). Paste them below, then every later
run of any notebook re-computes and can be diffed against this table to confirm the raw files were never
edited between experiments.

| File | SHA-256 | Recorded from notebook |
|------|---------|--------------------------|
| `train_FD001.txt` | *(963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8)* | `01_LSTM_FD001` |
| `test_FD001.txt`  | *(3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851)* | `01_LSTM_FD001` |
| `RUL_FD001.txt`    | *(a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca)* | `01_LSTM_FD001` |
| `train_FD003.txt`  | *(2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2)* | `02_LSTM_FD003` |
| `test_FD003.txt`   | *(299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790)* | `02_LSTM_FD003` |
| `RUL_FD003.txt`    | *(df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb)* | `02_LSTM_FD003` |

**Cross-check rule:** when `03_XGB_FD001` and `04_XGB_FD003` are run, their S5 output must print the
*same* six hashes for the matching files. If any hash differs, the raw data changed between runs and the
experiment is invalid until re-run on the same files.

## Data-processing fingerprints (derived, not raw)

Checksums of the *processed* arrays (train/val/test windows after scaling) are **not** stored here —
they are per-run artefacts saved automatically to `results/<model>_<dataset>/fingerprint.json`
(see README.md section 5 and 11). This file only covers the untouched raw `.txt` inputs.
