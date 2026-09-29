## Primary results (RUL clipped at 125), mean +/- std over 10 seeds

| model   | dataset   |   rmse_mean |   rmse_std |   mae_mean |   mae_std |   nasa_score_mean |   nasa_score_std |   within_5_mean |   within_10_mean |   within_15_mean |
|:--------|:----------|------------:|-----------:|-----------:|----------:|------------------:|-----------------:|----------------:|-----------------:|-----------------:|
| LSTM    | FD001     |      14.209 |      0.939 |     10.253 |     0.798 |           457.902 |          129.917 |           0.394 |            0.607 |            0.754 |
| XGB     | FD001     |      13.267 |      0.384 |     10.212 |     0.41  |           291.288 |           21.573 |           0.342 |            0.586 |            0.762 |
| LSTM    | FD003     |      14.199 |      0.68  |      9.795 |     0.685 |           818.447 |          231.708 |           0.436 |            0.655 |            0.772 |
| XGB     | FD003     |      14.744 |      0.221 |     10.867 |     0.262 |           516.141 |           30.948 |           0.374 |            0.585 |            0.702 |

## Statistical comparison (paired bootstrap, 10,000 resamples; Wilcoxon signed-rank)

| dataset   |   delta_rmse_lstm_minus_xgb |   bootstrap_ci_low_2.5pct |   bootstrap_ci_high_97.5pct | ci_excludes_zero   |   wilcoxon_stat |   wilcoxon_p |
|:----------|----------------------------:|--------------------------:|----------------------------:|:-------------------|----------------:|-------------:|
| FD001     |                      0.4594 |                   -0.8599 |                      1.7861 | False              |            2178 |       0.2328 |
| FD003     |                     -1.2961 |                   -3.2102 |                      0.4721 | False              |            1954 |       0.0496 |

## Efficiency comparison

| model   | dataset   |   model_size_kb |   mean_training_time_sec |   cpu_latency_single_ms_median |   cpu_latency_single_ms_p95 |   cpu_latency_batch_per_sample_ms_median | capacity                          |
|:--------|:----------|----------------:|-------------------------:|-------------------------------:|----------------------------:|-----------------------------------------:|:----------------------------------|
| LSTM    | FD001     |         221.619 |                   13.738 |                          0.565 |                       0.876 |                                    0.072 | 55873 parameters                  |
| XGB     | FD001     |        3998.87  |                    7.314 |                          0.079 |                       0.107 |                                    0.001 | 889 trees / 102529 nodes (seed 0) |
| LSTM    | FD003     |         221.619 |                   19.718 |                          0.806 |                       1.087 |                                    0.049 | 55873 parameters                  |
| XGB     | FD003     |        4280.98  |                    7.521 |                          0.081 |                       0.111 |                                    0.001 | 974 trees / 109450 nodes (seed 0) |
