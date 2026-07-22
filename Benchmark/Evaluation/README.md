# Evaluation

`Evaluation/` owns output formats and result readers used after training.

The training loop writes two compatible views of the same run:

- Benchmark JSON files for reproducible machine reading:
  `config.json`, `checkpoint_load.json`, `fine_tuning.json`, `history.json`, `result.json`, `all_seed_results.json`, and `summary.json`
- LaBraM-style files for downstream analysis: 
  `log.txt`, `checkpoint.pth`, `checkpoint-<epoch>.pth`, and  `checkpoint-best.pth`

`log.txt` is one JSON object per epoch. It includes train, validation, and test metrics using the names expected by LaBraM-style stability analysis, including `train_loss`, `train_class_acc` `train_balanced_accuracy`, `val_accuracy`, `val_balanced_accuracy`, `test_accuracy`, `test_balanced_accuracy`, and `n_parameters`.

The test split is logged every epoch. The Benchmark still selects `checkpoint-best.pth` using validation metrics only.

`analyze_log_txt.py` is the lightweight reader for these logs:

```bash
python Benchmark/Evaluation/analyze_log_txt.py \
  --output-root outputs/labram \
  --selection-metric val_balanced_accuracy \
  --summary-json outputs/labram/evaluation_summary.json \
  --summary-csv outputs/labram/evaluation_summary.csv
```
