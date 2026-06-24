# Verify Package

Run these checks from the source machine after creating or updating the package:

```bash
PKG=/nicoletye/workspace/unified_tuab/exports/meriem_reproduction_package

find "$PKG" -type f | sort
find "$PKG" -type f \( -name "*.h5" -o -name "*.edf" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" -o -name "*.pkl" -o -name "*.npz" \)
du -sh "$PKG"
```

Expected result:

- The file list contains only scripts and documentation.
- The banned-file scan returns no files.
- The package size is small enough for GitHub.

Optional syntax checks after copying scripts back into a working `unified_tuab` tree:

```bash
python -m py_compile scripts/build_canonical_tuab.py
python -m py_compile scripts/build_canonical_h5_max_coverage_split.py
python -m py_compile scripts/eegfm_adapters/eegfm_small_subset_train_worker.py
python -m py_compile scripts/eegfm_adapters/eegfm_small_subset_train_worker_remaining_models.py
```

This verification does not run training, preprocessing, or evaluation.
