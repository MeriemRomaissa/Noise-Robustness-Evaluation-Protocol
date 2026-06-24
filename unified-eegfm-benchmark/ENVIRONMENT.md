# Environment Notes

This package was prepared on AI Station.

Inspected environments:

```text
/nicoletye/venvs/eegnet
/nicoletye/venvs/eegpt_labram
/nicoletye/venvs/biot_env
/nicoletye/venvs/cbramod_csbrain
```

All four inspected venvs reported:

```text
Python: 3.10.15
Torch: 2.3.0a0+6ddf5cf85e.nv24.04
CUDA runtime reported by torch: 12.4
torch.cuda.is_available(): True
```

The package `requirements.txt` lists the common Python packages used by the wrappers and preprocessing scripts. It is not a complete replacement for the original EEG-FM repositories' own requirements.

Do not blindly reinstall PyTorch from `requirements.txt`. Install or reuse a PyTorch build compatible with the machine's CUDA/runtime. On AI Station, the existing venvs already provide the tested Torch/CUDA combination.

Quick core environment test:

```bash
python - <<'PY'
import numpy, pandas, scipy, sklearn, mne, h5py, torch
print("Core imports OK")
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

Suggested AI Station venv mapping:

```text
Unified preprocessing / EEGNet utilities: /nicoletye/venvs/eegnet
LaBraM and EEGPT: /nicoletye/venvs/eegpt_labram
BIOT: /nicoletye/venvs/biot_env
CBraMod, CSBrain, CodeBrain: /nicoletye/venvs/cbramod_csbrain
```

Some original EEG-FM repositories may require additional packages beyond this wrapper package. Check each upstream repository's `requirements.txt`, `environment.yml`, or README before running its native code.
