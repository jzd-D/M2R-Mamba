# M2R-Mamba: Experimental Code and Reproduction

This repository provides the model implementations, data preprocessing, training, evaluation, and result verification code for M2R-Mamba. The experiments cover the main comparisons and ablations on PTB-XL, together with seven baselines and the final calibrated candidate on MIT-BIH.

## Repository contents

- `src/`: Model, dataset, and metric implementations. The original M2R-Mamba implementation used in the PTB-XL experiments is `src/m2r_mamba/models/m2r_mamba.py`. The final MIT-BIH candidate uses `src/m2r_mamba/models/m2r_mamba_mit.py`, which includes center-beat readout. Both model versions preserve the corresponding frozen experimental implementations.
- `experiments/`: Training entry points and frozen configurations for three stages: 75 PTB-XL runs, 35 MIT-BIH baseline runs, and 5 runs of the final MIT-BIH candidate, totaling 115 independent training runs. The full PTB-XL model is shared between the main comparison and ablation analysis; these runs are counted once.
- `scripts/`: Data preparation, sequential execution, separate evaluation stages, MIT-BIH calibration, result aggregation, and integrity checks.
- `reference_results/`: Compact results for each seed, data split audits, statistical summaries, recorded runtime versions, and source hashes.

The snapshot contains the final training and calibration recipes and the compact records needed to verify their results. Raw ECG data, processed arrays, model weights, predictions for individual samples, training logs, caches, virtual environments, IDE/SSH settings, credentials, historical searches, and automatic scheduling scripts are excluded.

## 1. Quick verification without a GPU

Run the following commands from the repository root. They require only the Python standard library and do not train any models:

```bash
python scripts/verify_snapshot.py
python scripts/run.py all --dry-run
```

To recompute means and sample standard deviations from the recorded results for all five seeds:

```bash
python scripts/summarize.py
```

The summaries are written to `reproduced/summary/`. The script refuses to overwrite an existing output directory; use `--output` to select a new directory. Frozen reference results remain separate from newly generated outputs.

## 2. Training environment

Training targets Linux with an NVIDIA GPU and CUDA. The original frozen runtime image was checked and contains Python 3.11.10, PyTorch 2.5.0+cu124, CUDA 12.4, Mamba-SSM 2.3.1, and causal-conv1d 1.6.1. Other direct dependencies are pinned in `requirements.txt`; recorded training-related package versions are available in `reference_results/runtime.json`. The image identifier in that file is a provenance record, not a downloadable image reference. Identical results at the bit level across different hardware are not guaranteed.

In a fresh Python 3.11 environment, install the dependencies in this order:

```bash
python -m pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install causal-conv1d==1.6.1 --no-build-isolation
python -m pip install mamba-ssm==2.3.1 --no-build-isolation
```

The Mamba/CUDA extensions must be compatible with the installed PyTorch, CUDA, and compiler toolchain. A CUDA development toolchain is required if the extensions must be built from source. See the [official Mamba installation instructions](https://github.com/state-spaces/mamba). Preparing this snapshot did not include a fresh installation of all dependencies or a complete rerun of the 115 GPU training jobs.

After installation, run `python scripts/check_environment.py --models` to inspect package versions, check parameter counts for the 23 distinct model configurations, and test CPU forward passes for the ordinary CNN/RNN baselines. These checks passed in a separate CPU container using the original frozen runtime image.

## 3. Data preparation

Download [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and [MIT-BIH 1.0.0](https://physionet.org/content/mitdb/1.0.0/) from their official sources. Follow the dataset terms and cite the corresponding dataset publications. The datasets are not distributed with this repository.

Place `ptbxl_database.csv`, `scp_statements.csv`, and `records100/` under `data/raw/ptbxl/`. Place the MIT-BIH `.dat`, `.hea`, and `.atr` files under `data/raw/mitbih/`. Both `data/raw/` and `data/processed/` are ignored by Git.

```bash
python scripts/prepare_data.py ptbxl --raw data/raw/ptbxl
python scripts/prepare_data.py mitbih --raw data/raw/mitbih
```

PTB-XL uses 100 Hz recordings and the official folds: 1–8 for training, 9 for validation, and 10 for testing. Record handling follows the original preprocessing implementation. MIT-BIH uses the N/S/V/F classes, two waveform channels, eight RR feature channels, and a five-beat context. Record 201 is excluded from training because it belongs to the same subject as test record 202. Validation records are 106, 119, 223, and 230. Normalization statistics are computed from training data only.

After preprocessing, the script checks sample counts, class counts, and sample ID hashes against the frozen split audits. Expected training/validation/test sizes are 17,418/2,183/2,198 for PTB-XL and 40,097/8,858/49,603 for MIT-BIH. Investigate the dataset version if these checks fail; do not combine results obtained with different splits.

Existing processed-data directories are not overwritten. Use `--processed` to select a new destination. Data preparation requires several gigabytes of disk space and sufficient RAM. The raw data directory is read-only input.

## 4. Training and evaluation

The launcher runs jobs **sequentially**, with one training process at a time. Select the visible GPU using `CUDA_VISIBLE_DEVICES`. The launcher does not stop other users' jobs. Each stage's `manifest.json` defines the complete hyperparameters and seeds.

The frozen configurations retain the original GPU allocation limit of 15% of total GPU memory. This limit may be too low on smaller GPUs; adjust the `gpu_memory_fraction` resource setting in your own copy if necessary. Larger baselines require more memory than the proposed model.

Start with a small GPU smoke test after preparing the data:

```bash
python scripts/run.py ptb_main --job ptbxl_m2r_full_s1 --smoke --output reproduced/smoke_check
```

To train only the full PTB-XL model for all five seeds, select each job individually with `--job`, from `ptbxl_m2r_full_s1` through `ptbxl_m2r_full_s5`. To reproduce the complete experiment set, run:

```bash
python scripts/run.py standard
python scripts/run.py ptb_ablation
python scripts/run.py ptb_ablation --phase evaluate
python scripts/run.py mit_final
python scripts/run.py mit_final --phase evaluate
python scripts/calibrate_mit.py init
python scripts/calibrate_mit.py select
python scripts/calibrate_mit.py evaluate
python scripts/summarize.py --from-runs reproduced --output reproduced/new_summary
```

The `standard` stage contains 60 original PTB-XL comparison/ablation runs and 35 MIT-BIH baseline runs. Its trainer evaluates each run on the test set after selecting the best checkpoint using validation data. The `ptb_ablation` stage trains the 15 additional ablation runs, and `mit_final` trains the final candidate with five seeds. Each of these latter stages requires all training runs to finish and all checkpoints to be frozen before separate test evaluation.

Use `ptb_main` to select the 40 main PTB-XL comparison runs or `mit_baselines` to select the 35 MIT-BIH baseline runs. The `--seeds` option is useful for debugging a subset; all five seeds are required for the full summaries.

Processed data defaults to `data/processed/`, and new outputs default to `reproduced/`. Override these locations with the launcher's `--data` and `--output` options. When calling stage-level statistics or calibration scripts directly, set `M2R_DATA_ROOT` and `M2R_OUTPUT_ROOT` to the same locations. The launcher refuses to overwrite existing run directories, including interrupted runs.

For PTB-XL statistical analysis and inference benchmarking, run:

```bash
python experiments/standard/aggregate.py --bootstrap
python experiments/ptb_ablation/aggregate.py
python experiments/standard/benchmark.py
```

Patient bootstrap analysis requires the test predictions generated by training; this minimal snapshot includes only the original bootstrap summaries. These commands take longer than the quick checks. Latency benchmarking requires an idle GPU and is deferred if another compute process is detected. Latencies measured on different hardware are not directly interchangeable.

## Interpretation and reproducibility limits

All seeds are retained, and sample standard deviations use `ddof=1`. Ablation results are retained regardless of whether they favor the full model. PTB-XL is the primary study. MIT-BIH is a separately trained second-task evaluation, not a transfer of weights between datasets.

The final MIT-BIH candidate received additional tuning, so its search budget differs from those of the seven baselines. Historical test results had already been inspected during development. The original validation-based calibration selected an F-class logit penalty of 1.5. After retraining, select the penalty again using validation data; do not tune on test results to match the reference numbers. The selection rules and limitations are preserved in the calibration script.

## License and repository use

No project license is included because the source archive did not contain one that could be carried forward. The project authors must choose the project license; dependencies and datasets remain subject to their respective licenses and terms.

This directory can be used directly as the repository root. The included `.gitignore` excludes subsequently generated datasets, model weights, caches, and training logs.

