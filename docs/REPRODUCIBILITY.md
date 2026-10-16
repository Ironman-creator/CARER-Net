# Reproducibility guide

## Frozen protocol

The reported experiments use:

- random seeds `42`, `123`, and `3407`;
- an input resolution of `512 x 512`;
- batch size `1`;
- BCE + soft Dice loss with equal nominal weights;
- validation soft Dice for checkpoint selection;
- a fixed test threshold of `0.5`;
- recorded train/validation membership lists in `splits/`.

CARER-Core is `color_detail_only_unet`. The epoch-matched control is `gray_rgb_control_30_unet`. `carer_unet` adds the optional error-supervised dual-branch refiner. The current manuscript's five additional input controls and projection-comparison analysis are documented in `JMBE_REPRODUCTION.md`.

## Recommended order

1. Prepare the four datasets using `docs/DATASETS.md`.
2. Run `run_jiim_stage2_preflight_4070.ps1` to validate CUDA, data pairing, splits, and model construction.
3. Use `run_carer_3datasets_4070.ps1` for DRIVE, STARE, and CHASE_DB1.
4. Use the Python commands in the main README for the FIVES backbone and CARER arms.
5. Run the protocol-baseline, external-validation, robustness, and efficiency PowerShell workflows as required.
6. Compare generated CSV/JSON files with the corresponding files under `results/`.

## Result provenance

`results/` contains numerical summaries copied from the completed experiment outputs used for the manuscript. It contains no model weights and no dataset images. The files are provided to make table values and paired comparisons auditable without requiring a full retraining run.

## Determinism limits

The scripts seed Python, NumPy, and PyTorch and support deterministic cuDNN execution. Exact bitwise equality can still depend on GPU model, CUDA/cuDNN, PyTorch build, drivers, and low-level kernels. Compare both configuration files and aggregate metrics, not only checkpoint hashes, when reproducing on a different system.

## Checkpoint release

GitHub rejects ordinary files larger than 100 MB. Publish large checkpoints through a GitHub Release or an archival service such as Zenodo rather than committing them to Git history. Record the filename, model, dataset, seed, and SHA-256 hash for every released checkpoint.
