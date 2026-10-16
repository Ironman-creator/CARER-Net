# CARER-Net

Research code and reproducibility artifacts for **CARER-Net: Bounded Residual Color Adaptation for Retinal Vessel Segmentation**.

The default model is **CARER-Core** (`color_detail_only_unet`). It learns a zero-initialized, bounded correction to the grayscale input of a GroupNorm residual-attention U-Net from RGB and local green-channel contrast. `carer_unet` enables the optional error-supervised dual-branch refiner evaluated in the supplementary material.

## Current manuscript version

Version 1.1.0 adds all five FIVES color controls, structural-analysis code, a CSV-only statistical reproduction command, and the 19 numerical tables accompanying the JMBE manuscript. The adapter adds 1,201 parameters to the 32,778,789-parameter backbone. FIVES mean Dice is 0.898350 for Core, 0.898092 for linear RGB projection, and 0.886644 for the continued grayscale control. Core-minus-projection thin-centerline recall is +0.00343; its four-endpoint-adjusted 98.75% crossed-bootstrap interval is [0.00078, 0.00617].

## Repository contents

- `models/`: CARER-Core, CARER-Refine, the GroupNorm backbone, and implemented comparison models.
- `train.py`, `train_carer_ablation.py`: backbone training and 30-pass input-adaptation experiments.
- `test.py`, `evaluation.py`: fixed-threshold testing and metric implementation.
- `evaluate_robustness.py`, `analyze_probability_calibration.py`, `benchmark_jiim_models.py`: robustness, calibration, and efficiency audits.
- `splits/`: the recorded train/validation membership lists used in the reported experiments.
- `results/`: aggregate and per-image numerical outputs used to prepare the manuscript tables. No fundus photographs or ground-truth masks are redistributed.
- `docs/`: dataset preparation and reproduction instructions.
- `analysis/`: vascular structural metrics, crossed-bootstrap analysis, and the separate architecture benchmark.
- `results/jmbe_tables/`: all 19 manuscript CSV tables, including the projection comparison.

## Installation

Create an isolated Python environment, install the PyTorch build appropriate for your CUDA driver from the official PyTorch selector, and then install the remaining dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The reported GPU benchmarks were recorded on an NVIDIA GeForce RTX 4070 with PyTorch `2.11.0+cu128`. Exact speed depends on hardware, drivers, and PyTorch/CUDA versions.

## Data

Download DRIVE, STARE, CHASE_DB1, and FIVES from their original providers and arrange them as described in [`docs/DATASETS.md`](docs/DATASETS.md). The datasets are not included in this repository.

## Minimal reproduction

Train the GroupNorm residual-attention backbone on a recorded split:

```bash
python train.py \
  --dataset FIVES \
  --model res_attention_unet_gn \
  --data-root data \
  --split-file splits/jiim_stage2/FIVES/seed_42.json \
  --image-size 512 \
  --input-mode grayscale \
  --epochs 50 \
  --batch-size 1 \
  --learning-rate 0.0001 \
  --loss bce_dice \
  --selection-metric soft_dice \
  --min-epochs 30 \
  --patience 15 \
  --seed 42 \
  --deterministic \
  --device cuda
```

Train the epoch-matched control, CARER-Core, and optional CARER-Refine arms from that checkpoint:

```bash
python train_carer_ablation.py \
  --dataset FIVES \
  --data-root data \
  --split-file splits/jiim_stage2/FIVES/seed_42.json \
  --initial-checkpoint weight_files/FIVES_best_ResAttentionUNetGN_seed42.pth \
  --epochs 30 \
  --warmup-epochs 5 \
  --head-learning-rate 0.0001 \
  --tail-learning-rate 0.00001 \
  --run-root runs \
  --weight-dir weight_files \
  --base-loss-weight 0.25 \
  --error-loss-weight 0.10 \
  --train-seed 42 \
  --variants gray_rgb_control_30_unet color_detail_only_unet carer_unet \
  --device cuda
```

Evaluate CARER-Core with the fixed threshold used in the manuscript:

```bash
python test.py \
  --dataset FIVES \
  --model color_detail_only_unet \
  --checkpoint weight_files/FIVES_best_color_detail_only_unet_seed42.pth \
  --data-root data \
  --input-mode gray_rgb \
  --threshold 0.5 \
  --fov derived \
  --seed 42 \
  --device cuda
```

The Windows/RTX 4070 workflows for the multi-dataset confirmation, external validation, robustness, calibration, and efficiency experiments are provided as `run_*.ps1`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and the current [`JMBE reproduction guide`](docs/JMBE_REPRODUCTION.md) for paths and the five additional color-control commands.

Reproduce the projection-comparison statistics from the included per-image table, without downloading images or training models:

```bash
python analysis/reproduce_projection_statistics.py
```

The command checks every reproduced mean and interval against the released table and writes a separate CSV under `artifacts/`.

## Reproducibility notes

- Seeds: `42`, `123`, and `3407`.
- Image size: `512 x 512`.
- Test threshold: `0.5`, fixed before official-test evaluation.
- Model selection: validation soft Dice.
- Batch size: `1`.
- Trainable input modules receive 5 module-only and 25 tail-update epochs; fixed-input arms receive 30 tail-update epochs.
- The conventional dataset copies contain replicated grayscale inputs; FIVES retains native RGB.
- The FIVES benchmark was used during development and final evaluation. Structural Core--projection comparisons were performed after training using saved masks.

## Checkpoints

The repository includes code, recorded splits, and numerical outputs. Train checkpoints locally using the supplied commands; existing checkpoint hashes are included in the result manifests.

## License

The source code is released under the MIT License. Dataset images and annotations remain under the terms of their respective providers and are not covered by this repository's license.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite the accompanying article if you use this code.

## Contact

Zijian Chen  
Guangzhou Huanan Business College, Guangzhou, Guangdong 510550, China  
Email: mrchan717829@foxmail.com  
ORCID: https://orcid.org/0009-0005-8411-7218
