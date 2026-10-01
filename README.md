# CARER-Net

Official research code and reproducibility artifacts for **CARER-Net: Color Adaptation and Residual Error Refinement for Retinal Vessel Segmentation**.

The default model reported in the manuscript is **CARER-Core** (`color_detail_only_unet`). It learns a zero-initialized, bounded correction to the grayscale input of a GroupNorm residual-attention U-Net from RGB and local green-channel contrast. `carer_unet` additionally enables the optional signed error refiner; the manuscript does not present this extension as the default because its benefit was small and dataset-dependent.

## Repository contents

- `models/`: CARER-Core, CARER-Refine, the GroupNorm backbone, and implemented comparison models.
- `train.py`, `train_carer_ablation.py`: baseline training and compute-matched CARER training.
- `test.py`, `evaluation.py`: fixed-threshold testing and metric implementation.
- `evaluate_robustness.py`, `analyze_probability_calibration.py`, `benchmark_jiim_models.py`: robustness, calibration, and efficiency audits.
- `splits/`: the recorded train/validation membership lists used in the reported experiments.
- `results/`: aggregate and per-image numerical outputs used to prepare the manuscript tables. No fundus photographs or ground-truth masks are redistributed.
- `docs/`: dataset preparation and reproduction instructions.

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

The complete Windows/RTX 4070 workflows used for the multi-dataset confirmation, external validation, robustness, calibration, and efficiency experiments are provided as `run_*.ps1`. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before running them.

## Reproducibility notes

- Seeds: `42`, `123`, and `3407`.
- Image size: `512 x 512`.
- Test threshold: `0.5`, fixed before official-test evaluation.
- Model selection: validation soft Dice.
- Batch size: `1`.
- The published results are descriptive for the recorded datasets, splits, implementations, and hardware. They are not a claim of universal state-of-the-art performance.

## Checkpoints

Model checkpoints are intentionally omitted from the Git repository because of their size. The code, recorded splits, configurations, and numerical outputs are included. Checkpoints can be attached to a GitHub Release or an archival repository later, with SHA-256 hashes recorded in the release notes.

## License

The source code is released under the MIT License. Dataset images and annotations remain under the terms of their respective providers and are not covered by this repository's license.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite the accompanying article if you use this code.

## Contact

Zijian Chen  
Guangzhou Huanan Business College, Guangzhou, Guangdong 510550, China  
Email: mrchan717829@foxmail.com  
ORCID: https://orcid.org/0009-0005-8411-7218
