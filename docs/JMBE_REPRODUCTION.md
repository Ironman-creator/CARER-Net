# JMBE manuscript reproduction

The manuscript title is *CARER-Net: Bounded Residual Color Adaptation for Retinal Vessel Segmentation*. Its default configuration is `color_detail_only_unet` (CARER-Core). Historical class names and checkpoint keys are retained for loading existing models.

## Environments and data

Python 3.10 or newer is required. Install PyTorch and torchvision for your hardware, followed by `pip install -r requirements.txt`. The reported RTX 4070 environment used PyTorch 2.11.0+cu128. Prepare datasets according to `DATASETS.md`; retain native FIVES RGB images. The recorded DRIVE/STARE/CHASE experiments used replicated grayscale copies.

## Training and fixed-threshold testing

Run the README backbone command first for seeds 42, 123 and 3407, using each corresponding `splits/jiim_stage2/FIVES/seed_<seed>.json`. Train each continuation from its matching backbone checkpoint. The following is a single-line seed-42 example, usable in both PowerShell and a Unix shell:

```text
python train_carer_ablation.py --dataset FIVES --data-root data --split-file splits/jiim_stage2/FIVES/seed_42.json --initial-checkpoint weight_files/FIVES_best_ResAttentionUNetGN_seed42.pth --run-root runs_fives_color_controls --weight-dir weight_files_fives_color_controls --epochs 30 --warmup-epochs 5 --head-learning-rate 0.0001 --tail-learning-rate 0.00001 --train-seed 42 --variants green_channel_unet gray_clahe_unet rgb_1x1_projection_unet carer_rgb_only_unet carer_contrast_only_unet --device cuda
```

Each trainable input arm has 5 module-only epochs plus 25 decoder-tail epochs; fixed green/CLAHE inputs have 30 tail-update epochs. The source checkpoint, split and number of data passes are shared. `gray_rgb_clahe` preprocessing provides the extra CLAHE channel required by that control; other input arms use `gray_rgb`.

```text
python test.py --dataset FIVES --model rgb_1x1_projection_unet --checkpoint weight_files_fives_color_controls/FIVES_best_rgb_1x1_projection_unet_seed42.pth --data-root data --result-root result_fives_color_controls --input-mode gray_rgb --image-size 512 --threshold 0.5 --fov derived --seed 42 --device cuda
```

Repeat with each model, seed and checkpoint. For `gray_clahe_unet`, set `--input-mode gray_rgb_clahe`. The source evaluates binary predictions with `p > 0.5`. Checkpoint selection uses validation soft Dice.

`run_fives_color_controls_4070.ps1` preserves the historical deployment paths: `runs_fives_v2_screen/`, `weight_files_fives_v2_screen/`, and the included `reference_result_fives_carer_official_test/` numerical records. The Python commands above let a new checkout choose its own paths.

## Reproduce statistics using only released CSVs

```text
python analysis/reproduce_projection_statistics.py
```

This uses all 200 FIVES images, three seeds, the released 1,200 arm/seed/image rows, bootstrap seed 20261016, and 10,000 replicates. It reproduces three region metrics and four structure endpoints, with pointwise 95% and four-endpoint Bonferroni 98.75% structural intervals. It validates the results against `S17_FIVES_core_minus_projection_paired_statistics.csv` to absolute tolerance 1e-12.

## Recompute structure from saved prediction masks

Download FIVES separately. Produce binary masks and `per_image_metrics.csv` with `test.py`. `--core-dir` and `--projection-dir` should contain `seed_42/`, `seed_123/`, and `seed_3407/`, each with `predictions/` and `per_image_metrics.csv`.

```text
python analysis/projection_comparison.py structure --fives-test /path/to/FIVES_official/test --core-dir result_fives_carer_official_test/FIVES/color_detail_only_unet --projection-dir result_fives_color_controls/FIVES/rgb_1x1_projection_unet --output-dir artifacts/projection_structure --workers 3
```

Zhang--Suen skeletons, reference EDT radius <=1.5 pixels, and 2-pixel branch/boundary tolerances match the manuscript. The included earlier Core measurements provide an integrity reference; newly trained models will have different predictions and should be evaluated as independent reproductions rather than checked against the historical masks. For those runs, use the `analysis/structure_metrics.py` CLI to calculate their own structure measurements.

## Separate architecture timing

```text
python analysis/projection_comparison.py benchmark --output-dir artifacts/local_benchmark
```

This uses randomly initialized shared-backbone weights, FP32, batch one, 512 x 512 inputs, 20 warm-up calls per arm, and 5 randomized blocks of 20 timed calls. Image loading, preprocessing and transfer are outside the timed region. The local GTX 1650 results and RTX 4070 results are reported separately in the article.

## Numerical tables and metrics

`results/jmbe_tables/` contains the exact 19 CSV files accompanying the manuscript. Table S15 stores projection/Core image-level structure values; S17 stores their paired statistics; S18--S19 store the local architecture timing summary and trace. Other tables cover color arms, four-dataset results, stratification, transfer, perturbations and calibration. Dataset images, labels and pretrained weights are obtained or trained separately.
