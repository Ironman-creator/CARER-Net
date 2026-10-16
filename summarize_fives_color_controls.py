import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 123, 3407)
NEW_MODELS = (
    "green_channel_unet",
    "gray_clahe_unet",
    "rgb_1x1_projection_unet",
    "carer_rgb_only_unet",
    "carer_contrast_only_unet",
)
CONTROL = "gray_rgb_control_30_unet"
CORE = "color_detail_only_unet"
ALL_MODELS = (CONTROL,) + NEW_MODELS + (CORE,)
INPUT_MODES = {
    "green_channel_unet": "gray_rgb",
    "gray_clahe_unet": "gray_rgb_clahe",
    "rgb_1x1_projection_unet": "gray_rgb",
    "carer_rgb_only_unet": "gray_rgb",
    "carer_contrast_only_unet": "gray_rgb",
}
LABELS = {
    CONTROL: "Epoch-matched grayscale control",
    "green_channel_unet": "Fixed green channel",
    "gray_clahe_unet": "PIL grayscale + CLAHE",
    "rgb_1x1_projection_unet": "Learned 1x1 RGB projection",
    "carer_rgb_only_unet": "CARER RGB-only residual",
    "carer_contrast_only_unet": "CARER local-contrast-only residual",
    CORE: "CARER-Core (RGB + local contrast)",
}
TEST_METRICS = ("accuracy", "sensitivity", "specificity", "dice", "iou", "auc_roc", "auc_pr")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and summarize the predefined FIVES color-control experiment."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--reference-result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values):
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def bootstrap_interval(values, seed, repeats=10000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(values), size=(repeats, len(values)))
    means = values[sampled].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def result_directory(args, model, seed):
    root = args.result_root if model in NEW_MODELS else args.reference_result_root
    return root / "FIVES" / model / f"seed_{seed}"


def main():
    args = parse_args()
    training = {}
    reference_splits = {}
    reference_hashes = {}
    for seed in SEEDS:
        for model in NEW_MODELS:
            run_dir = args.run_root / "FIVES" / model / f"seed_{seed}"
            config = read_json(run_dir / "config.json")
            result = read_json(run_dir / "best_result.json")
            split = read_json(run_dir / "split.json")
            if config.get("epochs") != 30:
                raise RuntimeError(f"Expected 30 epochs for {model}, seed {seed}")
            expected_warmup = 0 if model in {"green_channel_unet", "gray_clahe_unet"} else 5
            if config.get("warmup_epochs") != expected_warmup:
                raise RuntimeError(f"Warmup mismatch for {model}, seed {seed}")
            if config.get("input_mode") != INPUT_MODES[model]:
                raise RuntimeError(f"Input-mode mismatch for {model}, seed {seed}")
            if config.get("official_test_accessed") or result.get("official_test_accessed"):
                raise RuntimeError(f"Training reports test access for {model}, seed {seed}")
            if int(config.get("train_seed")) != seed:
                raise RuntimeError(f"Training-seed mismatch for {model}, seed {seed}")
            if seed not in reference_splits:
                reference_splits[seed] = split
                reference_hashes[seed] = config["initial_checkpoint_sha256"]
            elif split != reference_splits[seed]:
                raise RuntimeError(f"Split mismatch for {model}, seed {seed}")
            if config["initial_checkpoint_sha256"] != reference_hashes[seed]:
                raise RuntimeError(f"Initial checkpoint mismatch for {model}, seed {seed}")
            training[(model, seed)] = (config, result)

    test_metadata = {}
    per_image = {}
    expected_samples = None
    for model in ALL_MODELS:
        for seed in SEEDS:
            directory = result_directory(args, model, seed)
            metadata = read_json(directory / "metrics.json")
            rows = read_csv(directory / "per_image_metrics.csv")
            if metadata.get("dataset") != "FIVES" or metadata.get("model") != model:
                raise RuntimeError(f"Test metadata mismatch for {model}, seed {seed}")
            if int(metadata.get("seed")) != seed:
                raise RuntimeError(f"Test seed mismatch for {model}, seed {seed}")
            if float(metadata.get("threshold")) != 0.5 or metadata.get("threshold_source") != "fixed":
                raise RuntimeError(f"Non-frozen threshold for {model}, seed {seed}")
            if int(metadata.get("test_samples")) != 200 or len(rows) != 200:
                raise RuntimeError(f"Expected 200 test samples for {model}, seed {seed}")
            keyed = {row["sample"]: row for row in rows}
            if len(keyed) != 200:
                raise RuntimeError(f"Duplicate test keys for {model}, seed {seed}")
            if expected_samples is None:
                expected_samples = set(keyed)
            elif set(keyed) != expected_samples:
                raise RuntimeError(f"Test key mismatch for {model}, seed {seed}")
            test_metadata[(model, seed)] = metadata
            per_image[(model, seed)] = keyed

    validation_rows = []
    for model in NEW_MODELS:
        for seed in SEEDS:
            config, result = training[(model, seed)]
            validation_rows.append(
                {
                    "model": model,
                    "label": LABELS[model],
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "parameter_count": config["parameter_count"],
                    "validation_dice": result["best_validation_dice"],
                    "validation_soft_dice": result["best_validation_soft_dice"],
                    "validation_sensitivity": result["best_validation_sensitivity"],
                    "validation_specificity": result["best_validation_specificity"],
                }
            )

    summary_rows = []
    for model in ALL_MODELS:
        row = {
            "model": model,
            "label": LABELS[model],
            "seeds": "42;123;3407",
            "parameter_count": test_metadata[(model, 42)]["parameter_count"],
        }
        for metric in TEST_METRICS:
            values = [
                float(test_metadata[(model, seed)]["metrics"][metric]["mean"])
                for seed in SEEDS
            ]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_across_seeds_{metric}"] = sample_std(values)
        summary_rows.append(row)

    averaged_per_image = {}
    for model in ALL_MODELS:
        averaged_per_image[model] = {}
        for sample in sorted(expected_samples):
            averaged_per_image[model][sample] = {
                metric: float(
                    np.mean(
                        [float(per_image[(model, seed)][sample][metric]) for seed in SEEDS]
                    )
                )
                for metric in ("dice", "sensitivity", "specificity", "iou")
            }

    comparisons = [(CORE, CONTROL)]
    for model in NEW_MODELS:
        comparisons.extend(((model, CONTROL), (model, CORE)))
    pairwise_rows = []
    for index, (candidate, baseline) in enumerate(comparisons):
        row = {
            "candidate": candidate,
            "candidate_label": LABELS[candidate],
            "baseline": baseline,
            "baseline_label": LABELS[baseline],
            "test_images": len(expected_samples),
            "seeds_averaged_per_image": 3,
        }
        for metric_index, metric in enumerate(("dice", "sensitivity", "specificity", "iou")):
            deltas = [
                averaged_per_image[candidate][sample][metric]
                - averaged_per_image[baseline][sample][metric]
                for sample in sorted(expected_samples)
            ]
            low, high = bootstrap_interval(
                deltas, seed=20261001 + index * 20 + metric_index
            )
            row[f"mean_delta_{metric}"] = float(np.mean(deltas))
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
            row[f"images_improved_{metric}"] = int(sum(value > 0 for value in deltas))
        pairwise_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "FIVES_color_controls_validation.csv", validation_rows)
    write_csv(args.output_dir / "FIVES_color_controls_test_summary.csv", summary_rows)
    write_csv(args.output_dir / "FIVES_color_controls_pairwise.csv", pairwise_rows)

    integrity = {
        "protocol": (
            "Five predefined controls; FIVES historical train/validation split per seed; "
            "same seed-specific pretrained GN checkpoint; 30 data passes; model selection "
            "by validation soft Dice; official test evaluated for every arm at fixed threshold 0.5"
        ),
        "new_models": list(NEW_MODELS),
        "reference_models": [CONTROL, CORE],
        "seeds": list(SEEDS),
        "initial_checkpoint_sha256_by_seed": {
            str(seed): reference_hashes[seed] for seed in SEEDS
        },
        "train_validation_splits_identical_within_seed": True,
        "test_threshold": 0.5,
        "test_model_selection_performed": False,
        "test_samples": 200,
        "bootstrap_unit": "test image after averaging three seeds",
        "bootstrap_multiplicity_adjusted": False,
    }
    with (args.output_dir / "FIVES_color_controls_integrity.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(integrity, file, indent=2, ensure_ascii=False)

    dice_lookup = {row["model"]: row for row in summary_rows}
    report_lines = [
        "FIVES predefined color-control experiment",
        "=========================================",
        "",
        "Official-test Dice (mean +/- SD across seeds):",
    ]
    for model in ALL_MODELS:
        row = dice_lookup[model]
        report_lines.append(
            f"{LABELS[model]}: {float(row['mean_dice']):.6f} +/- "
            f"{float(row['std_across_seeds_dice']):.6f}"
        )
    report_lines.extend(
        [
            "",
            "This script reports all five predefined arms. It does not choose a model or",
            "threshold from the official test results. Pairwise intervals are descriptive",
            "image-level bootstrap intervals after averaging the three seeds.",
        ]
    )
    (args.output_dir / "FIVES_color_controls_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    for row in summary_rows:
        print(
            f"{row['model']:<32} dice={float(row['mean_dice']):.6f} "
            f"+/-{float(row['std_across_seeds_dice']):.6f}"
        )
    print("Summary complete; no test-set model selection was performed.")


if __name__ == "__main__":
    main()
