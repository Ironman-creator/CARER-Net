"""Reproduce the paper's projection comparison from the released per-image CSV."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 3407)
REGION = ("dice", "sensitivity", "specificity")
STRUCTURE = ("cldice", "thin_centerline_recall", "branchpoint_f1", "boundary_f1_euclidean_2px")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def reproduce(input_path):
    spec = importlib.util.spec_from_file_location("structure_metrics", Path(__file__).with_name("structure_metrics.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rows = read(input_path)
    lookup = {(row["arm"], int(row["seed"]), row["sample"]): row for row in rows}
    samples = sorted({row["sample"] for row in rows})
    assert len(samples) == 200 and len(lookup) == len(rows) == 1200
    rng = np.random.default_rng(20261016)
    output = []
    for metric in (*REGION, *STRUCTURE):
        arrays = {arm: np.array([[float(lookup[(arm, seed, sample)][metric]) for sample in samples] for seed in SEEDS])
                  for arm in ("projection", "core")}
        delta = arrays["core"] - arrays["projection"]
        crossed = helper.crossed_bootstrap_distribution(delta, 10000, rng)
        image_delta = delta.mean(axis=0)
        image = image_delta[rng.integers(0, 200, (10000, 200))].mean(axis=1)
        low, high = np.quantile(crossed, [.025, .975])
        ilow, ihigh = np.quantile(image, [.025, .975])
        row = {"metric": metric, "core_mean": float(arrays["core"].mean()),
               "projection_mean": float(arrays["projection"].mean()),
               "core_minus_projection": float(delta.mean()),
               "crossed_95_low": float(low), "crossed_95_high": float(high),
               "image_only_95_low": float(ilow), "image_only_95_high": float(ihigh)}
        if metric in STRUCTURE:
            blo, bhi = np.quantile(crossed, [.00625, .99375])
            row.update(bonferroni_4_structure_98_75_low=float(blo), bonferroni_4_structure_98_75_high=float(bhi))
        output.append(row)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/jmbe_tables/S15_FIVES_projection_core_per_image_structure.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/jmbe_tables/S17_FIVES_core_minus_projection_paired_statistics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/projection_statistics.csv")
    args = parser.parse_args()
    output = reproduce(args.input)
    reference = {row["metric"]: row for row in read(args.reference)}
    maximum = max(abs(value - float(reference[row["metric"]][key]))
                  for row in output for key, value in row.items() if key != "metric")
    assert maximum < 1e-12, maximum
    keys = list(dict.fromkeys(key for row in output for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"metrics": len(output), "images": 200, "seeds": 3,
                      "max_absolute_difference_from_released_table": maximum}, indent=2))
