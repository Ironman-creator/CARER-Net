import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data_utils import RetinalDataset, clahe_uint8, discover_pairs
from models.Attention_UNet_V5 import GrayRGBControlUNet
from train_carer_ablation import (
    BASELINE_PRESERVING,
    INPUT_MODES,
    MODEL_BUILDERS,
    load_gn_weights,
)
from train_carer_paired import load_torch_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent
SEEDS = (42, 123, 3407)
MODELS = (
    "green_channel_unet",
    "gray_clahe_unet",
    "rgb_1x1_projection_unet",
    "carer_rgb_only_unet",
    "carer_contrast_only_unet",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preflight the predefined five-arm FIVES color-control experiment."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def check_cuda(required):
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {properties.name} "
            f"({properties.total_memory / (1024 ** 3):.1f} GiB)"
        )
    elif required:
        raise RuntimeError("CUDA is unavailable; the 4070 experiment requires CUDA")


def check_clahe():
    synthetic = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    first = clahe_uint8(synthetic, clip_limit=2.0, tile_grid_size=(8, 8))
    second = clahe_uint8(synthetic, clip_limit=2.0, tile_grid_size=(8, 8))
    if first.shape != synthetic.shape or first.dtype != np.uint8:
        raise RuntimeError("CLAHE returned an invalid shape or dtype")
    if not np.array_equal(first, second):
        raise RuntimeError("CLAHE preprocessing is not deterministic")
    print("Deterministic NumPy CLAHE check passed (clip=2.0, grid=8x8).")


def check_split(split_path, available_keys):
    with split_path.open("r", encoding="utf-8-sig") as file:
        record = json.load(file)
    train = list(record.get("train", []))
    validation = list(record.get("validation", []))
    if len(train) != 540 or len(validation) != 60:
        raise RuntimeError(
            f"Expected 540/60 FIVES train/validation keys in {split_path}, "
            f"found {len(train)}/{len(validation)}"
        )
    if set(train) & set(validation):
        raise RuntimeError(f"Train/validation overlap in {split_path}")
    if set(train) | set(validation) != available_keys:
        raise RuntimeError(f"Split keys do not match prepared FIVES training data: {split_path}")
    return record


def check_models(checkpoint, dataset_sample):
    control = GrayRGBControlUNet().eval()
    load_gn_weights(control, checkpoint, "gray_rgb_control_30_unet", "FIVES")
    gray_rgb = dataset_sample[:, :4]
    with torch.inference_mode():
        control_logits = control(gray_rgb)

    for model_name in MODELS:
        model = MODEL_BUILDERS[model_name]().eval()
        missing = load_gn_weights(model, checkpoint, model_name, "FIVES")
        sample = dataset_sample if INPUT_MODES[model_name] == "gray_rgb_clahe" else gray_rgb
        with torch.inference_mode():
            output = model(sample)
        if output.shape != control_logits.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"Invalid forward output for {model_name}")
        difference = float((output - control_logits).abs().max())
        if model_name in BASELINE_PRESERVING and difference > 1e-6:
            raise RuntimeError(
                f"Baseline-preserving initialization failed for {model_name}: {difference}"
            )
        if model_name == "rgb_1x1_projection_unet":
            expected = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            if not torch.allclose(model.rgb_projection.weight.detach().cpu(), expected):
                raise RuntimeError("The generic 1x1 projection has the wrong initialization")
            if not torch.equal(
                model.rgb_projection.bias.detach().cpu(), torch.zeros(1)
            ):
                raise RuntimeError("The generic 1x1 projection bias is not zero")
        print(
            f"  {model_name}: strict backbone load passed; "
            f"new_tensors={len(missing)}; initial_max_delta={difference:.9g}"
        )
        del model
        gc.collect()
    del control


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    check_cuda(args.require_cuda)
    check_clahe()

    train_pairs = discover_pairs(data_root, "FIVES", "train")
    if len(train_pairs) != 600:
        raise RuntimeError(f"Expected 600 prepared FIVES training pairs, found {len(train_pairs)}")
    available_keys = {pair[2] for pair in train_pairs}
    clahe_dataset = RetinalDataset(
        train_pairs[:1], image_size=128, input_mode="gray_rgb_clahe", augment=False
    )
    sample, _, _ = clahe_dataset[0]
    if sample.shape != (5, 128, 128):
        raise RuntimeError(f"Expected five-channel CLAHE representation, found {sample.shape}")
    sample = sample.unsqueeze(0)

    checkpoint_hashes = {}
    for seed in SEEDS:
        run_dir = project_root / "runs_fives_v2_screen" / "FIVES" / "res_attention_unet_gn" / f"seed_{seed}"
        split_path = run_dir / "split.json"
        checkpoint_path = (
            project_root
            / "weight_files_fives_v2_screen"
            / f"FIVES_best_ResAttentionUNetGN_seed{seed}.pth"
        )
        for path in (split_path, checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required reference file not found: {path}")
        check_split(split_path, available_keys)
        checkpoint = load_torch_checkpoint(checkpoint_path)
        check_models(checkpoint, sample)
        checkpoint_hashes[str(seed)] = file_sha256(checkpoint_path)
        print(f"Seed {seed}: split and GN checkpoint passed ({checkpoint_hashes[str(seed)][:12]}...).")

    print("FIVES color-control preflight passed.")
    print("Official FIVES test files were not discovered or opened by this preflight.")


if __name__ == "__main__":
    main()
