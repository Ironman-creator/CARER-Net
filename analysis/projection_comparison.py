"""Frozen-output structural comparison and separate local architecture benchmark.

No fitting, test-threshold search, checkpoint selection, or source-file changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/projection_structure"
SEEDS = (42, 123, 3407)
ARMS = {
    "projection": ROOT / "result_fives_color_controls/FIVES/rgb_1x1_projection_unet",
    "core": ROOT / "result_fives_carer_official_test/FIVES/color_detail_only_unet",
}
FIVES = ROOT / "data/FIVES_official/test"
STAGE1 = ROOT / "analysis/structure_metrics.py"
OLD = ROOT / "results/fives_structural/FIVES_逐图结构指标.csv"
STRUCTURE = ("cldice", "thin_centerline_recall", "branchpoint_f1", "boundary_f1_euclidean_2px")
REGION = ("dice", "sensitivity", "specificity")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def helper():
    spec = importlib.util.spec_from_file_location("frozen_stage1_helpers", STAGE1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory():
    records, sources = {}, set()
    for arm, directory in ARMS.items():
        for seed in SEEDS:
            path = directory / f"seed_{seed}/per_image_metrics.csv"
            raw = read_csv(path)
            indexed = {r["sample"].lower(): r for r in raw}
            assert len(raw) == len(indexed) == 200, (arm, seed)
            masks = {p.stem.lower(): p for p in (path.parent / "predictions").glob("*.png")}
            assert set(masks) == set(indexed), (arm, seed, "masks mismatch")
            records[(arm, seed)] = indexed
            sources.add(path)
            sources.update(masks.values())
    samples = sorted(records[("core", 42)])
    for indexed in records.values():
        assert set(indexed) == set(samples)
    for kind in ("Original", "Ground truth"):
        files = {p.stem.lower(): p for p in (FIVES / kind).iterdir() if p.is_file()}
        assert set(files) == set(samples), kind
        sources.update(files.values())
    sources.update((STAGE1, OLD))
    return records, samples, sorted(sources)


def preflight():
    _, samples, sources = inventory()
    protocol = {
        "date": "2026-10-16", "mode": "post-hoc frozen-prediction analysis; no new training",
        "arms": {k: str(v.resolve()) for k, v in ARMS.items()},
        "seeds": SEEDS, "n_official_test_images": len(samples), "image_size": 512,
        "predictions": "Existing 0.5-threshold binary masks; no ensemble and no threshold changes",
        "fov": "PIL RGB bilinear-resize 512, then PIL grayscale > 5/255",
        "metrics": list(STRUCTURE), "implementation": str(STAGE1.resolve()),
        "radius": "Ground-truth EDT radius <=1.5 pixels, not diameter",
        "boundary_and_branch_tolerance_pixels": 2,
        "bootstrap": {
            "replicates": 10000, "rng_seed": 20261016,
            "pointwise_confidence": 0.95, "resampling": "Crossed seed and image, independently; paired arms",
            "four_structure_family_confidence": 0.9875,
            "multiplicity": "Bonferroni intervals for four structural comparisons; still exploratory post-hoc",
            "additional": "95% image-only intervals after seed averaging; region metrics separate",
        },
        "runtime": {
            "hardware": "Local GPU, recorded by benchmark; never pooled with RTX 4070",
            "weights": "Random shared-backbone weights; architecture latency, not new accuracy inference",
            "batch_size": 1, "input_size": [1,4,512,512], "precision": "float32",
            "warmup_per_arm": 20, "blocks": 5, "timed_iterations_per_block": 20,
            "timing": "CUDA event and synchronized wall time, randomized arm order per block",
            "excluded": "Disk/image preprocessing, transfers and metric computation",
        },
        "python": sys.version, "platform": platform.platform(),
        "sample_ids": samples,
        "sources": [{"path": str(p.resolve()), "sha256": sha(p)} for p in sources],
    }
    path = OUT / "frozen_protocol.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        assert old["sources"] == protocol["sources"], "Frozen source hashes changed"
    else:
        save(path.name, protocol)
    print(f"PREFLIGHT: {len(samples)} images x 3 seeds x 2 arms; {len(sources)} source files hashed", flush=True)


def structure(workers):
    preflight()
    h = helper()
    records, samples, _ = inventory()
    originals = {p.stem.lower(): p for p in (FIVES/"Original").iterdir() if p.is_file()}
    labels = {p.stem.lower(): p for p in (FIVES/"Ground truth").iterdir() if p.is_file()}
    old = {(int(r["seed"]), r["sample"]): r for r in read_csv(OLD) if r["model"] == "color_detail_only_unet"}

    def sample_metrics(sample):
        truth = h.resize_mask(labels[sample], 512)
        rgb = h.resize_rgb(originals[sample], 512)
        fov = np.asarray(rgb.convert("L"), dtype=np.float32)/255.0 > 5.0/255.0
        truth &= fov
        skel = h.skeletonize_zhang_suen(truth)
        radius = ndi.distance_transform_edt(truth)
        rows = []
        for arm, directory in ARMS.items():
            for seed in SEEDS:
                path = directory/f"seed_{seed}/predictions/{sample}.png"
                with Image.open(path) as source:
                    raw = np.asarray(source.convert("L"))
                assert raw.shape == (512,512) and set(np.unique(raw)).issubset({0,255}), str(path)
                prediction = (raw > 127) & fov
                tp = int((prediction & truth).sum())
                fp = int((prediction & ~truth & fov).sum())
                fn = int((~prediction & truth).sum())
                tn = int((~prediction & ~truth & fov).sum())
                recorded = records[(arm,seed)][sample]
                for k,v in {"tp":tp,"fp":fp,"fn":fn,"tn":tn}.items():
                    assert int(recorded[k]) == v, (arm,seed,sample,k,v,recorded[k])
                metrics = h.structural_metrics(prediction,truth,fov,skel,radius,2.0,2.0,10)
                dice = 2*tp/(2*tp+fp+fn+1e-8)
                assert abs(dice-float(recorded["dice"])) < 1e-10
                max_old = 0.0
                if arm == "core":
                    max_old = max(abs(float(v)-float(old[(seed,sample)][k])) for k,v in metrics.items())
                    assert max_old < 1e-12, (sample,seed,"Core old/new mismatch",max_old)
                rows.append({"sample":sample,"arm":arm,"seed":seed,
                             **{k:float(recorded[k]) for k in REGION},
                             "tp":tp,"tn":tn,"fp":fp,"fn":fn,
                             **metrics,"core_old_structure_max_abs_difference":max_old})
        return rows

    rows = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for n, chunk in enumerate(executor.map(sample_metrics,samples),1):
            rows.extend(chunk)
            if n % 10 == 0:
                print(f"STRUCTURE: {n}/200 images, {len(rows)}/1200 masks, {time.perf_counter()-start:.1f}s",flush=True)
    write_csv("per_image_structure.csv",rows)
    index = {(r["arm"],r["seed"],r["sample"]):r for r in rows}
    summary, comparisons = [], []
    rng = np.random.default_rng(20261016)
    for metric in (*REGION,*STRUCTURE):
        arrays = {arm:np.array([[index[(arm,s,sample)][metric] for sample in samples] for s in SEEDS]) for arm in ARMS}
        for arm,a in arrays.items():
            seed_means = a.mean(axis=1)
            summary.append({"arm":arm,"metric":metric,"mean":float(a.mean()),
                            "seed_sd":float(seed_means.std(ddof=1)),
                            **{f"seed_{s}_mean":float(v) for s,v in zip(SEEDS,seed_means)}})
        d = arrays["core"]-arrays["projection"]
        crossed = h.crossed_bootstrap_distribution(d,10000,rng)
        image_d = d.mean(axis=0)
        image = image_d[rng.integers(0,200,(10000,200))].mean(axis=1)
        lo,hi = np.quantile(crossed,[.025,.975])
        il,ih = np.quantile(image,[.025,.975])
        row = {"metric":metric,"core_mean":float(arrays["core"].mean()),
               "projection_mean":float(arrays["projection"].mean()),"core_minus_projection":float(d.mean()),
               "crossed_95_low":float(lo),"crossed_95_high":float(hi),
               "image_only_95_low":float(il),"image_only_95_high":float(ih),
               "crossed_95_direction":"positive" if lo>0 else "negative" if hi<0 else "inconclusive",
               "bonferroni_4_structure_98_75_low":"","bonferroni_4_structure_98_75_high":"",
               **{f"seed_{s}_delta":float(v) for s,v in zip(SEEDS,d.mean(axis=1))},
               "positive_image_fraction_after_seed_mean":float((image_d>0).mean()),
               "analysis_role":"exploratory post-hoc"}
        if metric in STRUCTURE:
            bl,bh = np.quantile(crossed,[.00625,.99375])
            row.update(bonferroni_4_structure_98_75_low=float(bl),bonferroni_4_structure_98_75_high=float(bh))
        comparisons.append(row)
    write_csv("structure_arm_summary.csv",summary)
    write_csv("core_minus_projection_paired_statistics.csv",comparisons)
    protocol = json.loads((OUT/"frozen_protocol.json").read_text(encoding="utf-8"))
    changed = [r["path"] for r in protocol["sources"] if sha(ROOT/r["path"])!=r["sha256"]]
    assert not changed
    save("structure_integrity.json",{"source_hashes_unchanged":True,"n_sources":len(protocol["sources"]),
         "n_images":200,"n_masks":len(rows),"all_saved_confusion_counts_reproduced":True,
         "all_600_core_structure_rows_reproduced":True,"max_core_difference":max(r["core_old_structure_max_abs_difference"] for r in rows),
         "seconds":time.perf_counter()-start})
    print(json.dumps(comparisons,indent=2),flush=True)


def benchmark():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from thop import profile
    sys.path.insert(0,str(ROOT))
    from models.model_factory import build_model, count_parameters
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    names = {"gray":"gray_rgb_control_30_unet","projection":"rgb_1x1_projection_unet","core":"color_detail_only_unet"}
    sample_cpu = torch.rand(1,4,512,512,generator=torch.Generator().manual_seed(20261016))
    sample = sample_cpu.to("cuda")
    models, static = {}, {}
    for arm,name in names.items():
        torch.manual_seed(42)
        model = build_model(name,in_channels=4,out_channels=1).eval()
        # Profile a separate instance: THOP leaves hooks/buffers on its input model.
        torch.manual_seed(42)
        profiler = build_model(name,in_channels=4,out_channels=1).eval()
        macs,_ = profile(profiler,inputs=(sample_cpu,),verbose=False)
        del profiler
        static[arm] = {"parameters":count_parameters(model),"thop_macs":int(macs)}
        models[arm] = model.to("cuda")
    traces = []
    with torch.inference_mode():
        for arm,model in models.items():
            for _ in range(20):
                result = model(sample)
            assert torch.isfinite(result).all()
            del result
            torch.cuda.synchronize()
            print(f"BENCHMARK: {arm} warmup complete",flush=True)
        rng = np.random.default_rng(20261016)
        for block in range(5):
            for arm in rng.permutation(list(names)):
                model = models[arm]
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                memory_base = torch.cuda.memory_allocated()
                for iteration in range(20):
                    event_start,event_end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    wall_start = time.perf_counter()
                    event_start.record()
                    result = model(sample)
                    event_end.record()
                    torch.cuda.synchronize()
                    wall_ms = (time.perf_counter()-wall_start)*1000
                    assert torch.isfinite(result).all()
                    del result
                    traces.append({"arm":str(arm),"block":block+1,"iteration":iteration+1,
                                   "wall_ms":wall_ms,"cuda_ms":event_start.elapsed_time(event_end),
                                   "incremental_peak_allocated_bytes":torch.cuda.max_memory_allocated()-memory_base})
                print(f"BENCHMARK: block {block+1}/5 {arm} complete",flush=True)
    summary = []
    for arm in names:
        selected = [r for r in traces if r["arm"]==arm]
        wall = np.array([r["wall_ms"] for r in selected])
        gpu = np.array([r["cuda_ms"] for r in selected])
        summary.append({"arm":arm,**static[arm],"wall_ms_mean":float(wall.mean()),
                        "wall_ms_sd":float(wall.std(ddof=1)),"wall_ms_median":float(np.median(wall)),
                        "cuda_ms_mean":float(gpu.mean()),"cuda_ms_sd":float(gpu.std(ddof=1)),
                        "cuda_ms_median":float(np.median(gpu)),"timed_iterations":len(selected),
                        "max_incremental_peak_allocated_bytes":max(r["incremental_peak_allocated_bytes"] for r in selected),
                        "gpu":torch.cuda.get_device_name(0),"pytorch":torch.__version__,
                        "weights":"random shared-backbone seed=42", "precision":"FP32"})
    write_csv("local_gpu_latency_trace.csv",traces)
    write_csv("local_gpu_efficiency.csv",summary)
    save("local_gpu_efficiency.json",{"rows":summary,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
         "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic,
         "tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32,
         "full_torch_config":torch.__config__.show(),"note":"Architecture-only local GTX 1650 measurement; separate from prior RTX 4070 results"})
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","structure","benchmark"))
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--fives-test", type=Path, default=FIVES)
    parser.add_argument("--projection-dir", type=Path, default=ARMS["projection"])
    parser.add_argument("--core-dir", type=Path, default=ARMS["core"])
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    FIVES, OUT = args.fives_test.resolve(), args.output_dir.resolve()
    ARMS = {"projection": args.projection_dir.resolve(), "core": args.core_dir.resolve()}
    OUT.mkdir(parents=True, exist_ok=True)
    {"preflight":preflight,"benchmark":benchmark,"structure":lambda:structure(args.workers)}[args.mode]()
