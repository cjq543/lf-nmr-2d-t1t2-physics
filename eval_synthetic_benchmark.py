# -*- coding: utf-8 -*-
"""
Offline single-thread CPU verification and synthetic benchmark for Paper #2:
"Physics-Front-End Deep Learning for 2D T1-T2 Relaxation Correlation Spectroscopy
 on a Compact Low-Field NMR Analyzer"

Verifies:
1. Exact byte-level MD5/SHA-256 integrity of untouched frozen checkpoint `v2_refine.pth`.
2. Parameter-count parity across neural architectures (`RefineNet`: 19,393;
   `Inv2DNet`: 2,663,777; `DLEMLR SimpleViTGrid`: 73,548,416; zero learned weights in `PhysicsFrontEnd`).
3. Full held-out 200-sample synthetic benchmark (`rng=20261112+100`) comparing frozen `v2`
   against `baseline2d.reconstruct_2d(lam=1e-3)` on linear area-normalized spectra.
4. Spot execution check across adaptive SVD-GCV, Song (2002) BRD-NNLS, and FISTA-L1 solvers.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch

torch.set_num_threads(1)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from sim2d import TI_MS, NECH, T1_GRID, T2_GRID, DL1, DL2, make_sample
from baseline2d import reconstruct_2d
from train2d_v2 import PhysicsFrontEnd, RefineNet, log_pool_np
from train2d import Inv2DNet
from dlemlr_model import SimpleViTGrid
from adaptive_lambda_baseline import (
    KroneckerSVDCache,
    compute_metrics,
    reconstruct_svd_ridge_gcv,
    reconstruct_brd_fixed,
)
from sparse_l1_baseline import build_sparse_ctx, reconstruct_fista_l1


EXPECTED_WEIGHT_SIZE = 80768
EXPECTED_WEIGHT_MD5 = "804d0c5a3d1186f8782102c19398b209"
EXPECTED_WEIGHT_SHA256 = "b818ce5f21e414509102bb36f0640de91efa384f67ff05c7084defcfa7024ec8"


def check_weight_checksum(pth_path):
    data = open(pth_path, "rb").read()
    md5 = hashlib.md5(data).hexdigest()
    sha256 = hashlib.sha256(data).hexdigest()
    assert len(data) == EXPECTED_WEIGHT_SIZE, f"Unexpected size: {len(data)}"
    assert md5 == EXPECTED_WEIGHT_MD5, f"MD5 mismatch: {md5}"
    assert sha256 == EXPECTED_WEIGHT_SHA256, f"SHA-256 mismatch: {sha256}"
    return {"bytes": len(data), "md5": md5, "sha256": sha256}


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-json", default="", help="Optional path to write summary JSON")
    args = parser.parse_args()

    t_start = time.perf_counter()
    pth_path = os.path.join(HERE, "v2_refine.pth")
    weight_info = check_weight_checksum(pth_path)
    print(
        f"[1/4] Frozen weight OK: v2_refine.pth ({weight_info['bytes']} B, md5={weight_info['md5']})",
        flush=True,
    )

    front = PhysicsFrontEnd(lam=1e-3).eval()
    refine = RefineNet().eval()
    ck = torch.load(pth_path, map_location="cpu", weights_only=True)
    refine.load_state_dict(ck["refine"])

    inv_net = Inv2DNet().eval()
    vit_net = SimpleViTGrid().eval()
    params = {
        "PhysicsFrontEnd_trainable": count_trainable_params(front),
        "RefineNet_v2": count_trainable_params(refine),
        "Inv2DNet_v1": count_trainable_params(inv_net),
        "SimpleViTGrid_DLEMLR": count_trainable_params(vit_net),
    }
    assert params["PhysicsFrontEnd_trainable"] == 0
    assert params["RefineNet_v2"] == 19393
    assert params["Inv2DNet_v1"] == 2663777
    assert params["SimpleViTGrid_DLEMLR"] == 73548416
    print(f"[2/4] Model parameter parity verified: {params}", flush=True)

    # Hold-out 200-sample evaluation matching train2d_v2.py & e2b_peak_err_table1.py
    rng_test = np.random.default_rng(20261112 + 100)
    samples = [make_sample(rng_test) for _ in range(200)]

    # Warmup once outside timing
    y0 = np.real(samples[0]["y"]) / (np.abs(np.real(samples[0]["y"])).max() + 1e-9)
    with torch.no_grad():
        b0 = front(torch.tensor(y0[None], dtype=torch.float32))
        p0 = torch.tensor(log_pool_np(y0[None]), dtype=torch.float32)
        _ = refine(b0, p0)
    _ = reconstruct_2d(samples[0]["y"], samples[0]["mask"], lam=1e-3)

    v2_metrics, base_metrics = [], []
    t_v2 = 0.0
    t_base = 0.0

    for s in samples:
        yr = np.real(s["y"])
        ys = yr / (np.abs(yr).max() + 1e-9)

        t0 = time.perf_counter()
        yt = torch.tensor(ys[None], dtype=torch.float32)
        with torch.no_grad():
            base_log = front(yt)
            pooled = torch.tensor(log_pool_np(ys[None]), dtype=torch.float32)
            pred_log = refine(base_log, pooled)[0].numpy()
        pred = np.expm1(np.clip(pred_log, 0.0, 20.0)) / 20.0
        t_v2 += time.perf_counter() - t0

        t1 = time.perf_counter()
        base_m = reconstruct_2d(s["y"], s["mask"], lam=1e-3)
        t_base += time.perf_counter() - t1

        v2_metrics.append(compute_metrics(pred, s["spec"]))
        base_metrics.append(compute_metrics(base_m, s["spec"]))

    v2_arr = np.asarray(v2_metrics, dtype=np.float64)
    base_arr = np.asarray(base_metrics, dtype=np.float64)
    diff_r = v2_arr[:, 0] - base_arr[:, 0]

    # Compare against archived seed3 (checkpoint overwritten cleanly on seed 3) & summary JSONs
    v2_arch = json.load(open(os.path.join(HERE, "results_archive", "v2_summary.json"), encoding="utf-8"))
    t1_arch = json.load(open(os.path.join(HERE, "results_archive", "table1_peak_err.json"), encoding="utf-8"))

    v2_r_mean = float(v2_arr[:, 0].mean())
    base_r_mean = float(base_arr[:, 0].mean())
    assert abs(v2_r_mean - v2_arch["seed3"]["learned"]) < 1e-4, f"{v2_r_mean} vs {v2_arch['seed3']['learned']}"
    assert abs(base_r_mean - v2_arch["seed3"]["baseline"]) < 1e-4, f"{base_r_mean} vs {v2_arch['seed3']['baseline']}"
    assert float(np.median(v2_arr[:, 1])) == t1_arch["v2"]["t1_peak_err_dex_median"]
    assert float(np.median(v2_arr[:, 2])) == t1_arch["v2"]["t2_peak_err_dex_median"]

    summary = {
        "weight": weight_info,
        "params": params,
        "n_test": len(samples),
        "v2_seed3": {
            "pearson_mean": round(v2_r_mean, 4),
            "pearson_std": round(float(v2_arr[:, 0].std()), 4),
            "pearson_median": round(float(np.median(v2_arr[:, 0])), 4),
            "t1_err_dex_mean": round(float(v2_arr[:, 1].mean()), 4),
            "t1_err_dex_median": round(float(np.median(v2_arr[:, 1])), 4),
            "t2_err_dex_mean": round(float(v2_arr[:, 2].mean()), 4),
            "t2_err_dex_median": round(float(np.median(v2_arr[:, 2])), 4),
            "cpu_ms_per_sample": round(t_v2 * 1000.0 / len(samples), 2),
        },
        "svd_ridge_baseline": {
            "pearson_mean": round(base_r_mean, 4),
            "pearson_std": round(float(base_arr[:, 0].std()), 4),
            "pearson_median": round(float(np.median(base_arr[:, 0])), 4),
            "t1_err_dex_mean": round(float(base_arr[:, 1].mean()), 4),
            "t1_err_dex_median": round(float(np.median(base_arr[:, 1])), 4),
            "t2_err_dex_mean": round(float(base_arr[:, 2].mean()), 4),
            "t2_err_dex_median": round(float(np.median(base_arr[:, 2])), 4),
            "cpu_ms_per_sample": round(t_base * 1000.0 / len(samples), 2),
        },
        "paired_v2_minus_base": {
            "mean_diff": round(float(diff_r.mean()), 4),
            "wins": int(np.sum(diff_r > 1e-12)),
            "losses": int(np.sum(diff_r < -1e-12)),
        },
        "three_seed_archive_learned_mean": round(float(v2_arch["summary"]["learned_mean"]), 4),
    }

    print(
        f"[3/4] Synthetic test(200) PASS | v2(seed3) r={summary['v2_seed3']['pearson_mean']:.4f}"
        f"±{summary['v2_seed3']['pearson_std']:.4f} (3-seed mean={summary['three_seed_archive_learned_mean']:.4f}), "
        f"med peak err=({summary['v2_seed3']['t1_err_dex_median']:.4f}, {summary['v2_seed3']['t2_err_dex_median']:.4f}) dex, "
        f"{summary['v2_seed3']['cpu_ms_per_sample']:.2f} ms/sample | "
        f"SVD-ridge r={summary['svd_ridge_baseline']['pearson_mean']:.4f} | "
        f"wins={summary['paired_v2_minus_base']['wins']}/200",
        flush=True,
    )

    # [4/4] Quick CPU sanity execution across classical solvers on 3 held-out samples
    cache = KroneckerSVDCache()
    ctx = cache.get(TI_MS, NECH)
    sp_ctx = build_sparse_ctx(TI_MS, NECH, tol=1e-6)
    spot = {}
    for name, fn in [
        ("svd_ridge_gcv", lambda yr: reconstruct_svd_ridge_gcv(yr, ctx["ridge"])[0]),
        ("brd_nnls_fixed", lambda yr: reconstruct_brd_fixed(yr, ctx["brd"], lam=1e-3)),
        ("fista_l1", lambda yr: reconstruct_fista_l1(yr, sp_ctx, alpha_ratio=1e-3)[0]),
    ]:
        rs = [compute_metrics(fn(np.real(samples[i]["y"])), samples[i]["spec"])[0] for i in range(3)]
        spot[name] = round(float(np.mean(rs)), 4)
        assert np.isfinite(spot[name]) and spot[name] > 0.5
    summary["classical_spotcheck_3samples_r"] = spot
    summary["wall_s"] = round(time.perf_counter() - t_start, 2)
    print(f"[4/4] Classical solvers 3-sample spotcheck PASS: {spot} (total {summary['wall_s']}s)", flush=True)

    if args.write_json:
        with open(args.write_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
