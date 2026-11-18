# -*- coding: utf-8 -*-
"""
Task A+ 补充: 经典非负稀疏 L1 2D ILT（FISTA-L1 与 Guo 2019 风格两阶段 L1 + 活性集非负最小二乘）

严格预注册口径:
- 全流程纯 CPU 单线程，严禁触发 CUDA;
- 正则比例系数 α 在独立 dev 集 (seed=20261112, n=100) 上择优后冻结;
- 在与表 1 完全一致的留出 test 集 (rng=20261112+100, n=200) 与 13 真实样本上评测;
- 全指标（Pearson r, T1/T2 峰位误差 dex, 单线程 CPU 耗时）均在线性域单位 log-面积归一化谱上计算。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import json
import glob
import time
import numpy as np
import torch
torch.set_num_threads(1)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = HERE
sys.path.insert(0, PROTO)

from sim2d import TI_MS, NECH, TE_MS, N_T1, N_T2, T1_GRID, T2_GRID, make_sample
try:
    from realdata2d import LIB, load_sample, phase_correct
except ImportError:
    LIB = os.environ.get("NMR23_DATA_DIR", "")
    load_sample = None
    phase_correct = None
from train2d_v2 import log_pool_np
from adaptive_lambda_baseline import (
    KroneckerSVDCache,
    normalize_area,
    compute_metrics,
    marginal_peaks_ms,
    load_v2_cpu,
    base_for_cpu,
    reconstruct_brd_fixed,
)


def summarize_metrics(rows):
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "pearson_mean": round(float(arr[:, 0].mean()), 4),
        "pearson_std": round(float(arr[:, 0].std()), 4),
        "pearson_median": round(float(np.median(arr[:, 0])), 4),
        "t1_peak_err_dex_mean": round(float(arr[:, 1].mean()), 4),
        "t1_peak_err_dex_median": round(float(np.median(arr[:, 1])), 4),
        "t2_peak_err_dex_mean": round(float(arr[:, 2].mean()), 4),
        "t2_peak_err_dex_median": round(float(np.median(arr[:, 2])), 4),
    }


def build_sparse_ctx(ti_ms, nech, tol=1e-6):
    ti_arr = np.asarray(ti_ms, dtype=np.float64)
    echo_ms = np.arange(1, int(nech) + 1, dtype=np.float64) * float(TE_MS)
    K1 = 1.0 - 2.0 * np.exp(-ti_arr[:, None] / T1_GRID[None, :])
    K2 = np.exp(-echo_ms[:, None] / T2_GRID[None, :])
    U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
    U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
    r1 = int((S1 > S1[0] * tol).sum())
    r2 = int((S2 > S2[0] * tol).sum())
    A1 = np.ascontiguousarray(S1[:r1, None] * V1t[:r1], dtype=np.float64)
    A2t = np.ascontiguousarray((S2[:r2, None] * V2t[:r2]).T, dtype=np.float64)
    L = float((S1[0] * S2[0]) ** 2)
    return {
        "U1": U1[:, :r1],
        "U2": U2[:, :r2],
        "A1": A1,
        "A1t": np.ascontiguousarray(A1.T),
        "A2t": A2t,
        "A2": np.ascontiguousarray(A2t.T),
        "L": L,
        "r1": r1,
        "r2": r2,
    }


def _fista_nn_l1(Y_real, ctx, alpha_ratio=1e-3, max_iter=500, tol=1e-5, support_mask=None):
    """
    求解 min_{M >= 0} 0.5 ||A1 M A2^T - Ycore||_F^2 + lam1 * sum(M)
    其中 lam1 = alpha_ratio * max(|A^T Ycore|), 使归一化解对输入尺度严格不变。
    """
    Y = np.asarray(Y_real, dtype=np.float64)
    scale = float(np.max(np.abs(Y)))
    if scale <= 1e-12:
        return np.zeros((N_T1, N_T2), dtype=np.float64), 0
    Ys = Y / scale
    U1, U2 = ctx["U1"], ctx["U2"]
    A1, A1t, A2t, A2 = ctx["A1"], ctx["A1t"], ctx["A2t"], ctx["A2"]
    invL = 1.0 / ctx["L"]

    Ycore = U1.T @ Ys @ U2
    AtYc = A1t @ Ycore @ A2
    lam1 = float(alpha_ratio) * float(np.max(np.abs(AtYc))) if alpha_ratio > 0.0 else 0.0
    step_thr = lam1 * invL

    M = np.zeros((N_T1, N_T2), dtype=np.float64)
    Z = np.zeros((N_T1, N_T2), dtype=np.float64)
    t = 1.0
    n_it = 0

    for it in range(1, max_iter + 1):
        n_it = it
        res = (A1 @ Z @ A2t) - Ycore
        grad = A1t @ res @ A2
        M_next = np.maximum(0.0, Z - invL * grad - step_thr)
        if support_mask is not None:
            M_next = M_next * support_mask

        diff = M_next - M
        rel_chg = float(np.linalg.norm(diff)) / max(float(np.linalg.norm(M_next)), 1e-12)
        if it > 20 and rel_chg < tol:
            M = M_next
            break

        # O'Donoghue-Candes adaptive restart
        if float(np.sum((Z - M_next) * diff)) > 0.0:
            t_next = 1.0
            Z = M_next.copy()
        else:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            Z = M_next + ((t - 1.0) / t_next) * diff
        M = M_next
        t = t_next

    return M, n_it


def reconstruct_fista_l1(Y_real, ctx, alpha_ratio):
    M_raw, n_it = _fista_nn_l1(Y_real, ctx, alpha_ratio=alpha_ratio, max_iter=500)
    return normalize_area(M_raw), n_it


def reconstruct_guo_twostage(Y_real, ctx, alpha_ratio, lam2_ratio=0.0):
    """
    Guo et al. (JMR 2019, Eq. 7-8) 双目标方法在 Kronecker 压缩核上的实现:
    Stage 1: 非负 L1-FISTA 提取稀疏支撑集;
    Stage 2: 固定非零支撑集求解非负最小二乘 (Eq. 8) 去偏。
    """
    M1, n1 = _fista_nn_l1(Y_real, ctx, alpha_ratio=alpha_ratio, max_iter=450)
    mmax = float(np.max(M1))
    if mmax <= 1e-12:
        return normalize_area(M1), n1
    supp = (M1 > 1e-4 * mmax).astype(np.float64)
    M2, n2 = _fista_nn_l1(
        Y_real, ctx, alpha_ratio=lam2_ratio, max_iter=200, support_mask=supp
    )
    if float(np.sum(M2)) <= 1e-12:
        return normalize_area(M1), n1
    return normalize_area(M2), n1 + n2


def main():
    t0 = time.perf_counter()
    ctx = build_sparse_ctx(TI_MS, NECH, tol=1e-6)
    print(f"Sparse context (CPU 1-thread): r1={ctx['r1']}, r2={ctx['r2']}, L={ctx['L']:.3e}", flush=True)

    # 1. 尺度不变性核查
    rng_chk = np.random.default_rng(77)
    s_chk = make_sample(rng_chk, snr=80)
    y_chk = np.real(s_chk["y"])
    m_a, _ = reconstruct_fista_l1(y_chk, ctx, alpha_ratio=1e-3)
    m_b, _ = reconstruct_fista_l1(y_chk * 409.0, ctx, alpha_ratio=1e-3)
    scale_diff = float(np.max(np.abs(m_a - m_b)) / (np.max(np.abs(m_a)) + 1e-12))
    print(f"Scale invariance check (x409): max rel diff = {scale_diff:.3e}", flush=True)

    # 2. 独立 dev 集择优 (n=100, seed=20261112, 与 test 完全隔离)
    rng_dev = np.random.default_rng(20261112)
    dev_samples = [make_sample(rng_dev) for _ in range(100)]
    alpha_grid = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

    dev_fista, dev_guo = {}, {}
    best_a_fista, best_r_fista = None, -1.0
    best_a_guo, best_r_guo = None, -1.0

    print("\n=== Dev Selection (n=100, seed=20261112) ===", flush=True)
    for a in alpha_grid:
        rs_f, rs_g = [], []
        for s in dev_samples:
            y = np.real(s["y"])
            mf, _ = reconstruct_fista_l1(y, ctx, alpha_ratio=a)
            mg, _ = reconstruct_guo_twostage(y, ctx, alpha_ratio=a, lam2_ratio=0.0)
            rf, _, _ = compute_metrics(mf, s["spec"])
            rg, _, _ = compute_metrics(mg, s["spec"])
            rs_f.append(rf)
            rs_g.append(rg)
        mf_mean = float(np.mean(rs_f))
        mg_mean = float(np.mean(rs_g))
        dev_fista[f"{a:.0e}"] = round(mf_mean, 4)
        dev_guo[f"{a:.0e}"] = round(mg_mean, 4)
        print(f"  alpha={a:.0e}: fista_l1 r={mf_mean:.4f} | guo_twostage r={mg_mean:.4f}", flush=True)
        if mf_mean > best_r_fista:
            best_r_fista, best_a_fista = mf_mean, a
        if mg_mean > best_r_guo:
            best_r_guo, best_a_guo = mg_mean, a

    print(f"Selected on Dev -> fista_l1 alpha={best_a_fista:.0e} (r={best_r_fista:.4f}), "
          f"guo_twostage alpha={best_a_guo:.0e} (r={best_r_guo:.4f})", flush=True)

    # 3. 留出 test 集正式评测 (n=200, rng=20261112+100)
    rng_test = np.random.default_rng(20261112 + 100)
    test_samples = [make_sample(rng_test) for _ in range(200)]

    res_fista, res_guo = [], []
    iters_f, iters_g = [], []

    t_f0 = time.perf_counter()
    for s in test_samples:
        mf, nf = reconstruct_fista_l1(np.real(s["y"]), ctx, alpha_ratio=best_a_fista)
        res_fista.append(compute_metrics(mf, s["spec"]))
        iters_f.append(nf)
    ms_fista = (time.perf_counter() - t_f0) * 1000.0 / len(test_samples)

    t_g0 = time.perf_counter()
    for s in test_samples:
        mg, ng = reconstruct_guo_twostage(np.real(s["y"]), ctx, alpha_ratio=best_a_guo, lam2_ratio=0.0)
        res_guo.append(compute_metrics(mg, s["spec"]))
        iters_g.append(ng)
    ms_guo = (time.perf_counter() - t_g0) * 1000.0 / len(test_samples)

    sum_fista = summarize_metrics(res_fista)
    sum_fista["cpu_ms_per_sample"] = round(ms_fista, 2)
    sum_fista["frozen_alpha"] = best_a_fista
    sum_fista["mean_iters"] = round(float(np.mean(iters_f)), 1)

    sum_guo = summarize_metrics(res_guo)
    sum_guo["cpu_ms_per_sample"] = round(ms_guo, 2)
    sum_guo["frozen_alpha"] = best_a_guo
    sum_guo["mean_iters"] = round(float(np.mean(iters_g)), 1)

    print("\n=== Test Evaluation (n=200, rng=20261112+100) ===", flush=True)
    print(f"  fista_l1    : r={sum_fista['pearson_mean']:.4f}±{sum_fista['pearson_std']:.4f} "
          f"(med {sum_fista['pearson_median']:.4f}) | T1={sum_fista['t1_peak_err_dex_mean']:.4f}/{sum_fista['t1_peak_err_dex_median']:.4f} "
          f"| T2={sum_fista['t2_peak_err_dex_mean']:.4f}/{sum_fista['t2_peak_err_dex_median']:.4f} | {ms_fista:.2f} ms", flush=True)
    print(f"  guo_2stage  : r={sum_guo['pearson_mean']:.4f}±{sum_guo['pearson_std']:.4f} "
          f"(med {sum_guo['pearson_median']:.4f}) | T1={sum_guo['t1_peak_err_dex_mean']:.4f}/{sum_guo['t1_peak_err_dex_median']:.4f} "
          f"| T2={sum_guo['t2_peak_err_dex_mean']:.4f}/{sum_guo['t2_peak_err_dex_median']:.4f} | {ms_guo:.2f} ms", flush=True)

    # 4. 13 真实样本零样本对比
    cache = KroneckerSVDCache()
    front, refine = load_v2_cpu()
    real_rows = []
    dirs = sorted(glob.glob(os.path.join(LIB, "*__IR-CPMG"))) if (load_sample and LIB and os.path.isdir(LIB)) else []
    if not dirs:
        print("Restricted 23 MHz raw waveforms not attached (set NMR23_DATA_DIR or inspect results_archive/sparse_l1_baseline.json).", flush=True)
        return
    for d in dirs:
        s = load_sample(d)
        Yr = np.real(phase_correct(s["Yc"]))
        ys = Yr / (np.abs(Yr).max() + 1e-9)
        ti = s["ti"]
        nech = Yr.shape[1]
        ctx_r = build_sparse_ctx(ti, nech, tol=1e-6)
        ctx_base = cache.get(ti, nech)

        yt = torch.tensor(ys[None], dtype=torch.float32)
        with torch.no_grad():
            base_log, base_lin = base_for_cpu(ti, yt)
            pooled = torch.tensor(
                np.nan_to_num(log_pool_np(ys[None]), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            )
            pred_log = refine(base_log, pooled)[0].numpy()
        M_v2 = normalize_area(np.expm1(np.clip(pred_log, 0.0, 20.0)) / 20.0)
        M_base = normalize_area(base_lin)
        M_brd = reconstruct_brd_fixed(ys, ctx_base["brd"], lam=1e-3)
        M_fista, _ = reconstruct_fista_l1(ys, ctx_r, alpha_ratio=best_a_fista)
        M_guo, _ = reconstruct_guo_twostage(ys, ctx_r, alpha_ratio=best_a_guo, lam2_ratio=0.0)

        t1_f, t2_f = marginal_peaks_ms(M_fista)
        t1_g, t2_g = marginal_peaks_ms(M_guo)
        t1_v, t2_v = marginal_peaks_ms(M_v2)

        row = {
            "name": s["name"],
            "nti": int(Yr.shape[0]),
            "nech": int(nech),
            "r_v2_vs_fista_l1": round(float(np.corrcoef(M_v2.ravel(), M_fista.ravel())[0, 1]), 4),
            "r_v2_vs_guo_2stage": round(float(np.corrcoef(M_v2.ravel(), M_guo.ravel())[0, 1]), 4),
            "r_brd_vs_fista_l1": round(float(np.corrcoef(M_brd.ravel(), M_fista.ravel())[0, 1]), 4),
            "r_base_vs_fista_l1": round(float(np.corrcoef(M_base.ravel(), M_fista.ravel())[0, 1]), 4),
            "fista_T1": round(t1_f, 1), "fista_T2": round(t2_f, 1),
            "guo_T1": round(t1_g, 1), "guo_T2": round(t2_g, 1),
            "v2_T1": round(t1_v, 1), "v2_T2": round(t2_v, 1),
        }
        real_rows.append(row)
        print(f"  {row['name']:<12} | r(v2,fista)={row['r_v2_vs_fista_l1']:.4f} r(v2,guo)={row['r_v2_vs_guo_2stage']:.4f} "
              f"| fista={t1_f:.1f}/{t2_f:.1f} vs v2={t1_v:.1f}/{t2_v:.1f}", flush=True)

    mean_v2_fista = float(np.mean([r["r_v2_vs_fista_l1"] for r in real_rows]))
    mean_brd_fista = float(np.mean([r["r_brd_vs_fista_l1"] for r in real_rows]))
    mean_v2_guo = float(np.mean([r["r_v2_vs_guo_2stage"] for r in real_rows]))
    mean_base_fista = float(np.mean([r["r_base_vs_fista_l1"] for r in real_rows]))

    out = {
        "protocol": {
            "dev_split": "sim2d seed=20261112, n=100",
            "test_split": "sim2d rng=20261112+100, n=200",
            "scale_invariance_x409_max_rel_diff": scale_diff,
            "dev_fista_scores": dev_fista,
            "dev_guo_scores": dev_guo,
            "wall_s": round(time.perf_counter() - t0, 2),
        },
        "sim_test_n200": {
            "fista_l1_fixed": sum_fista,
            "guo_twostage_l1": sum_guo,
        },
        "real_n13": {
            "mean_r_v2_vs_fista_l1": round(mean_v2_fista, 4),
            "mean_r_v2_vs_guo_2stage": round(mean_v2_guo, 4),
            "mean_r_brd_vs_fista_l1": round(mean_brd_fista, 4),
            "mean_r_base_vs_fista_l1": round(mean_base_fista, 4),
            "samples": real_rows,
        },
    }
    out_path = os.path.join(HERE, "results_archive", "sparse_l1_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path} in {out['protocol']['wall_s']}s | "
          f"mean real r(v2, fista)={mean_v2_fista:.4f}, r(v2, guo)={mean_v2_guo:.4f}", flush=True)


if __name__ == "__main__":
    main()
