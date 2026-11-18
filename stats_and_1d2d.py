# -*- coding: utf-8 -*-
"""
纯 CPU 统计加固与补充量化分析 (2026-11-14):
1. Table 3 (n=9 食用油脂肪酸关联) 精确全置换检验 (9!=362,880)、Bootstrap 95% CI 与留一法 (LOO) 稳健性;
2. 跨反演器化学关联一致性: 比较 v2、svd_ridge_fixed、brd_nnls_fixed、1D CPMG T2-only 提特征的一致性;
3. Step 4 1D vs 2D 油脂判别与留一法成分外推增益定量对比;
4. K1 / K2 核在多阈值 (1e-2 .. 1e-10) 下的精确奇异值与有效秩审计 (统一 10x14=140 与 13x34=442 口径)。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import glob
import json
import time
import itertools
import numpy as np
import torch
from scipy.stats import spearmanr, rankdata

torch.set_num_threads(1)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = HERE
CLASSICAL = HERE
sys.path.insert(0, PROTO)
sys.path.insert(0, CLASSICAL)

from sim2d import TI_MS, NECH, TE_MS, N_T1, N_T2, T1_GRID, T2_GRID, LOG_T1, LOG_T2, DL1, DL2
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
    load_v2_cpu,
    base_for_cpu,
    reconstruct_brd_fixed,
    _brd_dual_newton,
)

# 8 种来源: 刘乐 AMR 2026 Table 1; 芝麻油来源: Oboulbiga et al. 2023 (PMC10292629)
FAT_9 = {
    "菜籽油": (7.0, 63.0, 30.0),
    "大豆油": (14.0, 25.0, 61.0),
    "橄榄油": (14.0, 76.0, 10.0),
    "红花籽油": (9.0, 12.0, 79.0),
    "花生油": (17.0, 46.0, 32.0),
    "葵花籽油": (11.0, 20.0, 69.0),
    "玉米油": (13.0, 28.0, 59.0),
    "亚麻籽油": (9.0, 18.0, 73.0),
    "芝麻油": (15.52, 42.86, 41.32),
}

EN_MAP = {
    "亚麻籽油": "Flaxseed", "大豆油": "Soybean", "菜籽油": "Rapeseed",
    "葵花籽油": "Sunflower", "红花籽油": "Safflower", "玉米油": "Corn",
    "橄榄油": "Olive", "芝麻油": "Sesame", "花生油": "Peanut",
    "甘油": "Glycerol", "环氧大豆油": "Epox. soybean", "椰油胺": "Cocoamine",
    "乙酸松油酯": "Terpinyl acetate",
}


def fwhm_1d(p, grid_log):
    im = int(np.argmax(p))
    half = p[im] * 0.5
    l = im
    while l > 0 and p[l] > half:
        l -= 1
    r = im
    while r < len(p) - 1 and p[r] > half:
        r += 1
    return float(10.0 ** grid_log[r] - 10.0 ** grid_log[l])


def extract_features_exact(M):
    M = np.asarray(M, dtype=np.float64)
    p1 = M.sum(1)
    p2 = M.sum(0)
    t1_peak = float(10.0 ** LOG_T1[int(np.argmax(p1))])
    t2_peak = float(10.0 ** LOG_T2[int(np.argmax(p2))])
    w1 = fwhm_1d(p1, LOG_T1)
    w2 = fwhm_1d(p2, LOG_T2)
    j1, j2 = np.meshgrid(LOG_T1, LOG_T2, indexing="ij")
    w = M / (M.sum() + 1e-12)
    m1 = float((w * j1).sum())
    m2 = float((w * j2).sum())
    cov = float((w * (j1 - m1) * (j2 - m2)).sum())
    sd1 = float(np.sqrt((w * (j1 - m1) ** 2).sum()))
    sd2 = float(np.sqrt((w * (j2 - m2) ** 2).sum()))
    xcorr = cov / (sd1 * sd2 + 1e-12)
    f_short = float((M * (T2_GRID[None, :] < 60.0)).sum() * DL1 * DL2)
    f_long = float((M * (T2_GRID[None, :] > 300.0)).sum() * DL1 * DL2)
    return {
        "t1_peak": t1_peak, "t2_peak": t2_peak, "ratio": t1_peak / t2_peak,
        "t1_fwhm": w1, "t2_fwhm": w2,
        "mean_logT1": m1, "mean_logT2": m2,
        "t1t2_xcorr": xcorr, "frac_shortT2": f_short, "frac_longT2": f_long,
    }


def solve_1d_t2(y_last_ti, lam=1e-3):
    """用最长 TI 行（CPMG 衰减）做物理匹配的 1D T2 反演（SVD-ridge + 非负约束 BRD）"""
    y1d = np.asarray(y_last_ti, dtype=np.float64)
    y1d = y1d / (np.max(np.abs(y1d)) + 1e-12)
    echo_ms = np.arange(1, len(y1d) + 1, dtype=np.float64) * float(TE_MS)
    K2 = np.exp(-echo_ms[:, None] / T2_GRID[None, :])
    U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
    smax = float(S2[0])

    # 1D SVD-ridge
    r_r = int((S2 > smax * 1e-10).sum())
    yc_r = U2[:, :r_r].T @ y1d
    c_r = S2[:r_r] * yc_r / (S2[:r_r] ** 2 + (lam * smax) ** 2)
    f_ridge = np.clip(V2t[:r_r].T @ c_r, 0.0, None)
    f_ridge /= (f_ridge.sum() * DL2 + 1e-12)

    # 1D BRD dual Newton NNLS
    r_b = int((S2 > smax * 1e-6).sum())
    K_sub = (S2[:r_b, None] * V2t[:r_b])  # (r_b, 96)
    yc_b = U2[:, :r_b].T @ y1d
    alpha = (lam * smax) ** 2
    # Reuse small dual Newton directly
    eye_p = np.eye(r_b, dtype=np.float64)
    c = np.zeros(r_b, dtype=np.float64)
    y_norm = float(np.linalg.norm(yc_b)) + 1e-12
    for _ in range(45):
        ktc = K_sub.T @ c
        act = ktc > 0.0
        f = np.where(act, ktc, 0.0)
        g = alpha * c + K_sub @ f - yc_b
        if np.linalg.norm(g) <= 1e-8 * y_norm:
            break
        Ka = K_sub[:, act] if np.any(act) else np.zeros((r_b, 0))
        H = alpha * eye_p + (Ka @ Ka.T)
        step = -np.linalg.solve(0.5 * (H + H.T), g)
        phi0 = 0.5 * alpha * np.dot(c, c) + 0.5 * np.dot(f, f) - np.dot(c, yc_b)
        slope = np.dot(g, step)
        t = 1.0
        for _ls in range(25):
            cn = c + t * step
            fn = np.maximum(0.0, K_sub.T @ cn)
            phin = 0.5 * alpha * np.dot(cn, cn) + 0.5 * np.dot(fn, fn) - np.dot(cn, yc_b)
            if phin <= phi0 + 1e-4 * t * slope:
                c = cn
                break
            t *= 0.5
    f_brd = np.maximum(0.0, K_sub.T @ c)
    f_brd /= (f_brd.sum() * DL2 + 1e-12)

    def feat1d(p):
        t2p = float(10.0 ** LOG_T2[int(np.argmax(p))])
        w2 = fwhm_1d(p, LOG_T2)
        wt = p / (p.sum() + 1e-12)
        mlog = float((wt * LOG_T2).sum())
        slog = float(np.sqrt((wt * (LOG_T2 - mlog) ** 2).sum()))
        return {"t2_peak_1d": t2p, "t2_fwhm_1d": w2, "mean_logT2_1d": mlog, "std_logT2_1d": slog}

    return feat1d(f_ridge), feat1d(f_brd), f_ridge, f_brd


def exact_spearman_stats(x, y, all_perms_idx, boot_rng, n_boot=10000):
    """
    计算:
    - asymptotic Spearman rho & p (scipy)
    - 精确全置换 (9! = 362,880) 双侧 p-value (正确处理结值 rankdata)
    - LOO Spearman min / max (逐一剔除每个样本)
    - Bootstrap 95% 置信区间
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho, p_asym = spearmanr(x, y)

    rx = rankdata(x)
    ry = rankdata(y)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx_c ** 2) * np.sum(ry_c ** 2))) + 1e-12

    # 精确全排列 y 的秩（362,880 × 9）
    ry_perm = ry_c[all_perms_idx]
    perm_rhos = (ry_perm @ rx_c) / denom
    p_exact = float(np.mean(np.abs(perm_rhos) >= abs(float(rho)) - 1e-12))

    # LOO min/max
    loo_rhos = []
    n = len(x)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r_i, _ = spearmanr(x[mask], y[mask])
        loo_rhos.append(float(r_i))

    # Bootstrap CI (忽略零方差退化样本)
    boot_rhos = []
    idx_mat = boot_rng.integers(0, n, size=(n_boot, n))
    for b_idx in idx_mat:
        xb = x[b_idx]
        yb = y[b_idx]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            continue
        rb, _ = spearmanr(xb, yb)
        if np.isfinite(rb):
            boot_rhos.append(float(rb))
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    return {
        "rho": round(float(rho), 4),
        "p_asym": round(float(p_asym), 5),
        "p_exact_perm": round(float(p_exact), 5),
        "loo_min": round(float(np.min(loo_rhos)), 4),
        "loo_max": round(float(np.max(loo_rhos)), 4),
        "boot_ci95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
    }


def loo_ridge_predict(X, y, alpha=1.0):
    """严格折内标准化的 LOO 岭回归外推（无任何数据泄漏）"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    preds = np.zeros(n, dtype=np.float64)
    eye_d = np.eye(d, dtype=np.float64)
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(axis=0)
        sd = np.where(Xtr.std(axis=0) > 1e-12, Xtr.std(axis=0), 1.0)
        Xtr_z = (Xtr - mu) / sd
        Xte_z = (X[i] - mu) / sd
        y_mu = ytr.mean()
        w = np.linalg.solve(Xtr_z.T @ Xtr_z + alpha * eye_d, Xtr_z.T @ (ytr - y_mu))
        preds[i] = float(y_mu + Xte_z @ w)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    mae = float(np.mean(np.abs(preds - y)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((preds - y) ** 2) / ss_tot)
    rho_loo, p_loo = spearmanr(y, preds)
    return {
        "loo_rmse": round(rmse, 2),
        "loo_mae": round(mae, 2),
        "loo_r2": round(r2, 4),
        "loo_spearman": round(float(rho_loo), 4),
        "loo_p": round(float(p_loo), 5),
        "preds": [round(float(v), 2) for v in preds],
    }


def svd_rank_audit():
    TI_EVEN10 = TI_MS[::2]
    TI_ODD10 = TI_MS[1::2]
    echo2000 = np.arange(1, NECH + 1, dtype=np.float64) * float(TE_MS)
    echo600 = np.arange(1, 601, dtype=np.float64) * float(TE_MS)

    K1_20 = 1.0 - 2.0 * np.exp(-TI_MS[:, None] / T1_GRID[None, :])
    K1_e10 = 1.0 - 2.0 * np.exp(-TI_EVEN10[:, None] / T1_GRID[None, :])
    K1_o10 = 1.0 - 2.0 * np.exp(-TI_ODD10[:, None] / T1_GRID[None, :])
    K2_2000 = np.exp(-echo2000[:, None] / T2_GRID[None, :])
    K2_600 = np.exp(-echo600[:, None] / T2_GRID[None, :])

    cutoffs = [1e-2, 1e-3, 1e-4, 1e-6, 1e-10]
    res = {}
    for label, K in [
        ("K1_20x48", K1_20), ("K1_even10x48", K1_e10), ("K1_odd10x48", K1_o10),
        ("K2_2000x96", K2_2000), ("K2_600x96", K2_600),
    ]:
        sv = np.linalg.svd(K, compute_uv=False)
        svn = sv / sv[0]
        ranks = {f">{c:.0e}": int(np.sum(svn > c)) for c in cutoffs}
        res[label] = {
            "s0": round(float(sv[0]), 4),
            "cond_to_1e10": float(f"{sv[0] / sv[max(ranks['>1e-10'] - 1, 0)]:.3e}"),
            "ranks": ranks,
            "top_norm_sv": [float(f"{v:.4e}") for v in svn[:min(20, len(svn))]],
        }
    res["kron_product_dof"] = {
        "threshold_1e-4_approx_snr100": res["K1_20x48"]["ranks"][">1e-04"] * res["K2_2000x96"]["ranks"][">1e-04"],
        "threshold_1e-6_brd_subspace": res["K1_20x48"]["ranks"][">1e-06"] * res["K2_2000x96"]["ranks"][">1e-06"],
        "threshold_1e-10_svd_ridge": res["K1_20x48"]["ranks"][">1e-10"] * res["K2_2000x96"]["ranks"][">1e-10"],
        "kron_sv_product_gt_1e-4": int(np.sum(
            np.outer(
                np.linalg.svd(K1_20, compute_uv=False) / np.linalg.svd(K1_20, compute_uv=False)[0],
                np.linalg.svd(K2_2000, compute_uv=False) / np.linalg.svd(K2_2000, compute_uv=False)[0],
            ) > 1e-4
        )),
        "kron_sv_product_gt_1e-3": int(np.sum(
            np.outer(
                np.linalg.svd(K1_20, compute_uv=False) / np.linalg.svd(K1_20, compute_uv=False)[0],
                np.linalg.svd(K2_2000, compute_uv=False) / np.linalg.svd(K2_2000, compute_uv=False)[0],
            ) > 1e-3
        )),
    }
    return res


def main():
    t0 = time.perf_counter()
    rank_info = svd_rank_audit()
    print("=== Exact SVD Rank Audit ===", flush=True)
    for k, v in rank_info.items():
        if k == "kron_product_dof":
            print(f"  2D Kronecker DOF summary: {v}", flush=True)
        else:
            print(f"  {k:<14}: {v['ranks']}", flush=True)

    # 1. Reconstruct all 13 samples across v2, svd_ridge_fixed, brd_nnls_fixed, and 1D T2-only
    cache = KroneckerSVDCache()
    front, refine = load_v2_cpu()

    sample_feats = {}
    dirs = sorted(glob.glob(os.path.join(LIB, "*__IR-CPMG"))) if (load_sample and LIB and os.path.isdir(LIB)) else []
    if not dirs:
        print("Restricted 23 MHz raw waveforms not attached; SVD rank table computed above, full derived tables archived in results_archive/stats_and_1d2d.json.", flush=True)
        return
    for d in dirs:
        s = load_sample(d)
        Yr = np.real(phase_correct(s["Yc"]))
        ys = Yr / (np.abs(Yr).max() + 1e-9)
        ti = s["ti"]
        nech = Yr.shape[1]
        ctx = cache.get(ti, nech)

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
        M_brd = reconstruct_brd_fixed(ys, ctx["brd"], lam=1e-3)

        f_1d_ridge, f_1d_brd, _, _ = solve_1d_t2(Yr[-1], lam=1e-3)
        sample_feats[s["name"]] = {
            "en": EN_MAP.get(s["name"], s["name"]),
            "nti": int(len(ti)),
            "nech": int(nech),
            "v2": extract_features_exact(M_v2),
            "svd_ridge_fixed": extract_features_exact(M_base),
            "brd_nnls_fixed": extract_features_exact(M_brd),
            "1d_svd_ridge": f_1d_ridge,
            "1d_nnls_brd": f_1d_brd,
        }

    # 2. Exact Permutation + Bootstrap CI + LOO across all solvers on n=9 oils
    oils = sorted(list(FAT_9.keys()))
    all_perms_idx = np.array(list(itertools.permutations(range(len(oils)))), dtype=np.int32)
    boot_rng = np.random.default_rng(20261114)

    pufa = np.array([FAT_9[k][2] for k in oils], dtype=np.float64)
    sfa = np.array([FAT_9[k][0] for k in oils], dtype=np.float64)

    pairs_2d = ["t2_fwhm", "t1_peak", "ratio", "t2_peak", "t1_fwhm", "mean_logT1", "mean_logT2"]
    chem_stats = {}

    print("\n=== Table 3 Hardening (n=9, Exact 9!=362,880 Permutations, LOO, Bootstrap CI) ===", flush=True)
    for solver in ["v2", "svd_ridge_fixed", "brd_nnls_fixed"]:
        chem_stats[solver] = {}
        print(f"\n  [{solver}]", flush=True)
        for fk in pairs_2d:
            vals = np.array([sample_feats[k][solver][fk] for k in oils], dtype=np.float64)
            st_pufa = exact_spearman_stats(vals, pufa, all_perms_idx, boot_rng)
            st_sfa = exact_spearman_stats(vals, sfa, all_perms_idx, boot_rng)
            chem_stats[solver][fk] = {"vs_PUFA": st_pufa, "vs_SFA": st_sfa}
            if fk in ("t2_fwhm", "t1_peak", "ratio", "t2_peak", "t1_fwhm"):
                print(
                    f"    {fk:<9}: PUFA rho={st_pufa['rho']:+.4f} (p_asym={st_pufa['p_asym']:.4f}, "
                    f"p_perm={st_pufa['p_exact_perm']:.4f}, LOO=[{st_pufa['loo_min']:+.3f},{st_pufa['loo_max']:+.3f}], "
                    f"CI={st_pufa['boot_ci95']}) | "
                    f"SFA rho={st_sfa['rho']:+.4f} (p_perm={st_sfa['p_exact_perm']:.4f}, "
                    f"LOO=[{st_sfa['loo_min']:+.3f},{st_sfa['loo_max']:+.3f}])",
                    flush=True,
                )

    for solver1d in ["1d_svd_ridge", "1d_nnls_brd"]:
        chem_stats[solver1d] = {}
        print(f"\n  [{solver1d}]", flush=True)
        for fk in ["t2_peak_1d", "t2_fwhm_1d", "mean_logT2_1d"]:
            vals = np.array([sample_feats[k][solver1d][fk] for k in oils], dtype=np.float64)
            st_pufa = exact_spearman_stats(vals, pufa, all_perms_idx, boot_rng)
            st_sfa = exact_spearman_stats(vals, sfa, all_perms_idx, boot_rng)
            chem_stats[solver1d][fk] = {"vs_PUFA": st_pufa, "vs_SFA": st_sfa}
            print(
                f"    {fk:<13}: PUFA rho={st_pufa['rho']:+.4f} (p_perm={st_pufa['p_exact_perm']:.4f}) | "
                f"SFA rho={st_sfa['rho']:+.4f} (p_perm={st_sfa['p_exact_perm']:.4f})",
                flush=True,
            )

    # 3. Step 4: 1D vs 2D Discrimination & Out-of-Sample LOO Composition Prediction
    # 3a. Unique peak collisions across 9 oils (using v2 vs 1D_nnls_brd)
    t2_1d_bins = {}
    t1t2_v2_bins = {}
    for k in oils:
        b1 = round(sample_feats[k]["1d_svd_ridge"]["t2_peak_1d"], 1)
        t2_1d_bins.setdefault(b1, []).append(sample_feats[k]["en"])
        b2 = (round(sample_feats[k]["v2"]["t1_peak"], 1), round(sample_feats[k]["v2"]["t2_peak"], 1))
        t1t2_v2_bins.setdefault(b2, []).append(sample_feats[k]["en"])

    # 3b. LOO out-of-sample regression predicting PUFA and SFA
    X_1d_peak = np.array([[np.log10(sample_feats[k]["1d_svd_ridge"]["t2_peak_1d"])] for k in oils])
    X_1d_shape = np.array([
        [sample_feats[k]["1d_svd_ridge"]["mean_logT2_1d"], np.log10(sample_feats[k]["1d_svd_ridge"]["t2_fwhm_1d"])]
        for k in oils
    ])
    X_1d_brd_shape = np.array([
        [sample_feats[k]["1d_nnls_brd"]["mean_logT2_1d"], np.log10(sample_feats[k]["1d_nnls_brd"]["t2_fwhm_1d"])]
        for k in oils
    ])
    X_2d_t1_only = np.array([
        [np.log10(sample_feats[k]["v2"]["t1_peak"]), np.log10(sample_feats[k]["v2"]["t1_fwhm"])]
        for k in oils
    ])
    X_2d_joint = np.array([
        [np.log10(sample_feats[k]["v2"]["t1_peak"]), np.log10(sample_feats[k]["v2"]["t2_fwhm"])]
        for k in oils
    ])
    X_2d_full3 = np.array([
        [np.log10(sample_feats[k]["v2"]["t1_peak"]),
         np.log10(sample_feats[k]["v2"]["t1_fwhm"]),
         np.log10(sample_feats[k]["v2"]["t2_fwhm"])]
        for k in oils
    ])

    feature_sets = {
        "1D_CPMG_T2_peak": X_1d_peak,
        "1D_CPMG_T2_shape_ridge": X_1d_shape,
        "1D_CPMG_T2_shape_brd": X_1d_brd_shape,
        "2D_v2_T1_only": X_2d_t1_only,
        "2D_v2_T1peak_plus_T2fwhm": X_2d_joint,
        "2D_v2_3feat_T1peak_T1fwhm_T2fwhm": X_2d_full3,
    }
    loo_pred = {"PUFA": {}, "SFA": {}}
    print("\n=== Step 4: LOO Out-of-Sample Composition Prediction (1D vs 2D) ===", flush=True)
    for target_name, y_vec in [("PUFA", pufa), ("SFA", sfa)]:
        print(f"  Target = {target_name}", flush=True)
        for fname, Xm in feature_sets.items():
            res = loo_ridge_predict(Xm, y_vec, alpha=1.0)
            loo_pred[target_name][fname] = res
            print(f"    {fname:<32}: RMSE={res['loo_rmse']:5.2f}%, R2={res['loo_r2']:+.4f}, "
                  f"rho={res['loo_spearman']:+.4f} (p={res['loo_p']:.4f})", flush=True)

    # Check E4 compatibility
    e4_path = os.path.join(PROTO, "e4_fingerprint.json")
    e4_orig = json.load(open(e4_path, encoding="utf-8"))
    max_peak_diff_e4 = 0.0
    for k, f in sample_feats.items():
        if k in e4_orig["features"]:
            max_peak_diff_e4 = max(
                max_peak_diff_e4,
                abs(round(f["svd_ridge_fixed"]["t1_peak"], 1) - e4_orig["features"][k]["t1_peak"]),
                abs(round(f["svd_ridge_fixed"]["t2_peak"], 1) - e4_orig["features"][k]["t2_peak"]),
            )
    print(f"\nE4 baseline parity check: max peak diff = {max_peak_diff_e4:.3f} ms", flush=True)

    out = {
        "protocol": {
            "n_oils": len(oils),
            "oils": oils,
            "fat_table": FAT_9,
            "n_exact_perms": int(len(all_perms_idx)),
            "n_bootstrap": 10000,
            "loo_ridge_alpha": 1.0,
            "wall_s": round(time.perf_counter() - t0, 2),
        },
        "svd_rank_audit": rank_info,
        "sample_features_13": sample_feats,
        "table3_hardening": chem_stats,
        "peak_collisions_9oils": {
            "1d_svd_ridge_unique_t2_bins": len(t2_1d_bins),
            "1d_svd_ridge_bins": {str(k): v for k, v in sorted(t2_1d_bins.items())},
            "2d_v2_unique_t1_t2_bins": len(t1t2_v2_bins),
            "2d_v2_bins": {f"{k[0]}/{k[1]}": v for k, v in sorted(t1t2_v2_bins.items())},
        },
        "step4_loo_prediction": loo_pred,
    }
    out_path = os.path.join(HERE, "results_archive", "stats_and_1d2d.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path} in {out['protocol']['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()

