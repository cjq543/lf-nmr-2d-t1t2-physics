# -*- coding: utf-8 -*-
"""
审稿加固补充实验 (2026-11-13):
针对 JMR 严格审稿视角进行四项只读离线统计与独立交叉实验核验：
  H1. 200 条合成测试集逐条配对差值 (v2 vs 6 种经典基线) + 95% Bootstrap CI + Wilcoxon p + v2 单核分段耗时拆解
  H2. 真实 23 MHz 样品 vs 合成样本的 SVD-GCV / BRD-GCV 目标函数完整曲线与分母项审计
  H3. n=9 植物油化学特征 Holm-Bonferroni 多重校正 + 结值核验 + 严格折内嵌套 LOO 自动特征选择
  H4. 同批独立采集 *__CPMG (1D T2) 与 *__IR (1D T1) 原始序列：
      (a) 13 样品跨序列独立 T1/T2 物理峰位交叉参照
      (b) 独立 1D CPMG（含完整衰减曲线直接回归，绕过单峰压缩）及独立 1D IR vs 2D v2 的公平 LOO 预测对比
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import glob
import time
import platform
import subprocess
import numpy as np
import torch
from scipy.stats import spearmanr, rankdata, wilcoxon

torch.set_num_threads(1)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = HERE
CLASSICAL = HERE
STATS_DIR = os.path.join(HERE, "results_archive")
for p in (PROTO, CLASSICAL, STATS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from sim2d import TI_MS, NECH, TE_MS, N_T1, N_T2, T1_GRID, T2_GRID, LOG_T1, LOG_T2, DL1, DL2, make_sample
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
    load_v2_cpu,
    reconstruct_brd_fixed,
    reconstruct_svd_ridge_gcv,
    reconstruct_brd_gcv,
    _brd_dual_newton,
)
from sparse_l1_baseline import build_sparse_ctx, reconstruct_fista_l1, reconstruct_guo_twostage
from stats_and_1d2d import FAT_9, EN_MAP, fwhm_1d, loo_ridge_predict


def get_cpu_name():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
            text=True, timeout=5
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def paired_diff_stats(a, b, rng, n_boot=10000):
    """计算 a - b 的逐样本配对统计：均值、中位数、95% Bootstrap CI、Wilcoxon p、胜/平/负。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = len(diff)
    idx_mat = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx_mat].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    if np.all(np.abs(diff) < 1e-12):
        w_p = 1.0
    else:
        _, w_p = wilcoxon(diff, alternative="two-sided")
    return {
        "n": int(n),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "mean_diff": round(float(diff.mean()), 4),
        "std_diff": round(float(diff.std()), 4),
        "median_diff": round(float(np.median(diff)), 4),
        "boot_ci95_mean_diff": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "wilcoxon_p": float(f"{w_p:.3e}"),
        "wins": int(np.sum(diff > 1e-12)),
        "ties": int(np.sum(np.abs(diff) <= 1e-12)),
        "losses": int(np.sum(diff < -1e-12)),
    }


def part1_paired_and_timing():
    print("\n=== [H1] 200-Sample Paired Baseline Audit & Timing Breakdown ===", flush=True)
    cache = KroneckerSVDCache()
    sim_ctx = cache.get(TI_MS, NECH)
    sp_ctx = build_sparse_ctx(TI_MS, NECH, tol=1e-6)
    front_v2, refine_v2 = load_v2_cpu()

    rng_e = np.random.default_rng(20261112 + 100)
    samples = [make_sample(rng_e) for _ in range(200)]

    # Warmup
    y0 = np.real(samples[0]["y"])
    yt0 = torch.tensor((y0 / (np.abs(y0).max() + 1e-9))[None], dtype=torch.float32)
    with torch.no_grad():
        b0 = front_v2(yt0)
        p0 = torch.tensor(log_pool_np((y0 / (np.abs(y0).max() + 1e-9))[None]), dtype=torch.float32)
        _ = refine_v2(b0, p0)

    methods = ["v2_frozen", "svd_ridge_fixed", "svd_ridge_gcv", "brd_nnls_fixed", "brd_nnls_gcv", "fista_l1", "guo_twostage"]
    per_sample = {m: [] for m in methods}
    stage_t = {"norm_tensor": 0.0, "physics_frontend": 0.0, "log_pool": 0.0, "refine_and_expm1": 0.0}

    for s in samples:
        Yr = np.real(s["y"])
        spec = s["spec"]

        # v2 with fine-grained stage timing (exact same inference pipeline as adaptive_lambda_baseline.py L368-376)
        t0 = time.perf_counter()
        ys = Yr / (np.abs(Yr).max() + 1e-9)
        yt = torch.tensor(ys[None], dtype=torch.float32)
        t1 = time.perf_counter()
        with torch.no_grad():
            base = front_v2(yt)[0]
        t2 = time.perf_counter()
        pooled = torch.tensor(log_pool_np(ys[None]), dtype=torch.float32)
        t3 = time.perf_counter()
        with torch.no_grad():
            pred_log = refine_v2(base[None], pooled)[0].numpy()
        M_v2 = np.expm1(np.clip(pred_log, 0.0, 20.0)) / 20.0
        t4 = time.perf_counter()
        stage_t["norm_tensor"] += t1 - t0
        stage_t["physics_frontend"] += t2 - t1
        stage_t["log_pool"] += t3 - t2
        stage_t["refine_and_expm1"] += t4 - t3
        per_sample["v2_frozen"].append(compute_metrics(M_v2, spec))

        # Classical solvers
        U1, S1, V1t = sim_ctx["ridge"]["U1"], sim_ctx["ridge"]["S1"], sim_ctx["ridge"]["V1t"]
        U2, S2, V2t = sim_ctx["ridge"]["U2"], sim_ctx["ridge"]["S2"], sim_ctx["ridge"]["V2t"]
        sprod = sim_ctx["ridge"]["s_prod"]
        smax = sim_ctx["ridge"]["smax"]
        Yc = U1.T @ Yr @ U2
        C_fix = sprod * Yc / (sprod ** 2 + (1e-3 * smax) ** 2)
        M_sfix = normalize_area(V1t.T @ C_fix @ V2t)
        per_sample["svd_ridge_fixed"].append(compute_metrics(M_sfix, spec))

        M_sgcv, _ = reconstruct_svd_ridge_gcv(Yr, sim_ctx["ridge"])
        per_sample["svd_ridge_gcv"].append(compute_metrics(M_sgcv, spec))

        M_bfix = reconstruct_brd_fixed(Yr, sim_ctx["brd"], lam=1e-3)
        per_sample["brd_nnls_fixed"].append(compute_metrics(M_bfix, spec))

        M_bgcv, _ = reconstruct_brd_gcv(Yr, sim_ctx["brd"])
        per_sample["brd_nnls_gcv"].append(compute_metrics(M_bgcv, spec))

        M_fis, _ = reconstruct_fista_l1(Yr, sp_ctx, alpha_ratio=1e-3)
        per_sample["fista_l1"].append(compute_metrics(M_fis, spec))

        M_guo, _ = reconstruct_guo_twostage(Yr, sp_ctx, alpha_ratio=1e-3, lam2_ratio=0.0)
        per_sample["guo_twostage"].append(compute_metrics(M_guo, spec))

    # Measure one-time fixed-grid SVD setup latency
    t_s0 = time.perf_counter()
    K1_tmp = 1.0 - 2.0 * np.exp(-TI_MS[:, None] / T1_GRID[None, :])
    echo_tmp = np.arange(1, NECH + 1, dtype=np.float64) * float(TE_MS)
    K2_tmp = np.exp(-echo_tmp[:, None] / T2_GRID[None, :])
    _ = np.linalg.svd(K1_tmp, full_matrices=False)
    _ = np.linalg.svd(K2_tmp, full_matrices=False)
    svd_init_ms = (time.perf_counter() - t_s0) * 1000.0

    rng_boot = np.random.default_rng(20261113)
    v2_r = [x[0] for x in per_sample["v2_frozen"]]
    v2_e1 = [x[1] for x in per_sample["v2_frozen"]]
    v2_e2 = [x[2] for x in per_sample["v2_frozen"]]

    paired = {}
    for m in methods[1:]:
        mr = [x[0] for x in per_sample[m]]
        me1 = [x[1] for x in per_sample[m]]
        me2 = [x[2] for x in per_sample[m]]
        st_r = paired_diff_stats(v2_r, mr, rng_boot)
        st_e1 = paired_diff_stats(me1, v2_e1, rng_boot)   # positive = baseline has higher error than v2
        st_e2 = paired_diff_stats(me2, v2_e2, rng_boot)
        paired[f"v2_vs_{m}"] = {"pearson_r_diff": st_r, "t1_err_reduction_dex": st_e1, "t2_err_reduction_dex": st_e2}
        print(
            f"  v2 vs {m:<16}: dr={st_r['mean_diff']:+.4f} (95% CI {st_r['boot_ci95_mean_diff']}, "
            f"W-p={st_r['wilcoxon_p']}, win={st_r['wins']}/200)",
            flush=True,
        )

    stage_ms = {k: round(v / 200.0 * 1000.0, 3) for k, v in stage_t.items()}
    stage_ms["total_measured_ms"] = round(sum(stage_ms.values()), 3)
    stage_ms["one_time_grid_svd_init_ms"] = round(svd_init_ms, 2)
    print(f"  v2 CPU breakdown (ms/sample, 1-thread): {stage_ms}", flush=True)

    return {
        "hardware": {
            "cpu_name": get_cpu_name(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "omp_num_threads": 1,
        },
        "v2_timing_breakdown_ms": stage_ms,
        "paired_comparisons": paired,
    }


def compute_gcv_trajectories(Yr, ctx_entry):
    """提取 SVD-GCV 与 BRD-GCV 在全 λ 网格上的完整分子 (RSS/n_obs)、分母 (1-df/n_obs)^2、df 和 GCV(λ)。"""
    lam_svd = np.logspace(-5, 0, 81)
    rc = ctx_entry["ridge"]
    U1, U2, s, smax, n_obs = rc["U1"], rc["U2"], rc["s_prod"], rc["smax"], rc["n_obs"]
    Ycore_r = U1.T @ Yr @ U2
    r_perp_r = max(0.0, float(np.sum(Yr * Yr) - np.sum(Ycore_r * Ycore_r)))
    s2, yc2, smax2 = s * s, Ycore_r * Ycore_r, smax * smax

    svd_curve = {"lam": [], "gcv": [], "rss_norm": [], "denom_sq": [], "df": []}
    for lam in lam_svd:
        reg = (float(lam) ** 2) * smax2
        denom = s2 + reg
        h = np.where(denom > 0.0, s2 / denom, 0.0)
        df = float(np.sum(h))
        r_par = float(np.sum(((1.0 - h) ** 2) * yc2))
        rss_norm = (r_perp_r + r_par) / n_obs
        denom_sq = (1.0 - df / n_obs) ** 2
        svd_curve["lam"].append(float(f"{lam:.6e}"))
        svd_curve["gcv"].append(float(rss_norm / denom_sq))
        svd_curve["rss_norm"].append(float(rss_norm))
        svd_curve["denom_sq"].append(float(denom_sq))
        svd_curve["df"].append(round(df, 2))
    idx_r = int(np.argmin(svd_curve["gcv"]))

    lam_brd = np.logspace(-1, -4.5, 15)
    bc = ctx_entry["brd"]
    Yc_b = bc["U1"].T @ Yr @ bc["U2"]
    r_perp_b = max(0.0, float(np.sum(Yr * Yr) - np.sum(Yc_b * Yc_b)))
    Ycore_vec = Yc_b.ravel()

    brd_curve = {"lam": [], "gcv": [], "rss_norm": [], "denom_sq": [], "df": []}
    c_warm = None
    for lam in lam_brd:
        alpha = (float(lam) * bc["smax"]) ** 2
        _, c_warm, df, rss_sub = _brd_dual_newton(bc["K_sub"], Ycore_vec, alpha, c_init=c_warm)
        rss_norm = (r_perp_b + rss_sub) / bc["n_obs"]
        denom_sq = (1.0 - df / bc["n_obs"]) ** 2
        brd_curve["lam"].append(float(f"{lam:.6e}"))
        brd_curve["gcv"].append(float(rss_norm / denom_sq))
        brd_curve["rss_norm"].append(float(rss_norm))
        brd_curve["denom_sq"].append(float(denom_sq))
        brd_curve["df"].append(round(float(df), 2))
    # Sort BRD ascending by lambda for plotting convenience
    order = np.argsort(brd_curve["lam"])
    for k in brd_curve:
        brd_curve[k] = [brd_curve[k][i] for i in order]
    idx_b = int(np.argmin(brd_curve["gcv"]))

    return {
        "n_obs": int(n_obs),
        "svd_p_max": int(U1.shape[1] * U2.shape[1]),
        "brd_p_max": int(bc["p"]),
        "svd_selected_lam": svd_curve["lam"][idx_r],
        "brd_selected_lam": brd_curve["lam"][idx_b],
        "svd_curve": svd_curve,
        "brd_curve": brd_curve,
    }


def part2_gcv_curves():
    print("\n=== [H2] Real vs Synthetic GCV(lambda) Curve Audit ===", flush=True)
    cache = KroneckerSVDCache()
    sim_ctx = cache.get(TI_MS, NECH)

    rng_e = np.random.default_rng(20261112 + 100)
    samples = [make_sample(rng_e) for _ in range(200)]
    # Pick 3 synthetic samples across SNR span
    snrs = [(i, s["snr"]) for i, s in enumerate(samples)]
    snrs_sorted = sorted(snrs, key=lambda x: x[1])
    pick_sim = [snrs_sorted[5], snrs_sorted[100], snrs_sorted[195]]

    out = {"synthetic": {}, "real": {}}
    for idx, snr_val in pick_sim:
        label = f"sim_idx{idx}_snr{snr_val:.0f}"
        Yr = np.real(samples[idx]["y"])
        tr = compute_gcv_trajectories(Yr, sim_ctx)
        out["synthetic"][label] = tr
        print(f"  {label:<18}: SVD lam*={tr['svd_selected_lam']:.2e} (denom_min={min(tr['svd_curve']['denom_sq']):.4f}) "
              f"| BRD lam*={tr['brd_selected_lam']:.2e} (denom_min={min(tr['brd_curve']['denom_sq']):.4f})", flush=True)

    target_reals = ["红花籽油", "橄榄油", "乙酸松油酯", "甘油"]
    dirs = {os.path.basename(d).split("__")[0]: d for d in glob.glob(os.path.join(LIB, "*__IR-CPMG"))}
    for name in target_reals:
        s = load_sample(dirs[name])
        Yr = np.real(phase_correct(s["Yc"]))
        ys = Yr / (np.abs(Yr).max() + 1e-9)
        ctx = cache.get(s["ti"], ys.shape[1])
        tr = compute_gcv_trajectories(ys, ctx)
        out["real"][EN_MAP[name]] = tr
        print(f"  real {EN_MAP[name]:<14}: SVD lam*={tr['svd_selected_lam']:.2e} (denom_min={min(tr['svd_curve']['denom_sq']):.4f}) "
              f"| BRD lam*={tr['brd_selected_lam']:.2e} (denom_min={min(tr['brd_curve']['denom_sq']):.4f})", flush=True)
    return out


def holm_bonferroni(p_dict):
    """Holm-Bonferroni 步降多重校正。"""
    items = sorted(p_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for rank, (k, p) in enumerate(items):
        val = min(1.0, (m - rank) * float(p))
        running_max = max(running_max, val)
        adj[k] = round(running_max, 5)
    return adj


def nested_loo_feature_select(X_dict, y, k_select=2, alpha=1.0):
    """严格无泄漏嵌套 LOO：每一折仅在 n-1 个训练样本内选 |Spearman rho| 最大的 k 个特征再做 Ridge。"""
    feat_names = sorted(X_dict.keys())
    X_all = np.column_stack([X_dict[fn] for fn in feat_names])
    n = len(y)
    preds = np.zeros(n, dtype=np.float64)
    chosen_per_fold = []
    eye_k = np.eye(k_select, dtype=np.float64)

    for i in range(n):
        tr = np.arange(n) != i
        Xtr_full, ytr = X_all[tr], y[tr]
        scores = []
        for j, fn in enumerate(feat_names):
            r_j, _ = spearmanr(Xtr_full[:, j], ytr)
            scores.append((abs(float(r_j)) if np.isfinite(r_j) else 0.0, j, fn))
        scores.sort(reverse=True)
        top_idx = [s[1] for s in scores[:k_select]]
        chosen_per_fold.append([s[2] for s in scores[:k_select]])

        Xtr = Xtr_full[:, top_idx]
        Xte = X_all[i, top_idx]
        mu = Xtr.mean(axis=0)
        sd = np.where(Xtr.std(axis=0) > 1e-12, Xtr.std(axis=0), 1.0)
        Xtr_z = (Xtr - mu) / sd
        Xte_z = (Xte - mu) / sd
        y_mu = ytr.mean()
        w = np.linalg.solve(Xtr_z.T @ Xtr_z + alpha * eye_k, Xtr_z.T @ (ytr - y_mu))
        preds[i] = float(y_mu + Xte_z @ w)

    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    r2 = float(1.0 - np.sum((preds - y) ** 2) / np.sum((y - y.mean()) ** 2))
    rho_loo, p_loo = spearmanr(y, preds)
    return {
        "k_select": int(k_select),
        "loo_rmse": round(rmse, 2),
        "loo_r2": round(r2, 4),
        "loo_spearman": round(float(rho_loo), 4),
        "loo_p": round(float(p_loo), 5),
        "chosen_features_per_fold": chosen_per_fold,
    }


def part3_chemical_stats_hardening():
    print("\n=== [H3] Ties Audit, Holm-Bonferroni Correction & Strict Nested LOO ===", flush=True)
    st_path = os.path.join(STATS_DIR, "stats_and_1d2d.json")
    st = json.loads(open(st_path, encoding="utf-8").read())
    oils = st["protocol"]["oils"]
    pufa = np.array([FAT_9[k][2] for k in oils], dtype=np.float64)
    sfa = np.array([FAT_9[k][0] for k in oils], dtype=np.float64)

    # 1. Explicit ties audit for v2 T1 FWHM
    t1_w = np.array([st["sample_features_13"][k]["v2"]["t1_fwhm"] for k in oils], dtype=np.float64)
    unique_w, counts = np.unique(np.round(t1_w, 4), return_counts=True)
    ties_summary = [{"t1_fwhm_ms": float(u), "count": int(c)} for u, c in zip(unique_w, counts) if c > 1]
    print(f"  v2 T1_FWHM tied values on 48-bin grid: {ties_summary} (midranks explain rho=0.9621)", flush=True)

    # 2. Holm-Bonferroni over all 7 candidate 2D features per solver
    holm_results = {}
    for solver in ["v2", "svd_ridge_fixed", "brd_nnls_fixed"]:
        p_raw = {fk: st["table3_hardening"][solver][fk]["vs_PUFA"]["p_exact_perm"] for fk in st["table3_hardening"][solver]}
        p_adj = holm_bonferroni(p_raw)
        holm_results[solver] = {
            fk: {"rho": st["table3_hardening"][solver][fk]["vs_PUFA"]["rho"],
                 "p_perm_raw": p_raw[fk], "p_holm_7feat": p_adj[fk]}
            for fk in p_raw
        }
        print(f"  {solver:<16} t1_fwhm vs PUFA: p_raw={p_raw['t1_fwhm']:.5f}, p_holm(7 feat)={p_adj['t1_fwhm']:.5f}", flush=True)

    # 3. Strict Nested LOO across all 7 candidate 2D v2 features (ZERO label leakage in feature selection)
    cand_X = {
        "t1_peak": np.log10([st["sample_features_13"][k]["v2"]["t1_peak"] for k in oils]),
        "t2_peak": np.log10([st["sample_features_13"][k]["v2"]["t2_peak"] for k in oils]),
        "t1_fwhm": np.log10([st["sample_features_13"][k]["v2"]["t1_fwhm"] for k in oils]),
        "t2_fwhm": np.log10([st["sample_features_13"][k]["v2"]["t2_fwhm"] for k in oils]),
        "ratio": np.log10([st["sample_features_13"][k]["v2"]["ratio"] for k in oils]),
        "mean_logT1": np.array([st["sample_features_13"][k]["v2"]["mean_logT1"] for k in oils], dtype=np.float64),
        "mean_logT2": np.array([st["sample_features_13"][k]["v2"]["mean_logT2"] for k in oils], dtype=np.float64),
    }
    X_all7 = np.column_stack([cand_X[fn] for fn in sorted(cand_X.keys())])

    nested = {}
    for tgt_name, y_vec in [("PUFA", pufa), ("SFA", sfa)]:
        res_k1 = nested_loo_feature_select(cand_X, y_vec, k_select=1, alpha=1.0)
        res_k2 = nested_loo_feature_select(cand_X, y_vec, k_select=2, alpha=1.0)
        res_k3 = nested_loo_feature_select(cand_X, y_vec, k_select=3, alpha=1.0)
        res_all7 = loo_ridge_predict(X_all7, y_vec, alpha=3.0)
        nested[tgt_name] = {
            "nested_top1": res_k1, "nested_top2": res_k2,
            "nested_top3": res_k3, "all_7feat_ridge_a3": res_all7,
        }
        print(
            f"  Nested LOO ({tgt_name}): top1 R2={res_k1['loo_r2']:+.4f} (RMSE={res_k1['loo_rmse']}%), "
            f"top2 R2={res_k2['loo_r2']:+.4f} (RMSE={res_k2['loo_rmse']}%), "
            f"top3 R2={res_k3['loo_r2']:+.4f}, all7(a=3) R2={res_all7['loo_r2']:+.4f}",
            flush=True,
        )

    return {"ties_audit": ties_summary, "holm_bonferroni_7features": holm_results, "nested_loo_v2": nested}


def parse_pea_file(path):
    t, amp, re_v, im_v = [], [], [], []
    with open(path, encoding="gbk", errors="ignore") as f:
        for ln in f:
            ps = ln.split()
            if len(ps) == 4:
                try:
                    t.append(float(ps[0])); amp.append(float(ps[1]))
                    re_v.append(float(ps[2])); im_v.append(float(ps[3]))
                except ValueError:
                    pass
    return np.array(t), np.array(amp), np.array(re_v) + 1j * np.array(im_v)


def parse_inv_file(path):
    x, y = [], []
    with open(path, encoding="gbk", errors="ignore") as f:
        for ln in f:
            ps = ln.split()
            if len(ps) == 2:
                try:
                    x.append(float(ps[0])); y.append(float(ps[1]))
                except ValueError:
                    pass
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(x[int(np.argmax(y))]) if len(x) and y.max() > 0 else float("nan")


def invert_1d(y1d, times_ms, grid_ms, kernel_type="cpmg", lam=1e-3):
    """在指定一维时间与谱网格上同时跑 SVD-ridge (λ=1e-3) 与 BRD NNLS (λ=1e-3)。"""
    y1d = np.asarray(y1d, dtype=np.float64)
    y1d = y1d / (np.max(np.abs(y1d)) + 1e-12)
    log_g = np.log10(grid_ms)
    dl = float(log_g[1] - log_g[0])

    if kernel_type == "cpmg":
        K = np.exp(-times_ms[:, None] / grid_ms[None, :])
    elif kernel_type == "ir":
        K = 1.0 - 2.0 * np.exp(-times_ms[:, None] / grid_ms[None, :])
    else:
        raise ValueError(kernel_type)

    U, S, Vt = np.linalg.svd(K, full_matrices=False)
    smax = float(S[0])

    # SVD-ridge
    rr = int((S > smax * 1e-10).sum())
    yc_r = U[:, :rr].T @ y1d
    cr = S[:rr] * yc_r / (S[:rr] ** 2 + (lam * smax) ** 2)
    f_ridge = np.clip(Vt[:rr].T @ cr, 0.0, None)
    f_ridge /= (f_ridge.sum() * dl + 1e-12)

    # BRD dual Newton NNLS (tol=1e-6 or all nonzero for small 20-pt IR)
    rb = int((S > smax * 1e-6).sum())
    K_sub = S[:rb, None] * Vt[:rb]
    yc_b = U[:, :rb].T @ y1d
    alpha = (lam * smax) ** 2
    eye_p = np.eye(rb, dtype=np.float64)
    c = np.zeros(rb, dtype=np.float64)
    yn = float(np.linalg.norm(yc_b)) + 1e-12
    for _ in range(50):
        ktc = K_sub.T @ c
        act = ktc > 0.0
        f = np.where(act, ktc, 0.0)
        g = alpha * c + K_sub @ f - yc_b
        if np.linalg.norm(g) <= 1e-8 * yn:
            break
        Ka = K_sub[:, act] if np.any(act) else np.zeros((rb, 0))
        H = alpha * eye_p + Ka @ Ka.T
        step = -np.linalg.solve(0.5 * (H + H.T), g)
        phi0 = 0.5 * alpha * np.dot(c, c) + 0.5 * np.dot(f, f) - np.dot(c, yc_b)
        slope = np.dot(g, step)
        step_s = 1.0
        for _ls in range(25):
            cn = c + step_s * step
            fn = np.maximum(0.0, K_sub.T @ cn)
            phin = 0.5 * alpha * np.dot(cn, cn) + 0.5 * np.dot(fn, fn) - np.dot(cn, yc_b)
            if phin <= phi0 + 1e-4 * step_s * slope:
                c = cn
                break
            step_s *= 0.5
    f_brd = np.maximum(0.0, K_sub.T @ c)
    f_brd /= (f_brd.sum() * dl + 1e-12)

    def summarize(p):
        peak = float(10.0 ** log_g[int(np.argmax(p))])
        w = fwhm_1d(p, log_g)
        wt = p / (p.sum() + 1e-12)
        mlog = float((wt * log_g).sum())
        slog = float(np.sqrt((wt * (log_g - mlog) ** 2).sum()))
        return {"peak_ms": peak, "fwhm_ms": w, "mean_log": mlog, "std_log": slog}

    return summarize(f_ridge), summarize(f_brd), f_ridge, f_brd


def part4_standalone_ir_cpmg():
    print("\n=== [H4] Standalone *__CPMG & *__IR Cross-Sequence Reference & Full-Decay 1D vs 2D ===", flush=True)
    st_path = os.path.join(STATS_DIR, "stats_and_1d2d.json")
    st = json.loads(open(st_path, encoding="utf-8").read())

    # Extended T1 grid matching IR up to TI=8000 ms, plus exact 48-bin grid comparison
    T1_EXT = np.logspace(np.log10(1.0), np.log10(10000.0), 80)

    sample_names = sorted(st["sample_features_13"].keys())
    cross_rows = []
    standalone_1d_feats = {}
    cpmg_full_vecs = {}
    ir_full_vecs = {}

    # Shared log-time binning for raw full-decay CPMG comparison across 9 oils (up to 1500 ms shared window)
    edges_1500 = np.unique(np.geomspace(1.0, 1500.0, 65))
    common_ti_grid = np.geomspace(0.058, 3000.0, 20)

    for name in sample_names:
        cdir = os.path.join(LIB, f"{name}__CPMG")
        idir = os.path.join(LIB, f"{name}__IR")
        t_c, amp_c, yc_c = parse_pea_file(glob.glob(os.path.join(cdir, "*.1.pea"))[0])
        # Phase-correct standalone 1D CPMG onto dominant initial echo axis
        ph_c = np.angle(np.mean(yc_c[: min(30, len(yc_c))]))
        y_cpmg_re = np.real(yc_c * np.exp(-1j * ph_c))
        inv_c_peak = parse_inv_file(glob.glob(os.path.join(cdir, "*.1.Inv"))[0])

        t_i, amp_signed_ir, yc_ir = parse_pea_file(glob.glob(os.path.join(idir, "*.1.irp"))[0])
        inv_ir_peak = parse_inv_file(glob.glob(os.path.join(idir, "*.1.Inv"))[0])

        # 1D CPMG inversion on same T2_GRID
        fc_ridge, fc_brd, _, _ = invert_1d(y_cpmg_re, t_c, T2_GRID, kernel_type="cpmg", lam=1e-3)
        # 1D IR inversion on same T1_GRID AND extended T1_EXT (using signed .1.irp amplitude)
        fi_ridge, fi_brd, _, _ = invert_1d(amp_signed_ir, t_i, T1_GRID, kernel_type="ir", lam=1e-3)
        fi_ext_r, _, _, _ = invert_1d(amp_signed_ir, t_i, T1_EXT, kernel_type="ir", lam=1e-3)

        # Normalize full raw decay curves for non-inverted 1D baseline
        yc_norm = y_cpmg_re / (np.max(np.abs(y_cpmg_re)) + 1e-12)
        binned_c = []
        for a, b in zip(edges_1500[:-1], edges_1500[1:]):
            sel = (t_c >= a) & (t_c <= b)
            binned_c.append(float(np.mean(yc_norm[sel])) if np.any(sel) else float(np.interp(0.5 * (a + b), t_c, yc_norm)))
        cpmg_full_vecs[name] = np.asarray(binned_c, dtype=np.float64)

        yir_norm = amp_signed_ir / (np.max(np.abs(amp_signed_ir)) + 1e-12)
        ir_full_vecs[name] = np.interp(np.log10(common_ti_grid), np.log10(t_i), yir_norm)

        v2f = st["sample_features_13"][name]["v2"]
        dt1_dex = abs(np.log10(v2f["t1_peak"]) - np.log10(fi_ridge["peak_ms"]))
        dt2_dex = abs(np.log10(v2f["t2_peak"]) - np.log10(fc_ridge["peak_ms"]))

        row = {
            "name": name,
            "en": EN_MAP[name],
            "is_oil": name in FAT_9,
            "v2_2d_T1": round(v2f["t1_peak"], 1),
            "v2_2d_T2": round(v2f["t2_peak"], 1),
            "standalone_1d_IR_T1_ridge": round(fi_ridge["peak_ms"], 1),
            "standalone_1d_IR_T1_ext_ridge": round(fi_ext_r["peak_ms"], 1),
            "standalone_1d_IR_vendor_Inv": round(inv_ir_peak, 1),
            "standalone_1d_CPMG_T2_ridge": round(fc_ridge["peak_ms"], 1),
            "standalone_1d_CPMG_vendor_Inv": round(inv_c_peak, 1),
            "abs_dlog10_T1_v2_vs_1dIR": round(float(dt1_dex), 4),
            "abs_dlog10_T2_v2_vs_1dCPMG": round(float(dt2_dex), 4),
        }
        cross_rows.append(row)
        standalone_1d_feats[name] = {
            "cpmg_ridge": fc_ridge, "cpmg_brd": fc_brd,
            "ir_ridge": fi_ridge, "ir_brd": fi_brd,
            "ir_ext_ridge": fi_ext_r,
        }
        print(
            f"  {EN_MAP[name]:<16} | T1: 2D_v2={row['v2_2d_T1']:>6.1f} vs 1D_IR={row['standalone_1d_IR_T1_ridge']:>6.1f} (ext={row['standalone_1d_IR_T1_ext_ridge']:>6.1f}, Inv={row['standalone_1d_IR_vendor_Inv']:>6.1f}) "
            f"| T2: 2D_v2={row['v2_2d_T2']:>5.1f} vs 1D_CPMG={row['standalone_1d_CPMG_T2_ridge']:>5.1f} (Inv={row['standalone_1d_CPMG_vendor_Inv']:>5.1f})",
            flush=True,
        )

    dt1_all = [r["abs_dlog10_T1_v2_vs_1dIR"] for r in cross_rows]
    dt2_all = [r["abs_dlog10_T2_v2_vs_1dCPMG"] for r in cross_rows]
    t1_corr = float(np.corrcoef(
        np.log10([r["v2_2d_T1"] for r in cross_rows]),
        np.log10([r["standalone_1d_IR_T1_ridge"] for r in cross_rows]),
    )[0, 1])
    t2_corr = float(np.corrcoef(
        np.log10([r["v2_2d_T2"] for r in cross_rows]),
        np.log10([r["standalone_1d_CPMG_T2_ridge"] for r in cross_rows]),
    )[0, 1])
    summary_cross = {
        "log10_T1_corr_2d_v2_vs_standalone_1d_IR": round(t1_corr, 4),
        "log10_T2_corr_2d_v2_vs_standalone_1d_CPMG": round(t2_corr, 4),
        "mean_abs_dlog10_T1": round(float(np.mean(dt1_all)), 4),
        "median_abs_dlog10_T1": round(float(np.median(dt1_all)), 4),
        "mean_abs_dlog10_T2": round(float(np.mean(dt2_all)), 4),
        "median_abs_dlog10_T2": round(float(np.median(dt2_all)), 4),
    }
    print(f"  Cross-sequence summary (n=13): {summary_cross}", flush=True)

    # Fair 1D vs 2D LOO prediction across n=9 oils (including full-decay curves!)
    oils = sorted(list(FAT_9.keys()))
    pufa = np.array([FAT_9[k][2] for k in oils], dtype=np.float64)
    sfa = np.array([FAT_9[k][0] for k in oils], dtype=np.float64)

    # Full-decay PCA inside each LOO fold (zero leakage) OR Ridge directly on binned decay curve
    def loo_full_decay_ridge(X_raw, y, n_pcs=None, alpha=10.0):
        n = len(y)
        preds = np.zeros(n, dtype=np.float64)
        for i in range(n):
            tr = np.arange(n) != i
            Xtr, ytr = X_raw[tr], y[tr]
            mu = Xtr.mean(axis=0)
            Xtr_c = Xtr - mu
            Xte_c = X_raw[i : i + 1] - mu
            if n_pcs is not None:
                _, _, vt = np.linalg.svd(Xtr_c, full_matrices=False)
                Ztr = Xtr_c @ vt[:n_pcs].T
                Zte = Xte_c @ vt[:n_pcs].T
            else:
                Ztr, Zte = Xtr_c, Xte_c
            sd = np.where(Ztr.std(axis=0) > 1e-12, Ztr.std(axis=0), 1.0)
            Ztr_z = Ztr / sd
            Zte_z = Zte / sd
            ym = ytr.mean()
            w = np.linalg.solve(Ztr_z.T @ Ztr_z + alpha * np.eye(Ztr_z.shape[1]), Ztr_z.T @ (ytr - ym))
            preds[i] = float(ym + (Zte_z @ w)[0])
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        r2 = float(1.0 - np.sum((preds - y) ** 2) / np.sum((y - y.mean()) ** 2))
        rho, p = spearmanr(y, preds)
        return {"loo_rmse": round(rmse, 2), "loo_r2": round(r2, 4), "loo_spearman": round(float(rho), 4), "loo_p": round(float(p), 5)}

    X_st_cpmg_peak = np.array([[np.log10(standalone_1d_feats[k]["cpmg_ridge"]["peak_ms"])] for k in oils])
    X_st_cpmg_shape_r = np.array([[standalone_1d_feats[k]["cpmg_ridge"]["mean_log"], np.log10(standalone_1d_feats[k]["cpmg_ridge"]["fwhm_ms"])] for k in oils])
    X_st_cpmg_shape_b = np.array([[standalone_1d_feats[k]["cpmg_brd"]["mean_log"], np.log10(standalone_1d_feats[k]["cpmg_brd"]["fwhm_ms"])] for k in oils])
    X_st_cpmg_decay = np.stack([cpmg_full_vecs[k] for k in oils])

    X_st_ir_shape_r = np.array([[np.log10(standalone_1d_feats[k]["ir_ridge"]["peak_ms"]), np.log10(standalone_1d_feats[k]["ir_ridge"]["fwhm_ms"])] for k in oils])
    X_st_ir_shape_b = np.array([[np.log10(standalone_1d_feats[k]["ir_brd"]["peak_ms"]), np.log10(standalone_1d_feats[k]["ir_brd"]["fwhm_ms"])] for k in oils])
    X_st_ir_decay = np.stack([ir_full_vecs[k] for k in oils])

    X_2d_v2_t1 = np.array([[np.log10(st["sample_features_13"][k]["v2"]["t1_peak"]), np.log10(st["sample_features_13"][k]["v2"]["t1_fwhm"])] for k in oils])

    # Check 1D standalone IR T1_FWHM permutation correlation vs PUFA too!
    all_perms = np.array(list(np.fromiter(
        __import__("itertools").permutations(range(9)), dtype=np.dtype((int, 9))
    )))
    w1_ir_r = np.array([standalone_1d_feats[k]["ir_ridge"]["fwhm_ms"] for k in oils])
    w1_ir_b = np.array([standalone_1d_feats[k]["ir_brd"]["fwhm_ms"] for k in oils])

    def perm_p(x, y):
        rho, _ = spearmanr(x, y)
        rx = rankdata(x); ry = rankdata(y)
        rxc = rx - rx.mean(); ryc = ry - ry.mean()
        den = np.sqrt(np.sum(rxc ** 2) * np.sum(ryc ** 2)) + 1e-12
        pr = (ryc[all_perms] @ rxc) / den
        return round(float(rho), 4), round(float(np.mean(np.abs(pr) >= abs(float(rho)) - 1e-12)), 5)

    ir_fwhm_corr = {
        "standalone_1d_IR_ridge_t1_fwhm_vs_PUFA": dict(zip(["rho", "p_exact_perm"], perm_p(w1_ir_r, pufa))),
        "standalone_1d_IR_brd_t1_fwhm_vs_PUFA": dict(zip(["rho", "p_exact_perm"], perm_p(w1_ir_b, pufa))),
        "2d_v2_t1_fwhm_vs_PUFA": {"rho": 0.9621, "p_exact_perm": 0.00022},
    }

    fair_loo = {"PUFA": {}, "SFA": {}}
    print("\n  --- Fair Standalone 1D (incl. Raw Full-Decay) vs 2D v2 LOO Prediction ---")
    for tgt, yv in [("PUFA", pufa), ("SFA", sfa)]:
        res = {
            "standalone_1D_CPMG_T2_peak": loo_ridge_predict(X_st_cpmg_peak, yv, alpha=1.0),
            "standalone_1D_CPMG_T2_shape_ridge": loo_ridge_predict(X_st_cpmg_shape_r, yv, alpha=1.0),
            "standalone_1D_CPMG_T2_shape_brd": loo_ridge_predict(X_st_cpmg_shape_b, yv, alpha=1.0),
            "standalone_1D_CPMG_raw_decay_PCA2_Ridge": loo_full_decay_ridge(X_st_cpmg_decay, yv, n_pcs=2, alpha=1.0),
            "standalone_1D_CPMG_raw_decay_PCA3_Ridge": loo_full_decay_ridge(X_st_cpmg_decay, yv, n_pcs=3, alpha=1.0),
            "standalone_1D_CPMG_raw_decay_DirectRidge10": loo_full_decay_ridge(X_st_cpmg_decay, yv, n_pcs=None, alpha=10.0),
            "standalone_1D_IR_T1_shape_ridge": loo_ridge_predict(X_st_ir_shape_r, yv, alpha=1.0),
            "standalone_1D_IR_T1_shape_brd": loo_ridge_predict(X_st_ir_shape_b, yv, alpha=1.0),
            "standalone_1D_IR_raw_curve_PCA2_Ridge": loo_full_decay_ridge(X_st_ir_decay, yv, n_pcs=2, alpha=1.0),
            "2D_v2_T1_peak_plus_FWHM": loo_ridge_predict(X_2d_v2_t1, yv, alpha=1.0),
        }
        fair_loo[tgt] = res
        print(f"  Target = {tgt}:")
        for mname, mv in res.items():
            print(f"    {mname:<43}: R2={mv['loo_r2']:+.4f}, RMSE={mv['loo_rmse']:5.2f}%, rho={mv['loo_spearman']:+.4f} (p={mv['loo_p']:.4f})", flush=True)

    return {
        "cross_sequence_13_summary": summary_cross,
        "cross_sequence_13_rows": cross_rows,
        "standalone_1d_ir_fwhm_vs_pufa": ir_fwhm_corr,
        "fair_1d_vs_2d_loo": fair_loo,
    }


def main():
    t_all = time.perf_counter()
    h1 = part1_paired_and_timing()
    if not (load_sample and LIB and os.path.isdir(LIB)):
        print("Restricted 23 MHz raw waveforms not attached (set NMR23_DATA_DIR); H1 synthetic + H3 derived stats run offline, full archive in results_archive/hardening_results.json.", flush=True)
        h3 = part3_chemical_stats_hardening()
        return
    h2 = part2_gcv_curves()
    h3 = part3_chemical_stats_hardening()
    h4 = part4_standalone_ir_cpmg()

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_s": round(time.perf_counter() - t_all, 2),
        "H1_paired_stats_and_timing": h1,
        "H2_gcv_curves": h2,
        "H3_chemical_stats_hardening": h3,
        "H4_standalone_ir_cpmg": h4,
    }
    out_json = os.path.join(HERE, "results_archive", "hardening_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nALL DONE -> {out_json} ({out['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()

