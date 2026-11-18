# -*- coding: utf-8 -*-
"""
Task A: 经典强基线 —— 逐样本自适应正则 (GCV) 与非负约束对偶牛顿 (Song 2002 BRD)

实现与对比四类经典 2D ILT（全部在同一物理模型与 Kronecker 网格上运行，只读引用冻结资产）：
1. svd_ridge_fixed (λ=1e-3)     : 复现冻结 baseline2d 基准
2. svd_ridge_gcv                : 对角化 SVD-ridge 逐样本全数据 GCV 自适应 λ
3. brd_nnls_fixed  (λ=1e-3)     : Song 2002 / Venkataramanan 2002 对偶牛顿非负约束 Tikhonov
4. brd_nnls_gcv                 : 对偶牛顿非负约束 + 活性集有效自由度 GCV 自适应 λ

口径：所有指标（pearson、T1/T2 峰位误差 dex、真实样本一致性）均在线性域单位 log-面积归一化谱上计算。
"""
import os
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROTO = HERE
sys.path.insert(0, PROTO)

from sim2d import (
    TI_MS, NECH, TE_MS, N_T1, N_T2, T1_GRID, T2_GRID,
    LOG_T1, LOG_T2, DL1, DL2, K_T1, K_T2, make_sample
)
from baseline2d import reconstruct_2d as frozen_baseline_reconstruct
try:
    from realdata2d import LIB, load_sample, phase_correct
except ImportError:
    LIB = os.environ.get("NMR23_DATA_DIR", "")
    load_sample = None
    phase_correct = None
from train2d_v2 import PhysicsFrontEnd, RefineNet, log_pool_np

LOG_T1_F = np.asarray(LOG_T1, dtype=np.float64)
LOG_T2_F = np.asarray(LOG_T2, dtype=np.float64)
GRID_AREA = float(DL1 * DL2)


def normalize_area(M):
    M = np.clip(np.asarray(M, dtype=np.float64), 0.0, None)
    area = float(M.sum()) * GRID_AREA
    if area > 1e-12:
        M = M / area
    return M


def compute_metrics(M, spec_true):
    p = normalize_area(M)
    r = normalize_area(spec_true)
    pear = float(np.corrcoef(p.ravel(), r.ravel())[0, 1])
    e1 = abs(float(LOG_T1_F[int(np.argmax(p.sum(1)))] - LOG_T1_F[int(np.argmax(r.sum(1)))]))
    e2 = abs(float(LOG_T2_F[int(np.argmax(p.sum(0)))] - LOG_T2_F[int(np.argmax(r.sum(0)))]))
    return pear, e1, e2


def marginal_peaks_ms(M):
    t1p = 10.0 ** float(LOG_T1_F[int(np.argmax(M.sum(1)))])
    t2p = 10.0 ** float(LOG_T2_F[int(np.argmax(M.sum(0)))])
    return t1p, t2p


class KroneckerSVDCache:
    """按 (tuple(ti_ms), nech) 缓存 SVD 与 BRD 压缩核，兼容仿真与多形状真实数据。"""

    def __init__(self, ridge_tol=1e-10, brd_tol=1e-6):
        self.ridge_tol = ridge_tol
        self.brd_tol = brd_tol
        self._cache = {}

    def get(self, ti_ms, nech):
        ti_arr = np.asarray(ti_ms, dtype=np.float64)
        key = (tuple(np.round(ti_arr, 6)), int(nech))
        if key in self._cache:
            return self._cache[key]

        echo_ms = np.arange(1, int(nech) + 1, dtype=np.float64) * float(TE_MS)
        K1 = 1.0 - 2.0 * np.exp(-ti_arr[:, None] / T1_GRID[None, :])
        K2 = np.exp(-echo_ms[:, None] / T2_GRID[None, :])

        U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
        U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
        smax = float(S1[0] * S2[0])

        r1_r = int((S1 > S1[0] * self.ridge_tol).sum())
        r2_r = int((S2 > S2[0] * self.ridge_tol).sum())
        ridge_ctx = {
            "U1": U1[:, :r1_r], "S1": S1[:r1_r], "V1t": V1t[:r1_r],
            "U2": U2[:, :r2_r], "S2": S2[:r2_r], "V2t": V2t[:r2_r],
            "s_prod": S1[:r1_r, None] * S2[:r2_r][None, :],
            "smax": smax,
            "n_obs": int(len(ti_arr) * int(nech)),
        }

        r1_b = int((S1 > S1[0] * self.brd_tol).sum())
        r2_b = int((S2 > S2[0] * self.brd_tol).sum())
        # K_sub: (r1_b * r2_b, N_T1 * N_T2), 与 np.outer 向量化顺序一致: vec(A) = A.reshape(-1)
        # Y_core = U1^T Y U2 = S1 C S2^T, C = V1t M V2t^T => vec(K1_sub M K2_sub^T) = K_sub vec(M)
        V1_b = V1t[:r1_b]                 # (r1_b, N_T1)
        V2_b = V2t[:r2_b]                 # (r2_b, N_T2)
        K_sub = (
            (S1[:r1_b, None] * V1_b)[:, None, :, None]
            * (S2[:r2_b, None] * V2_b)[None, :, None, :]
        ).reshape(r1_b * r2_b, N_T1 * N_T2)
        K_sub = np.ascontiguousarray(K_sub, dtype=np.float64)

        brd_ctx = {
            "U1": U1[:, :r1_b], "U2": U2[:, :r2_b],
            "K_sub": K_sub, "p": r1_b * r2_b,
            "smax": smax, "n_obs": int(len(ti_arr) * int(nech)),
            "r1": r1_b, "r2": r2_b,
        }
        entry = {"ridge": ridge_ctx, "brd": brd_ctx}
        self._cache[key] = entry
        return entry


def reconstruct_svd_ridge_gcv(Y_real, ctx, lam_grid=None):
    """在全数据观测网格上按对角化 SVD-ridge 最小化精确 GCV(λ) 选取逐样本 λ。"""
    if lam_grid is None:
        lam_grid = np.logspace(-5, 0, 81)
    Y = np.asarray(Y_real, dtype=np.float64)
    U1, U2 = ctx["U1"], ctx["U2"]
    V1t, V2t = ctx["V1t"], ctx["V2t"]
    s = ctx["s_prod"]
    smax = ctx["smax"]
    n_obs = ctx["n_obs"]

    Ycore = U1.T @ Y @ U2
    r_perp = max(0.0, float(np.sum(Y * Y) - np.sum(Ycore * Ycore)))

    s2 = s * s
    yc2 = Ycore * Ycore
    smax2 = smax * smax

    best_gcv = np.inf
    best_lam = float(lam_grid[len(lam_grid) // 2])
    for lam in lam_grid:
        reg = (float(lam) ** 2) * smax2
        denom = s2 + reg
        h = np.where(denom > 0.0, s2 / denom, 0.0)
        df = float(np.sum(h))
        if df >= 0.9 * n_obs:
            continue
        r_par = float(np.sum(((1.0 - h) ** 2) * yc2))
        gcv = ((r_perp + r_par) / n_obs) / ((1.0 - df / n_obs) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)

    reg_opt = (best_lam ** 2) * smax2
    denom_opt = s2 + reg_opt
    C = np.where(denom_opt > 0.0, s * Ycore / np.where(denom_opt > 0.0, denom_opt, 1.0), 0.0)
    M = normalize_area(V1t.T @ C @ V2t)
    return M, best_lam


def _brd_dual_newton(K_sub, y_core, alpha, c_init=None, max_iter=45):
    """
    Song 2002 / Venkataramanan 2002 对偶牛顿求解:
        min_{f >= 0} 0.5 ||K_sub f - y_core||^2 + 0.5 alpha ||f||^2
    对偶目标 Φ(c) = 0.5 alpha ||c||^2 + 0.5 ||max(0, K_sub^T c)||^2 - c^T y_core
    """
    p = K_sub.shape[0]
    y_norm = float(np.linalg.norm(y_core)) + 1e-12
    c = np.zeros(p, dtype=np.float64) if c_init is None else c_init.copy()

    def eval_phi_f_g(c_vec):
        ktc = K_sub.T @ c_vec
        act = ktc > 0.0
        f = np.where(act, ktc, 0.0)
        phi = 0.5 * alpha * float(np.dot(c_vec, c_vec)) + 0.5 * float(np.dot(f, f)) - float(np.dot(c_vec, y_core))
        g = alpha * c_vec + (K_sub @ f) - y_core
        return phi, f, act, g

    phi, f, act, g = eval_phi_f_g(c)
    eye_p = np.eye(p, dtype=np.float64)

    for _ in range(max_iter):
        if float(np.linalg.norm(g)) <= 1e-8 * y_norm:
            break
        if np.any(act):
            Ka = K_sub[:, act]
            H = alpha * eye_p + (Ka @ Ka.T)
            H = 0.5 * (H + H.T)
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = -np.linalg.solve(H + 1e-10 * alpha * eye_p, g)
        else:
            step = -g / alpha

        slope = float(np.dot(g, step))
        if slope >= 0.0:
            step = -g / max(alpha, 1e-12)
            slope = float(np.dot(g, step))

        t = 1.0
        accepted = False
        for _ls in range(25):
            c_try = c + t * step
            phi_try, f_try, act_try, g_try = eval_phi_f_g(c_try)
            if phi_try <= phi + 1e-4 * t * slope:
                c, phi, f, act, g = c_try, phi_try, f_try, act_try, g_try
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break

    # 计算精确活性集有效自由度 df = tr(H_A) = p - alpha * tr((alpha I + Ka Ka^T)^{-1})
    if np.any(act):
        Ka = K_sub[:, act]
        H_fin = alpha * eye_p + (Ka @ Ka.T)
        H_fin = 0.5 * (H_fin + H_fin.T)
        try:
            H_inv = np.linalg.inv(H_fin)
            df = float(p - alpha * np.trace(H_inv))
        except np.linalg.LinAlgError:
            df = float(min(p, int(act.sum())))
    else:
        df = 0.0
    df = float(np.clip(df, 0.0, float(p)))
    r_sub = y_core - (K_sub @ f)
    rss_sub = float(np.dot(r_sub, r_sub))
    return f.reshape(N_T1, N_T2), c, df, rss_sub


def reconstruct_brd_fixed(Y_real, ctx, lam=1e-3):
    Y = np.asarray(Y_real, dtype=np.float64)
    scale = float(np.max(np.abs(Y)))
    if scale <= 1e-12:
        return np.zeros((N_T1, N_T2), dtype=np.float64)
    Ys = Y / scale
    Ycore = (ctx["U1"].T @ Ys @ ctx["U2"]).reshape(-1)
    alpha = (float(lam) * ctx["smax"]) ** 2
    M_raw, _, _, _ = _brd_dual_newton(ctx["K_sub"], Ycore, alpha)
    return normalize_area(M_raw)


def reconstruct_brd_gcv(Y_real, ctx, lam_desc=None):
    """
    沿递减 λ 网格（大 λ -> 小 λ 暖启动）执行 Song 2002 对偶牛顿，并按活性集 df 选最小 GCV(λ)。
    """
    if lam_desc is None:
        lam_desc = np.logspace(-1, -4.5, 15)  # 1e-1 down to 3.16e-5 (15 点)
    Y = np.asarray(Y_real, dtype=np.float64)
    scale = float(np.max(np.abs(Y)))
    if scale <= 1e-12:
        return np.zeros((N_T1, N_T2), dtype=np.float64), float(lam_desc[len(lam_desc) // 2])
    Ys = Y / scale
    Ycore = (ctx["U1"].T @ Ys @ ctx["U2"]).reshape(-1)
    r_perp = max(0.0, float(np.sum(Ys * Ys) - np.sum(Ycore * Ycore)))
    n_obs = ctx["n_obs"]
    smax = ctx["smax"]

    c_warm = None
    best_gcv = np.inf
    best_lam = float(lam_desc[len(lam_desc) // 2])
    best_M = None
    for lam in lam_desc:
        alpha = (float(lam) * smax) ** 2
        M_raw, c_warm, df, rss_sub = _brd_dual_newton(ctx["K_sub"], Ycore, alpha, c_init=c_warm)
        if df >= 0.9 * n_obs:
            continue
        gcv = ((r_perp + rss_sub) / n_obs) / ((1.0 - df / n_obs) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)
            best_M = M_raw
    if best_M is None:
        return np.zeros((N_T1, N_T2), dtype=np.float64), best_lam
    return normalize_area(best_M), best_lam


def load_v2_cpu():
    ck = torch.load(os.path.join(PROTO, "v2_refine.pth"), map_location="cpu", weights_only=True)
    refine = RefineNet()
    refine.load_state_dict(ck["refine"])
    refine.eval()
    front = PhysicsFrontEnd()
    front.eval()
    return front, refine


def base_for_cpu(ti_grid, yt):
    """复刻 eval_real_v2.base_for（线性域与 log1p 域返回）用于对比。"""
    K1 = 1.0 - 2.0 * np.exp(-np.asarray(ti_grid)[:, None] / T1_GRID[None, :])
    K2 = np.exp(-np.arange(1, yt.shape[2] + 1)[:, None] * float(TE_MS) / T2_GRID[None, :])
    K1t = torch.tensor(K1, dtype=torch.float32)
    K2t = torch.tensor(K2, dtype=torch.float32)
    U1, S1, V1t = torch.linalg.svd(K1t, full_matrices=False)
    U2, S2, V2t = torch.linalg.svd(K2t, full_matrices=False)
    r1 = int((S1 > S1[0] * 1e-10).sum())
    r2 = int((S2 > S2[0] * 1e-10).sum())
    with torch.no_grad():
        Ycore = torch.einsum("ti,btj->bij", U1[:, :r1], yt)
        Ycore = torch.einsum("bij,jk->bik", Ycore, U2[:, :r2])
        s = S1[:r1][:, None] * S2[:r2][None, :]
        smax = S1[0] * S2[0]
        C = s * Ycore / (s ** 2 + (1e-3 * smax) ** 2).clamp(min=1e-12)
        M = torch.relu(V1t[:r1].t() @ C @ V2t[:r2])
        area = (M.sum() * GRID_AREA).clamp(min=1e-9)
        return torch.log1p(M / area * 20.0), (M / area)[0].numpy()


def main():
    cache = KroneckerSVDCache()
    sim_ctx = cache.get(TI_MS, NECH)
    print(f"Sim SVD cache ready: ridge=({sim_ctx['ridge']['U1'].shape[1]}, {sim_ctx['ridge']['U2'].shape[1]}), "
          f"brd=({sim_ctx['brd']['r1']}, {sim_ctx['brd']['r2']}, p={sim_ctx['brd']['p']})", flush=True)

    # 1. 先检查闭式 SVD-ridge 的尺度不变性与复现一致性 (3 个小样本验证 + 200 留出集)
    front_v2, refine_v2 = load_v2_cpu()

    rng_e = np.random.default_rng(20261112 + 100)
    samples = [make_sample(rng_e) for _ in range(200)]

    # 预热一次
    _ = frozen_baseline_reconstruct(samples[0]["y"], samples[0]["mask"], lam=1e-3)
    _ = reconstruct_svd_ridge_gcv(np.real(samples[0]["y"]), sim_ctx["ridge"])
    _ = reconstruct_brd_fixed(np.real(samples[0]["y"]), sim_ctx["brd"], lam=1e-3)
    _ = reconstruct_brd_gcv(np.real(samples[0]["y"]), sim_ctx["brd"])

    methods = ["svd_ridge_fixed", "svd_ridge_gcv", "brd_nnls_fixed", "brd_nnls_gcv", "v2_frozen"]
    records = {m: [] for m in methods}
    times_s = {m: 0.0 for m in methods}
    lam_logs = {"svd_ridge_gcv": [], "brd_nnls_gcv": []}

    print("Running 200 held-out synthetic samples (rng=20261112+100)...", flush=True)
    for idx, s in enumerate(samples):
        Yr = np.real(s["y"])
        spec = s["spec"]

        # 1) Frozen baseline parity check
        t0 = time.perf_counter()
        M_fix = frozen_baseline_reconstruct(s["y"], s["mask"], lam=1e-3)
        times_s["svd_ridge_fixed"] += time.perf_counter() - t0
        records["svd_ridge_fixed"].append(compute_metrics(M_fix, spec))

        # 2) SVD-ridge + GCV
        t0 = time.perf_counter()
        M_rgcv, lam_r = reconstruct_svd_ridge_gcv(Yr, sim_ctx["ridge"])
        times_s["svd_ridge_gcv"] += time.perf_counter() - t0
        records["svd_ridge_gcv"].append(compute_metrics(M_rgcv, spec))
        lam_logs["svd_ridge_gcv"].append(lam_r)

        # 3) BRD NNLS fixed (λ=1e-3)
        t0 = time.perf_counter()
        M_bfix = reconstruct_brd_fixed(Yr, sim_ctx["brd"], lam=1e-3)
        times_s["brd_nnls_fixed"] += time.perf_counter() - t0
        records["brd_nnls_fixed"].append(compute_metrics(M_bfix, spec))

        # 4) BRD NNLS + active-set GCV
        t0 = time.perf_counter()
        M_bgcv, lam_b = reconstruct_brd_gcv(Yr, sim_ctx["brd"])
        times_s["brd_nnls_gcv"] += time.perf_counter() - t0
        records["brd_nnls_gcv"].append(compute_metrics(M_bgcv, spec))
        lam_logs["brd_nnls_gcv"].append(lam_b)

        # 5) Frozen v2 (CPU wall time reference)
        t0 = time.perf_counter()
        ys = Yr / (np.abs(Yr).max() + 1e-9)
        yt = torch.tensor(ys[None], dtype=torch.float32)
        with torch.no_grad():
            base = front_v2(yt)[0]
            pooled = torch.tensor(log_pool_np(ys[None]), dtype=torch.float32)
            pred_log = refine_v2(base[None], pooled)[0].numpy()
        M_v2 = np.expm1(np.clip(pred_log, 0.0, 20.0)) / 20.0
        times_s["v2_frozen"] += time.perf_counter() - t0
        records["v2_frozen"].append(compute_metrics(M_v2, spec))

        if (idx + 1) % 50 == 0:
            print(f"  progress {idx+1}/200 | svd_gcv={np.mean([x[0] for x in records['svd_ridge_gcv']]):.4f} "
                  f"| brd_fix={np.mean([x[0] for x in records['brd_nnls_fixed']]):.4f} "
                  f"| brd_gcv={np.mean([x[0] for x in records['brd_nnls_gcv']]):.4f}", flush=True)

    synth_summary = {}
    for m in methods:
        arr = np.asarray(records[m], dtype=np.float64)
        synth_summary[m] = {
            "pearson_mean": round(float(arr[:, 0].mean()), 4),
            "pearson_std": round(float(arr[:, 0].std()), 4),
            "pearson_median": round(float(np.median(arr[:, 0])), 4),
            "t1_peak_err_dex_mean": round(float(arr[:, 1].mean()), 4),
            "t1_peak_err_dex_median": round(float(np.median(arr[:, 1])), 4),
            "t2_peak_err_dex_mean": round(float(arr[:, 2].mean()), 4),
            "t2_peak_err_dex_median": round(float(np.median(arr[:, 2])), 4),
            "cpu_total_s": round(float(times_s[m]), 2),
            "cpu_ms_per_sample": round(float(times_s[m] / 200.0 * 1000.0), 2),
        }
        if m in lam_logs:
            lams = np.asarray(lam_logs[m], dtype=np.float64)
            synth_summary[m]["selected_lam_median"] = float(f"{np.median(lams):.4e}")
            synth_summary[m]["selected_lam_min"] = float(f"{np.min(lams):.4e}")
            synth_summary[m]["selected_lam_max"] = float(f"{np.max(lams):.4e}")

    print("\n=== Synthetic n=200 Summary (CPU 1-thread) ===")
    for m, st in synth_summary.items():
        print(f"{m:<18} r={st['pearson_mean']:.4f} | T1={st['t1_peak_err_dex_mean']:.4f}/{st['t1_peak_err_dex_median']:.4f} "
              f"| T2={st['t2_peak_err_dex_mean']:.4f}/{st['t2_peak_err_dex_median']:.4f} | {st['cpu_ms_per_sample']:.2f} ms/sample",
              flush=True)

    # 2. 13 真实样本评测（输入 max|real| 归一化，完全复刻 realdata2d + eval_real_v2 预处理）
    real_rows = []
    dirs = sorted(glob.glob(os.path.join(LIB, "*__IR-CPMG"))) if (load_sample and LIB and os.path.isdir(LIB)) else []
    if not dirs:
        print("Restricted 23 MHz raw waveforms not attached (set NMR23_DATA_DIR or inspect results_archive/adaptive_baseline.json).", flush=True)
        return
    print(f"\nRunning {len(dirs)} real samples...", flush=True)
    for d in dirs:
        s = load_sample(d)
        Yr = np.real(phase_correct(s["Yc"]))
        ys = Yr / (np.abs(Yr).max() + 1e-9)
        ti = s["ti"]
        nech = Yr.shape[1]
        ctx = cache.get(ti, nech)

        # v2 + 1e-3 基线
        yt = torch.tensor(ys[None], dtype=torch.float32)
        with torch.no_grad():
            base_log, base_lin = base_for_cpu(ti, yt)
            pooled = torch.tensor(
                np.nan_to_num(log_pool_np(ys[None]), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            )
            pred_log = refine_v2(base_log, pooled)[0].numpy()
        M_v2 = normalize_area(np.expm1(np.clip(pred_log, 0.0, 20.0)) / 20.0)
        M_base = normalize_area(base_lin)

        M_rgcv, lam_r = reconstruct_svd_ridge_gcv(ys, ctx["ridge"])
        M_bfix = reconstruct_brd_fixed(ys, ctx["brd"], lam=1e-3)
        M_bgcv, lam_b = reconstruct_brd_gcv(ys, ctx["brd"])

        t1_b, t2_b = marginal_peaks_ms(M_base)
        t1_rg, t2_rg = marginal_peaks_ms(M_rgcv)
        t1_bf, t2_bf = marginal_peaks_ms(M_bfix)
        t1_bg, t2_bg = marginal_peaks_ms(M_bgcv)
        t1_v2, t2_v2 = marginal_peaks_ms(M_v2)

        row = {
            "name": s["name"],
            "nti": int(len(ti)),
            "nech": int(nech),
            "r_v2_vs_base": round(float(np.corrcoef(M_v2.ravel(), M_base.ravel())[0, 1]), 4),
            "r_v2_vs_svd_gcv": round(float(np.corrcoef(M_v2.ravel(), M_rgcv.ravel())[0, 1]), 4),
            "r_v2_vs_brd_fix": round(float(np.corrcoef(M_v2.ravel(), M_bfix.ravel())[0, 1]), 4),
            "r_v2_vs_brd_gcv": round(float(np.corrcoef(M_v2.ravel(), M_bgcv.ravel())[0, 1]), 4),
            "r_brd_gcv_vs_base": round(float(np.corrcoef(M_bgcv.ravel(), M_base.ravel())[0, 1]), 4),
            "r_svd_gcv_vs_base": round(float(np.corrcoef(M_rgcv.ravel(), M_base.ravel())[0, 1]), 4),
            "svd_gcv_lam": float(f"{lam_r:.4e}"),
            "brd_gcv_lam": float(f"{lam_b:.4e}"),
            "peaks_ms": {
                "base_fixed": [round(t1_b, 1), round(t2_b, 1)],
                "svd_gcv": [round(t1_rg, 1), round(t2_rg, 1)],
                "brd_fixed": [round(t1_bf, 1), round(t2_bf, 1)],
                "brd_gcv": [round(t1_bg, 1), round(t2_bg, 1)],
                "v2": [round(t1_v2, 1), round(t2_v2, 1)],
            },
        }
        real_rows.append(row)
        print(f"  {s['name']:<14} ({len(ti):>2}x{nech:<4}) r(v2,brd_gcv)={row['r_v2_vs_brd_gcv']:.4f} "
              f"r(v2,base)={row['r_v2_vs_base']:.4f} | v2={t1_v2:>6.1f}/{t2_v2:>6.1f} "
              f"| brd_gcv={t1_bg:>6.1f}/{t2_bg:>6.1f} | lam_b={lam_b:.1e}", flush=True)

    real_summary = {
        "n_samples": len(real_rows),
        "mean_r_v2_vs_base": round(float(np.mean([r["r_v2_vs_base"] for r in real_rows])), 4),
        "mean_r_v2_vs_svd_gcv": round(float(np.mean([r["r_v2_vs_svd_gcv"] for r in real_rows])), 4),
        "mean_r_v2_vs_brd_fix": round(float(np.mean([r["r_v2_vs_brd_fix"] for r in real_rows])), 4),
        "mean_r_v2_vs_brd_gcv": round(float(np.mean([r["r_v2_vs_brd_gcv"] for r in real_rows])), 4),
        "mean_r_brd_gcv_vs_base": round(float(np.mean([r["r_brd_gcv_vs_base"] for r in real_rows])), 4),
        "mean_r_svd_gcv_vs_base": round(float(np.mean([r["r_svd_gcv_vs_base"] for r in real_rows])), 4),
        "samples": real_rows,
    }
    print(f"\nReal 13 means: r(v2,base)={real_summary['mean_r_v2_vs_base']:.4f} "
          f"| r(v2,svd_gcv)={real_summary['mean_r_v2_vs_svd_gcv']:.4f} "
          f"| r(v2,brd_fix)={real_summary['mean_r_v2_vs_brd_fix']:.4f} "
          f"| r(v2,brd_gcv)={real_summary['mean_r_v2_vs_brd_gcv']:.4f}", flush=True)

    out_path = os.path.join(HERE, "results_archive", "adaptive_baseline.json")
    payload = {
        "protocol": {
            "synth_seed": "20261112 + 100",
            "synth_n": 200,
            "metric_domain": "linear area-normalized spectrum",
            "svd_ridge_tol": 1e-10,
            "brd_svd_tol": 1e-6,
            "svd_gcv_lam_grid": "logspace(-5, 0, 81)",
            "brd_gcv_lam_grid": "logspace(-1, -4.5, 15)",
            "threads": 1,
        },
        "synthetic_n200": synth_summary,
        "real_n13": real_summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
