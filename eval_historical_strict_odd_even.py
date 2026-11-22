# -*- coding: utf-8 -*-
"""
Read-only strict Odd->Even offline re-evaluation of the post-freeze consolidated
23 MHz Niumag archival cohorts using frozen Paper #2 weights (v2_refine.pth, 19,393 params).

Protocol parity:
  - Mirrors ReproBundle_paper2 real_uncertainty_and_standards.json protocol:
    splits complex matrix `Yc` BEFORE phasing (`Yc_o = Yc[:, 0::2]`, `Yc_e = Yc[:, 1::2]`),
    estimates `ph_o` solely from `Yc_o` and `ph_e` solely from `Yc_e`, calibrates amplitude
    normalization `scale_o = max|Yo_s|` and forward scalar `alpha_o` solely on odd echoes,
    and evaluates strict zero-even-leakage `odd_to_even_strict_nrmse` on `Ye_on_o_phase`.
  - Also records `odd_to_even_aligned_nrmse` and `odd_to_even_shared_phase_aligned_nrmse`
    to transparently bridge to prior shared-phase aligned diagnostics.

Dataset sections (Supplementary Note S8.4 / Table S13):
  1. Core cohort (`core_liver_lard_n11`): 11 porcine liver-lard fat-addition phantoms with
     paired standalone 1D IR (`.1.irp`) & 1D CPMG (`.1.pea`), primary axis `added_lard_wt_pct`.
  2. Supplementary single case (`supplementary_lard_n1`): archived lard (`15x1000`).
  3. Separate acquisition-condition case (`separate_condition_c57_n1`): `10x5000`
     oil-family standard (`NECH = 5000 > 2000` pooling window).

Modes:
  - Default offline mode (no `--archive-root` / unset `NMR2_HISTORICAL_ARCHIVE_DIR`):
    loads and validates sanitized `results_archive/historical_strict_odd_even_eval.json`
    and prints the Table S13 summary.
  - Raw-archive re-evaluation mode (`--archive-root <DIR>` or `NMR2_HISTORICAL_ARCHIVE_DIR`,
    with optional `--layout-json`): deterministically recomputes all metrics from restricted
    raw instrument directories and writes sanitized JSON to `--out`.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import spearmanr, wilcoxon

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_ARCHIVE_JSON = os.path.join(HERE, "results_archive", "historical_strict_odd_even_eval.json")
DEFAULT_LAYOUT = {
    "liver_lard_2d_rel": "liver_lard/2d/sample_{idx:02d}",
    "liver_lard_1d_cpmg_rel": "liver_lard/1d_cpmg/sample_{idx:02d}",
    "liver_lard_1d_ir_rel": "liver_lard/1d_ir/sample_{idx:02d}",
    "lard_2d_rel": "lard_01/2d",
    "lard_1d_rel": "lard_01/1d_cpmg",
    "lard_xls_rel": "lard_01/output.xls",
    "c57_2d_rel": "std_c57/2d",
    "c57_prefix": "C57H98012-S",
}

from sim2d import N_T1, N_T2, T1_GRID, T2_GRID, LOG_T1, LOG_T2, DL1, DL2  # noqa: E402
from train2d_v2 import RefineNet, N_POOL, POOLIDX, log_pool_np  # noqa: E402
import adaptive_lambda_baseline as alb  # noqa: E402

GRID_AREA = float(DL1 * DL2)
SOLVER_ORDER = [
    "svd_ridge_fixed",
    "svd_ridge_gcv",
    "brd_nnls_fixed_1e-3",
    "brd_nnls_fixed_1e-2",
    "brd_nnls_gcv",
    "v2_phys",
    "v2_orig",
    "v2_zero",
]
COMPARISON_BASELINES = [
    "svd_ridge_fixed",
    "brd_nnls_fixed_1e-3",
    "brd_nnls_fixed_1e-2",
    "brd_nnls_gcv",
]


def load_frozen_v2():
    ck = torch.load(os.path.join(HERE, "v2_refine.pth"), map_location="cpu", weights_only=True)
    net = RefineNet()
    net.load_state_dict(ck["refine"])
    net.eval()
    return net


def norm_area(M):
    M = np.clip(np.asarray(M, dtype=np.float64), 0.0, None)
    a = float(M.sum()) * GRID_AREA
    return M / a if a > 1e-12 else M


def read_par(path):
    d = {}
    with open(path, encoding="gbk", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if "=" in ln:
                k, v = ln.split("=", 1)
                d[k.strip()] = v.strip()
            else:
                parts = ln.split(None, 1)
                if len(parts) == 2:
                    d[parts[0].strip()] = parts[1].strip()
    return d


def par_float(par, keys, default=None):
    for k in keys:
        if k in par and par[k] != "":
            return float(par[k])
    return default


def read_col4(path):
    """Parse 4-column Niumag text files (.pea or .irp): Time, Amplitude, Real, Imag."""
    t, a, c = [], [], []
    with open(path, encoding="gbk", errors="replace") as f:
        for ln in f:
            p = ln.split()
            if len(p) == 4:
                try:
                    t.append(float(p[0]))
                    a.append(float(p[1]))
                    c.append(complex(float(p[2]), float(p[3])))
                except ValueError:
                    pass
    return np.array(t, dtype=np.float64), np.array(a, dtype=np.float64), np.array(c, dtype=np.complex128)


def estimate_phase_rad(Yc):
    ref = Yc[-1] / max(float(np.abs(Yc[-1]).mean()), 1e-9)
    return float(np.angle(np.mean(ref)))


def phase_correct(Yc):
    ph = estimate_phase_rad(Yc)
    return np.real(Yc * np.exp(-1j * ph)), ph


def phase_correct_1d(c, ref_idx=-1):
    ph = float(np.angle(c[ref_idx]))
    return np.real(c * np.exp(-1j * ph))


def load_pea_matrix(sample_dir, prefix=None):
    pat_par = f"{prefix}.par" if prefix else "*.par"
    pars = sorted(glob.glob(os.path.join(sample_dir, pat_par)))
    if not pars and prefix:
        pars = sorted(glob.glob(os.path.join(sample_dir, "*.par")))
    if not pars:
        pars = sorted(glob.glob(os.path.join(os.path.dirname(sample_dir), pat_par)))
    par = read_par(pars[0]) if pars else {}
    d1s = par.get("D1S", "")
    ti = (
        np.array([float(x) for x in d1s.replace(",", " ").split() if float(x) > 0], dtype=np.float64)
        if d1s
        else None
    )
    pat = f"{prefix}.1.*.pea" if prefix else "*.1.*.pea"
    peas = sorted(
        glob.glob(os.path.join(sample_dir, pat)),
        key=lambda p: int(re.search(r"\.1\.(\d+)\.pea$", p).group(1)),
    )
    curves, amps = [], []
    t0 = None
    for f in peas:
        t, a, c = read_col4(f)
        if t0 is None:
            t0 = t
        curves.append(c)
        amps.append(a)
    Yc = np.array(curves, dtype=np.complex128)
    Ya = np.array(amps, dtype=np.float64)
    return {"ti": ti, "t_echo": t0, "Yc": Yc, "Ya": Ya, "par": par, "n_pea": len(peas)}


def read_ircp_file(path):
    lines = open(path, encoding="gbk", errors="replace").read().splitlines()
    nech = int(float(lines[1].strip()))
    nti = int(float(lines[2].strip()))
    ti_vals = [float(x) for x in lines[3].split()]
    ti = np.array(
        ti_vals[1:] if (len(ti_vals) == nti + 1 and ti_vals[0] == 0.0) else ti_vals[:nti],
        dtype=np.float64,
    )
    t_echo, mat = [], []
    for ln in lines[4 : 4 + nech]:
        p = [float(x) for x in ln.split()]
        if len(p) >= nti + 1:
            t_echo.append(p[0])
            mat.append(p[1 : 1 + nti])
    return ti, np.array(t_echo, dtype=np.float64), np.array(mat, dtype=np.float64).T


def pool_dynamic_time(ys, echo_ms, pool_te_ms=1.027):
    pooled_np = np.zeros((1, ys.shape[0], N_POOL), dtype=np.float32)
    em = np.asarray(echo_ms, dtype=np.float64)
    te_p = float(pool_te_ms)
    for j, (a, b) in enumerate(POOLIDX):
        lo = 0.0 if a == 0 else (a + 0.5) * te_p
        hi = (b + 0.5) * te_p
        mask = (em >= lo) & (em < hi)
        if np.any(mask):
            pooled_np[:, :, j] = ys[:, mask].mean(axis=1)
    return pooled_np


def make_contexts(ti_ms, echo_ms, pool_te_ms, ridge_tol=1e-10, brd_tol=1e-6):
    ti = np.asarray(ti_ms, dtype=np.float64)
    em = np.asarray(echo_ms, dtype=np.float64)
    K1 = 1.0 - 2.0 * np.exp(-ti[:, None] / T1_GRID[None, :])
    K2 = np.exp(-em[:, None] / T2_GRID[None, :])

    U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
    U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
    smax = float(S1[0] * S2[0])
    n_obs = int(len(ti) * len(em))

    r1_r = int((S1 > S1[0] * ridge_tol).sum())
    r2_r = int((S2 > S2[0] * ridge_tol).sum())
    ridge_ctx = {
        "U1": U1[:, :r1_r],
        "S1": S1[:r1_r],
        "V1t": V1t[:r1_r],
        "U2": U2[:, :r2_r],
        "S2": S2[:r2_r],
        "V2t": V2t[:r2_r],
        "s_prod": S1[:r1_r, None] * S2[:r2_r][None, :],
        "smax": smax,
        "n_obs": n_obs,
    }

    r1_b = int((S1 > S1[0] * brd_tol).sum())
    r2_b = int((S2 > S2[0] * brd_tol).sum())
    V1_b, V2_b = V1t[:r1_b], V2t[:r2_b]
    K_sub = (
        (S1[:r1_b, None] * V1_b)[:, None, :, None]
        * (S2[:r2_b, None] * V2_b)[None, :, None, :]
    ).reshape(r1_b * r2_b, N_T1 * N_T2)
    brd_ctx = {
        "U1": U1[:, :r1_b],
        "U2": U2[:, :r2_b],
        "K_sub": np.ascontiguousarray(K_sub, dtype=np.float64),
        "p": r1_b * r2_b,
        "smax": smax,
        "n_obs": n_obs,
        "r1": r1_b,
        "r2": r2_b,
    }

    K1t = torch.tensor(K1, dtype=torch.float32)
    K2t = torch.tensor(K2, dtype=torch.float32)
    U1t, S1t, V1tt = torch.linalg.svd(K1t, full_matrices=False)
    U2t, S2t, V2tt = torch.linalg.svd(K2t, full_matrices=False)
    r1_t = int((S1t > S1t[0] * 1e-10).sum())
    r2_t = int((S2t > S2t[0] * 1e-10).sum())
    v2_ctx = {
        "U1": U1t[:, :r1_t],
        "S1": S1t[:r1_t],
        "V1t": V1tt[:r1_t],
        "U2": U2t[:, :r2_t],
        "S2": S2t[:r2_t],
        "V2t": V2tt[:r2_t],
        "s_prod": S1t[:r1_t][:, None] * S2t[:r2_t][None, :],
        "smax": float(S1t[0] * S2t[0]),
        "echo_ms": em,
        "pool_te_ms": float(pool_te_ms),
    }
    return K1, K2, ridge_ctx, brd_ctx, v2_ctx


def run_svd_fixed(Yr, ridge_ctx, lam=1e-3):
    Yc = ridge_ctx["U1"].T @ Yr @ ridge_ctx["U2"]
    s = ridge_ctx["s_prod"]
    denom = s ** 2 + (lam * ridge_ctx["smax"]) ** 2
    C = np.where(denom > 0.0, s * Yc / np.where(denom > 0.0, denom, 1.0), 0.0)
    return norm_area(ridge_ctx["V1t"].T @ C @ ridge_ctx["V2t"])


def run_v2(Yr, v2_ctx, refine_net, mode="phys"):
    ys = np.real(Yr) / (np.abs(np.real(Yr)).max() + 1e-9)
    yt = torch.tensor(ys[None], dtype=torch.float32)
    with torch.no_grad():
        Yc = torch.einsum("ti,btj->bij", v2_ctx["U1"], yt)
        Yc = torch.einsum("bij,jk->bik", Yc, v2_ctx["U2"])
        s = v2_ctx["s_prod"]
        smax = v2_ctx["smax"]
        C = s * Yc / (s ** 2 + (1e-3 * smax) ** 2).clamp(min=1e-12)
        M_base = torch.relu(v2_ctx["V1t"].t() @ C @ v2_ctx["V2t"])
        area = (M_base.sum() * GRID_AREA).clamp(min=1e-9)
        base_log = torch.log1p(M_base / area * 20.0)
        if mode == "orig":
            pooled_np = log_pool_np(ys[None])
        elif mode == "phys":
            pooled_np = pool_dynamic_time(ys, v2_ctx["echo_ms"], pool_te_ms=v2_ctx["pool_te_ms"])
        elif mode == "zero":
            pooled_np = np.zeros((1, ys.shape[0], N_POOL), dtype=np.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        pred_log = refine_net(
            base_log,
            torch.tensor(np.nan_to_num(pooled_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32),
        )[0].numpy()
        pred = np.expm1(np.clip(pred_log, 0.0, 20.0)) / 20.0
        raw_area = float(np.clip(pred, 0.0, None).sum()) * GRID_AREA
    return norm_area(pred), raw_area, pooled_np


def fwd_nrmse_strict(M_odd, K1, K2_odd, K2_even, Yo_s, Ye_on_o_phase):
    """Exact baseline zero-even-leakage Odd->Even NRMSE."""
    scale_o = float(np.max(np.abs(Yo_s))) + 1e-9
    Yo_n = Yo_s / scale_o
    Ye_n = Ye_on_o_phase / scale_o
    ps_o = (K1 @ M_odd @ K2_odd.T) * GRID_AREA
    alpha_o = max(0.0, float(np.sum(ps_o * Yo_n) / max(float(np.sum(ps_o ** 2)), 1e-12)))
    ps_e = (K1 @ M_odd @ K2_even.T) * GRID_AREA
    return float(np.sqrt(np.mean((alpha_o * ps_e - Ye_n) ** 2)))


def fwd_nrmse_aligned(M, K1_eval, K2_eval, Y_eval):
    """Even-echo (or full-scan) amplitude-aligned forward NRMSE."""
    Y_norm = Y_eval / (float(np.max(np.abs(Y_eval))) + 1e-9)
    ps = (K1_eval @ M @ K2_eval.T) * GRID_AREA
    alpha = max(0.0, float(np.sum(ps * Y_norm) / max(float(np.sum(ps ** 2)), 1e-12)))
    return float(np.sqrt(np.mean((alpha * ps - Y_norm) ** 2)))


def fwhm_both(curve, grid_log):
    c = np.asarray(curve, dtype=np.float64)
    if c.max() <= 0:
        return 0.0, 0.0
    pk = int(np.argmax(c))
    hm = 0.5 * c[pk]
    l = pk
    while l > 0 and c[l] >= hm:
        l -= 1
    r = pk
    while r < len(c) - 1 and c[r] >= hm:
        r += 1
    return float(10.0 ** grid_log[r] - 10.0 ** grid_log[l]), float(grid_log[r] - grid_log[l])


def spec_stats(M):
    M = norm_area(M)
    tot = float(M.sum()) * GRID_AREA
    if tot <= 1e-12:
        raise ValueError("Zero-mass spectrum encountered")
    p1 = M.sum(axis=1) * DL2
    p2 = M.sum(axis=0) * DL1
    p1n = p1 / float(p1.sum())
    p2n = p2 / float(p2.sum())

    t1_pk = float(T1_GRID[int(np.argmax(p1))])
    t2_pk = float(T2_GRID[int(np.argmax(p2))])
    t1_lm = float(10.0 ** float(np.sum(p1n * LOG_T1)))
    t2_lm = float(10.0 ** float(np.sum(p2n * LOG_T2)))
    fw1_ms, fw1_dex = fwhm_both(p1, LOG_T1)
    fw2_ms, fw2_dex = fwhm_both(p2, LOG_T2)

    b1_lo = float(M[:2, :].sum() * GRID_AREA / tot * 100.0)
    b1_hi = float(M[-2:, :].sum() * GRID_AREA / tot * 100.0)
    b2_lo = float(M[:, :2].sum() * GRID_AREA / tot * 100.0)
    b2_hi = float(M[:, -2:].sum() * GRID_AREA / tot * 100.0)
    bmax = max(b1_lo, b1_hi, b2_lo, b2_hi)
    m_t2_lt10 = float(p2n[T2_GRID < 10.0].sum() * 100.0)

    return {
        "T1_peak_ms": round(t1_pk, 2),
        "T2_peak_ms": round(t2_pk, 2),
        "T1_lm_ms": round(t1_lm, 2),
        "T2_lm_ms": round(t2_lm, 2),
        "T1_fwhm_ms": round(fw1_ms, 2),
        "T2_fwhm_ms": round(fw2_ms, 2),
        "T1_fwhm_dex": round(fw1_dex, 4),
        "T2_fwhm_dex": round(fw2_dex, 4),
        "bdry_T1_lo_pct": round(b1_lo, 2),
        "bdry_T1_hi_pct": round(b1_hi, 2),
        "bdry_T2_lo_pct": round(b2_lo, 2),
        "bdry_T2_hi_pct": round(b2_hi, 2),
        "bdry_max_pct": round(bmax, 2),
        "mass_T2_lt_10ms_pct": round(m_t2_lt10, 2),
    }


def solve_one(solver_name, Yr, r_ctx, b_ctx, v_ctx, refine_net):
    if solver_name == "svd_ridge_fixed":
        return run_svd_fixed(Yr, r_ctx, lam=1e-3), {}
    if solver_name == "svd_ridge_gcv":
        M, lam = alb.reconstruct_svd_ridge_gcv(Yr, r_ctx)
        return norm_area(M), {"lam": float(lam)}
    if solver_name == "brd_nnls_fixed_1e-3":
        return norm_area(alb.reconstruct_brd_fixed(Yr, b_ctx, lam=1e-3)), {}
    if solver_name == "brd_nnls_fixed_1e-2":
        return norm_area(alb.reconstruct_brd_fixed(Yr, b_ctx, lam=1e-2)), {}
    if solver_name == "brd_nnls_gcv":
        M, lam = alb.reconstruct_brd_gcv(Yr, b_ctx)
        return norm_area(M), {"lam": float(lam)}
    if solver_name == "v2_phys":
        M, ra, p_np = run_v2(Yr, v_ctx, refine_net, mode="phys")
        return M, {"raw_area": ra, "pooled": p_np}
    if solver_name == "v2_orig":
        M, ra, p_np = run_v2(Yr, v_ctx, refine_net, mode="orig")
        return M, {"raw_area": ra, "pooled": p_np}
    if solver_name == "v2_zero":
        M, ra, _ = run_v2(Yr, v_ctx, refine_net, mode="zero")
        return M, {"raw_area": ra}
    raise ValueError(f"Unknown solver: {solver_name}")


def phase_diagnostics(Yc):
    Yr_full, ph_full = phase_correct(Yc)
    ph_o = estimate_phase_rad(Yc[:, 0::2])
    ph_e = estimate_phase_rad(Yc[:, 1::2])
    Zr = Yc * np.exp(-1j * ph_full)
    re_e = float(np.sum(np.real(Zr) ** 2))
    im_e = float(np.sum(np.imag(Zr) ** 2))
    tail_cols = min(200, Yc.shape[1])
    tail_im = np.imag(Zr)[:, -tail_cols:]
    snr = float(np.max(np.abs(Yr_full)) / (float(np.std(tail_im)) + 1e-12))

    # Row-wise residual phase scatter on high-amplitude rows (mod pi for IR zero-crossing)
    row_ph = []
    n_head = min(20, Yc.shape[1])
    max_abs = float(np.max(np.abs(Zr)))
    for i in range(Yc.shape[0]):
        v = np.mean(Zr[i, :n_head])
        if abs(v) > 0.05 * max_abs:
            ang = math.degrees(np.angle(v))
            if ang > 90.0:
                ang -= 180.0
            elif ang < -90.0:
                ang += 180.0
            row_ph.append(ang)

    dph = ((math.degrees(ph_o - ph_e) + 180.0) % 360.0) - 180.0
    return Yr_full, {
        "ph_full_deg": round(math.degrees(ph_full), 2),
        "ph_odd_deg": round(math.degrees(ph_o), 2),
        "ph_even_deg": round(math.degrees(ph_e), 2),
        "delta_ph_odd_even_deg": round(dph, 3),
        "imag_over_real_rms": round(math.sqrt(im_e / (re_e + 1e-30)), 5),
        "row_phase_std_deg": round(float(np.std(row_ph)), 3) if row_ph else 0.0,
        "row_phase_min_deg": round(float(np.min(row_ph)), 3) if row_ph else 0.0,
        "row_phase_max_deg": round(float(np.max(row_ph)), 3) if row_ph else 0.0,
        "snr_complex": round(snr, 2),
    }


def evaluate_matrix_benchmark(Yc, ti_ms, te_ms, refine_net, Yr_override=None):
    """Run all 8 solvers under exact strict Odd->Even + shared-phase diagnostic protocols."""
    if Yr_override is None:
        Yr_full, pdiag = phase_diagnostics(Yc)
        Yc_o, Yc_e = Yc[:, 0::2], Yc[:, 1::2]
        Yo_s, _ = phase_correct(Yc_o)
        ph_o = estimate_phase_rad(Yc_o)
        Ye_on_o_phase = np.real(Yc_e * np.exp(-1j * ph_o))
        Ye_s, _ = phase_correct(Yc_e)
        Yo_sh, Ye_sh = Yr_full[:, 0::2], Yr_full[:, 1::2]
    else:
        Yr_full = np.asarray(Yr_override, dtype=np.float64)
        pdiag = {"source": "magnitude_signed_matrix"}
        Yo_s = Yr_full[:, 0::2]
        Ye_on_o_phase = Yr_full[:, 1::2]
        Ye_s = Yr_full[:, 1::2]
        Yo_sh, Ye_sh = Yo_s, Ye_s

    nech = Yr_full.shape[1]
    em = np.arange(1, nech + 1, dtype=np.float64) * float(te_ms)
    em_o, em_e = em[0::2], em[1::2]

    K1, K2_f, rc_f, bc_f, vc_f = make_contexts(ti_ms, em, pool_te_ms=te_ms)
    _, K2_o, rc_o, bc_o, vc_o = make_contexts(ti_ms, em_o, pool_te_ms=te_ms)
    _, K2_e, rc_e, bc_e, vc_e = make_contexts(ti_ms, em_e, pool_te_ms=te_ms)

    solvers_out = {}
    full_maps = {}
    full_pooled = {}

    for sname in SOLVER_ORDER:
        Mf, inf_f = solve_one(sname, Yr_full, rc_f, bc_f, vc_f, refine_net)
        Mo, inf_o = solve_one(sname, Yo_s, rc_o, bc_o, vc_o, refine_net)
        Me, inf_e = solve_one(sname, Ye_s, rc_e, bc_e, vc_e, refine_net)

        if Yr_override is None:
            Mo_sh, _ = solve_one(sname, Yo_sh, rc_o, bc_o, vc_o, refine_net)
            shared_al_nrmse = fwd_nrmse_aligned(Mo_sh, K1, K2_e, Ye_sh)
        else:
            shared_al_nrmse = fwd_nrmse_aligned(Mo, K1, K2_e, Ye_sh)

        st = spec_stats(Mf)
        st["full_fwd_nrmse"] = round(fwd_nrmse_aligned(Mf, K1, K2_f, Yr_full), 5)
        st["odd_to_even_strict_nrmse"] = round(fwd_nrmse_strict(Mo, K1, K2_o, K2_e, Yo_s, Ye_on_o_phase), 5)
        st["odd_to_even_aligned_nrmse"] = round(fwd_nrmse_aligned(Mo, K1, K2_e, Ye_on_o_phase), 5)
        st["odd_to_even_shared_phase_aligned_nrmse"] = round(shared_al_nrmse, 5)
        st["echo_split_r"] = round(float(np.corrcoef(Mo.ravel(), Me.ravel())[0, 1]), 5)
        if "raw_area" in inf_f:
            st["raw_area"] = round(float(inf_f["raw_area"]), 4)
        if "pooled" in inf_f:
            full_pooled[sname] = inf_f["pooled"]
        if "lam" in inf_f:
            st["selected_lam_full"] = float(f"{inf_f['lam']:.4e}")
            st["selected_lam_odd"] = float(f"{inf_o['lam']:.4e}")
            st["selected_lam_even"] = float(f"{inf_e['lam']:.4e}")
            if sname == "svd_ridge_gcv":
                st["hit_lower_bound_full"] = bool(inf_f["lam"] <= 1.01e-5)
            elif sname == "brd_nnls_gcv":
                st["hit_lower_bound_full"] = bool(inf_f["lam"] <= 3.20e-5)

        full_maps[sname] = Mf
        solvers_out[sname] = st

    max_pool_diff = float(np.max(np.abs(full_pooled["v2_orig"] - full_pooled["v2_phys"])))
    max_spec_diff = float(np.max(np.abs(full_maps["v2_orig"] - full_maps["v2_phys"])))
    return {
        "phase_diagnostics": pdiag,
        "nech_exceeds_pool_window": bool(nech > 2000),
        "orig_vs_phys_max_pooled_diff": max_pool_diff,
        "orig_vs_phys_max_spec_diff": max_spec_diff,
        "solvers": solvers_out,
    }


def fit_separable_monoexp(Yr, ti_ms, echo_ms, init_t1=200.0, init_t2=80.0):
    _, S, _ = np.linalg.svd(Yr, full_matrices=False)
    rank1_energy = float((S[0] ** 2) / max(float(np.sum(S ** 2)), 1e-30))
    sc = float(np.max(np.abs(Yr))) + 1e-12
    Yn = Yr / sc

    def res(p):
        A, b, lt1, lt2 = p
        pred = A * np.outer(1.0 - b * np.exp(-ti_ms / (10.0 ** lt1)), np.exp(-echo_ms / (10.0 ** lt2)))
        return (pred - Yn).ravel()

    sol = least_squares(
        res,
        [1.0, 1.95, np.log10(init_t1), np.log10(init_t2)],
        bounds=([0.05, 0.3, -1.0, -1.0], [20.0, 2.5, 4.5, 4.5]),
        ftol=1e-10,
        xtol=1e-10,
    )
    A, b, lt1, lt2 = sol.x
    T1_fit = float(10.0 ** lt1)
    T2_fit = float(10.0 ** lt2)
    pred = A * np.outer(1.0 - b * np.exp(-ti_ms / T1_fit), np.exp(-echo_ms / T2_fit))
    r2 = 1.0 - float(np.sum((Yn - pred) ** 2) / max(float(np.sum((Yn - Yn.mean()) ** 2)), 1e-30))
    nrmse = float(np.sqrt(np.mean((Yn - pred) ** 2)))
    return {
        "svd_rank1_energy_ratio": round(rank1_energy, 6),
        "T1_ms": round(T1_fit, 2),
        "T2_ms": round(T2_fit, 2),
        "b": round(float(b), 4),
        "R2": round(r2, 6),
        "NRMSE": round(nrmse, 5),
    }


def fit_1d_ir_monoexp(ti_ms, c_ir):
    y = phase_correct_1d(c_ir, -1)
    y = y / (float(np.max(np.abs(y))) + 1e-12)
    sol = least_squares(
        lambda p: p[0] * (1.0 - p[1] * np.exp(-ti_ms / p[2])) - y,
        [1.0, 2.0, 220.0],
        bounds=([0.05, 0.5, 5.0], [5.0, 2.5, 5000.0]),
        ftol=1e-10,
        xtol=1e-10,
    )
    A, b, t1 = [float(v) for v in sol.x]
    pred = A * (1.0 - b * np.exp(-ti_ms / t1))
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / max(float(np.sum((y - y.mean()) ** 2)), 1e-30))
    nrmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {"T1_1D_ms": round(t1, 2), "b_1D": round(b, 4), "R2": round(r2, 6), "NRMSE": round(nrmse, 5)}


def fit_1d_cpmg_monoexp(t_ms, c_cpmg):
    y = phase_correct_1d(c_cpmg, 0)
    y = y / (float(np.max(np.abs(y))) + 1e-12)
    sol = least_squares(
        lambda p: p[0] * np.exp(-t_ms / p[1]) - y,
        [1.0, 55.0],
        bounds=([0.05, 1.0], [5.0, 5000.0]),
        ftol=1e-10,
        xtol=1e-10,
    )
    A, t2 = [float(v) for v in sol.x]
    pred = A * np.exp(-t_ms / t2)
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / max(float(np.sum((y - y.mean()) ** 2)), 1e-30))
    nrmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {"T2_1D_ms": round(t2, 2), "R2": round(r2, 6), "NRMSE": round(nrmse, 5)}


def vendor_sheet3_peaks(xls_path):
    with zipfile.ZipFile(xls_path) as z:
        sheet = ET.fromstring(z.read("xl/worksheets/sheet3.xml"))
        rows = []
        for r in sheet.iter():
            if r.tag.endswith("}row"):
                vals = [next((x.text for x in c if x.tag.endswith("}v")), None) for c in r]
                vals = [v for v in vals if v is not None]
                if len(vals) == 4:
                    try:
                        rows.append([float(x) for x in vals])
                    except ValueError:
                        pass
    arr = np.array(rows, dtype=np.float64)
    return {
        "T1_proj_peak_ms": round(float(arr[int(np.argmax(arr[:, 1])), 0] * 1000.0), 2),
        "T2_proj_peak_ms": round(float(arr[int(np.argmax(arr[:, 3])), 2] * 1000.0), 2),
    }


def safe_wilcoxon(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.allclose(a, b, atol=1e-12, rtol=0.0):
        return {"W": 0.0, "p": 1.0}
    res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"W": round(float(res.statistic), 4), "p": float(f"{res.pvalue:.4e}")}


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "p": None, "constant_input": True}
    res = spearmanr(x, y)
    return {
        "rho": round(float(res.statistic), 4),
        "p": float(f"{res.pvalue:.4e}"),
    }


def summarize_liver_cohort(samples):
    wt_pct = [s["composition"]["added_lard_wt_pct"] for s in samples]
    add_g = [s["composition"]["added_lard_g"] for s in samples]
    t1_1d = [s["paired_1d_ir"]["T1_1D_ms"] for s in samples]
    t2_1d = [s["paired_1d_cpmg"]["T2_1D_ms"] for s in samples]
    t1_2d = [s["monoexp_2d"]["T1_ms"] for s in samples]
    t2_2d = [s["monoexp_2d"]["T2_ms"] for s in samples]

    ref_trends = {
        "vs_added_lard_wt_pct": {
            "1D_IR_T1_ms": safe_spearman(wt_pct, t1_1d),
            "1D_CPMG_T2_ms": safe_spearman(wt_pct, t2_1d),
            "2D_monoexp_T1_ms": safe_spearman(wt_pct, t1_2d),
            "2D_monoexp_T2_ms": safe_spearman(wt_pct, t2_2d),
        },
        "vs_added_lard_g": {
            "1D_IR_T1_ms": safe_spearman(add_g, t1_1d),
            "1D_CPMG_T2_ms": safe_spearman(add_g, t2_1d),
            "2D_monoexp_T1_ms": safe_spearman(add_g, t1_2d),
            "2D_monoexp_T2_ms": safe_spearman(add_g, t2_2d),
        },
    }

    solver_summaries = {}
    per_solver_errs = {}
    for sname in SOLVER_ORDER:
        rows = [s["solvers"][sname] for s in samples]
        oe_s = [r["odd_to_even_strict_nrmse"] for r in rows]
        oe_a = [r["odd_to_even_aligned_nrmse"] for r in rows]
        oe_sh = [r["odd_to_even_shared_phase_aligned_nrmse"] for r in rows]
        spl = [r["echo_split_r"] for r in rows]

        e_t1_pk = [abs(math.log10(r["T1_peak_ms"] / t1r)) for r, t1r in zip(rows, t1_1d)]
        e_t1_lm = [abs(math.log10(r["T1_lm_ms"] / t1r)) for r, t1r in zip(rows, t1_1d)]
        e_t2_pk = [abs(math.log10(r["T2_peak_ms"] / t2r)) for r, t2r in zip(rows, t2_1d)]
        e_t2_lm = [abs(math.log10(r["T2_lm_ms"] / t2r)) for r, t2r in zip(rows, t2_1d)]
        per_solver_errs[sname] = {
            "oe_strict": oe_s,
            "e_t1_pk": e_t1_pk,
            "e_t1_lm": e_t1_lm,
            "e_t2_pk": e_t2_pk,
            "e_t2_lm": e_t2_lm,
        }

        ssum = {
            "odd_to_even_strict_nrmse": {
                "mean": round(float(np.mean(oe_s)), 5),
                "median": round(float(np.median(oe_s)), 5),
                "max": round(float(np.max(oe_s)), 5),
            },
            "odd_to_even_aligned_nrmse": {
                "mean": round(float(np.mean(oe_a)), 5),
                "median": round(float(np.median(oe_a)), 5),
                "max": round(float(np.max(oe_a)), 5),
            },
            "odd_to_even_shared_phase_aligned_nrmse": {
                "mean": round(float(np.mean(oe_sh)), 5),
                "median": round(float(np.median(oe_sh)), 5),
                "max": round(float(np.max(oe_sh)), 5),
            },
            "echo_split_r": {
                "mean": round(float(np.mean(spl)), 5),
                "min": round(float(np.min(spl)), 5),
                "median": round(float(np.median(spl)), 5),
            },
            "abs_dlog10_T1_vs_1D_IR": {
                "T1_peak": {
                    "mean_dex": round(float(np.mean(e_t1_pk)), 4),
                    "median_dex": round(float(np.median(e_t1_pk)), 4),
                    "mean_steps": round(float(np.mean(e_t1_pk) / DL1), 2),
                    "median_steps": round(float(np.median(e_t1_pk) / DL1), 2),
                },
                "T1_lm": {
                    "mean_dex": round(float(np.mean(e_t1_lm)), 4),
                    "median_dex": round(float(np.median(e_t1_lm)), 4),
                    "mean_steps": round(float(np.mean(e_t1_lm) / DL1), 2),
                    "median_steps": round(float(np.median(e_t1_lm) / DL1), 2),
                },
            },
            "abs_dlog10_T2_vs_1D_CPMG": {
                "T2_peak": {
                    "mean_dex": round(float(np.mean(e_t2_pk)), 4),
                    "median_dex": round(float(np.median(e_t2_pk)), 4),
                    "mean_steps": round(float(np.mean(e_t2_pk) / DL2), 2),
                    "median_steps": round(float(np.median(e_t2_pk) / DL2), 2),
                },
                "T2_lm": {
                    "mean_dex": round(float(np.mean(e_t2_lm)), 4),
                    "median_dex": round(float(np.median(e_t2_lm)), 4),
                    "mean_steps": round(float(np.mean(e_t2_lm) / DL2), 2),
                    "median_steps": round(float(np.median(e_t2_lm) / DL2), 2),
                },
            },
            "spearman_vs_added_lard_wt_pct": {
                "T1_peak_ms": safe_spearman(wt_pct, [r["T1_peak_ms"] for r in rows]),
                "T1_lm_ms": safe_spearman(wt_pct, [r["T1_lm_ms"] for r in rows]),
                "T2_peak_ms": safe_spearman(wt_pct, [r["T2_peak_ms"] for r in rows]),
                "T2_lm_ms": safe_spearman(wt_pct, [r["T2_lm_ms"] for r in rows]),
            },
            "spearman_vs_added_lard_g": {
                "T1_peak_ms": safe_spearman(add_g, [r["T1_peak_ms"] for r in rows]),
                "T1_lm_ms": safe_spearman(add_g, [r["T1_lm_ms"] for r in rows]),
                "T2_peak_ms": safe_spearman(add_g, [r["T2_peak_ms"] for r in rows]),
                "T2_lm_ms": safe_spearman(add_g, [r["T2_lm_ms"] for r in rows]),
            },
            "max_bdry_pct": round(float(max(r["bdry_max_pct"] for r in rows)), 2),
            "mean_mass_T2_lt_10ms_pct": round(float(np.mean([r["mass_T2_lt_10ms_pct"] for r in rows])), 2),
            "max_mass_T2_lt_10ms_pct": round(float(np.max([r["mass_T2_lt_10ms_pct"] for r in rows])), 2),
        }
        if "selected_lam_full" in rows[0]:
            ssum["gcv_lower_bound_hits"] = {
                "full": int(sum(bool(r["hit_lower_bound_full"]) for r in rows)),
                "n_total": len(rows),
                "unique_lams_full": sorted(set(r["selected_lam_full"] for r in rows)),
                "unique_lams_odd": sorted(set(r["selected_lam_odd"] for r in rows)),
                "unique_lams_even": sorted(set(r["selected_lam_even"] for r in rows)),
            }
        solver_summaries[sname] = ssum

    v2_e = per_solver_errs["v2_phys"]
    wilcoxon_oe = {}
    for sname in SOLVER_ORDER:
        if sname == "v2_phys":
            continue
        b_oe = per_solver_errs[sname]["oe_strict"]
        wilcoxon_oe[f"v2_phys_vs_{sname}"] = {
            "v2_phys_wins_of_11": int(sum(v < b for v, b in zip(v2_e["oe_strict"], b_oe))),
            "strict_oe": safe_wilcoxon(v2_e["oe_strict"], b_oe),
        }

    wilcoxon_1d = {}
    for sname in COMPARISON_BASELINES:
        b_e = per_solver_errs[sname]
        wilcoxon_1d[f"v2_phys_vs_{sname}"] = {
            "T1_peak_err": safe_wilcoxon(v2_e["e_t1_pk"], b_e["e_t1_pk"]),
            "T1_lm_err": safe_wilcoxon(v2_e["e_t1_lm"], b_e["e_t1_lm"]),
            "T2_peak_err": safe_wilcoxon(v2_e["e_t2_pk"], b_e["e_t2_pk"]),
            "T2_lm_err": safe_wilcoxon(v2_e["e_t2_lm"], b_e["e_t2_lm"]),
        }

    return {
        "n_samples": len(samples),
        "all_full_echo_orig_vs_phys_max_spec_diff": round(max(s["orig_vs_phys_max_spec_diff"] for s in samples), 9),
        "reference_1d_and_2d_trends": ref_trends,
        "solvers": solver_summaries,
        "paired_wilcoxon_strict_odd_even": wilcoxon_oe,
        "paired_wilcoxon_1d_parity": wilcoxon_1d,
    }


def print_summary(payload):
    liver_summary = payload["core_liver_lard_n11"]["summary"]
    bench_lard_pea = payload["supplementary_lard_n1"]["benchmark_pea_complex"]
    bench_c57 = payload["separate_condition_c57_n1"]["benchmark_pea_complex"]

    print("\n=== CORE LIVER-LARD COHORT (n=11) SUMMARY ===")
    for sname in SOLVER_ORDER:
        ss = liver_summary["solvers"][sname]
        t1p = ss["abs_dlog10_T1_vs_1D_IR"]["T1_peak"]
        t1l = ss["abs_dlog10_T1_vs_1D_IR"]["T1_lm"]
        t2p = ss["abs_dlog10_T2_vs_1D_CPMG"]["T2_peak"]
        t2l = ss["abs_dlog10_T2_vs_1D_CPMG"]["T2_lm"]
        sp_wt = ss["spearman_vs_added_lard_wt_pct"]
        gcv_s = f" | gcv_hits={ss['gcv_lower_bound_hits']['full']}/11" if "gcv_lower_bound_hits" in ss else ""
        print(
            f"{sname:<20} | oe_strict={ss['odd_to_even_strict_nrmse']['mean']:.5f} "
            f"(al={ss['odd_to_even_aligned_nrmse']['mean']:.5f}, sh={ss['odd_to_even_shared_phase_aligned_nrmse']['mean']:.5f}) "
            f"| split_r={ss['echo_split_r']['mean']:.5f} "
            f"| dT1(pk/lm)={t1p['mean_dex']:.4f}/{t1l['mean_dex']:.4f} "
            f"| dT2(pk/lm)={t2p['mean_dex']:.4f}/{t2l['mean_dex']:.4f} "
            f"| rho_wt(T1pk/T1lm/T2lm)={sp_wt['T1_peak_ms']['rho']}/{sp_wt['T1_lm_ms']['rho']}/{sp_wt['T2_lm_ms']['rho']}{gcv_s}"
        )
    print("\nWilcoxon v2_phys vs baselines (strict OE):", json.dumps(liver_summary["paired_wilcoxon_strict_odd_even"]))
    print("Wilcoxon v2_phys vs baselines (1D parity):", json.dumps(liver_summary["paired_wilcoxon_1d_parity"]))
    print(
        "Lard v2_phys strict OE vs shared aligned:",
        bench_lard_pea["solvers"]["v2_phys"]["odd_to_even_strict_nrmse"],
        "vs",
        bench_lard_pea["solvers"]["v2_phys"]["odd_to_even_shared_phase_aligned_nrmse"],
        "| BRD(1e-3) strict OE:",
        bench_lard_pea["solvers"]["brd_nnls_fixed_1e-3"]["odd_to_even_strict_nrmse"],
        "| BRD(1e-2) strict OE:",
        bench_lard_pea["solvers"]["brd_nnls_fixed_1e-2"]["odd_to_even_strict_nrmse"],
    )
    print(
        "C57 v2_phys strict OE vs shared aligned:",
        bench_c57["solvers"]["v2_phys"]["odd_to_even_strict_nrmse"],
        "vs",
        bench_c57["solvers"]["v2_phys"]["odd_to_even_shared_phase_aligned_nrmse"],
        "| nech>2000:",
        bench_c57["nech_exceeds_pool_window"],
        "| orig_vs_phys_max_spec_diff:",
        bench_c57["orig_vs_phys_max_spec_diff"],
    )


def run_from_raw_archive(archive_root, out_json_path, layout=None):
    cfg = dict(DEFAULT_LAYOUT)
    if layout:
        cfg.update(layout)

    refine_net = load_frozen_v2()

    liver_samples = []
    for idx in range(1, 12):
        sdir = os.path.join(archive_root, cfg["liver_lard_2d_rel"].format(idx=idx))
        c_dir = os.path.join(archive_root, cfg["liver_lard_1d_cpmg_rel"].format(idx=idx))
        i_dir = os.path.join(archive_root, cfg["liver_lard_1d_ir_rel"].format(idx=idx))

        sd = load_pea_matrix(sdir)
        par = sd["par"]
        ti = sd["ti"]
        te_ms = par_float(par, ["TE(ms)", "TE"])
        note = par.get("Note", "")
        nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", note)
        assert len(nums) >= 2, f"Cannot parse masses from Note={note!r}"
        liver_g, lard_g = float(nums[0]), float(nums[1])
        wt_pct = lard_g / (liver_g + lard_g) * 100.0

        em = np.arange(1, sd["Yc"].shape[1] + 1, dtype=np.float64) * te_ms
        Yr_full, _ = phase_correct(sd["Yc"])
        mono_2d = fit_separable_monoexp(Yr_full, ti, em, init_t1=210.0, init_t2=55.0)
        bench = evaluate_matrix_benchmark(sd["Yc"], ti, te_ms, refine_net)

        tc, _, cc = read_col4(sorted(glob.glob(os.path.join(c_dir, "*.1.pea")))[0])
        ti1, _, ci1 = read_col4(sorted(glob.glob(os.path.join(i_dir, "*.1.irp")))[0])
        p_1d_cpmg = fit_1d_cpmg_monoexp(tc, cc)
        p_1d_ir = fit_1d_ir_monoexp(ti1, ci1)

        liver_samples.append({
            "sample_id": f"LiverLard_{idx:02d}",
            "shape": list(sd["Yc"].shape),
            "acquisition": {
                "B0_MHz": par_float(par, ["sf(MHz)"]),
                "TE_ms": te_ms,
                "TW_ms": par_float(par, ["TW(ms)"]),
                "TI_min_ms": round(float(ti[0]), 3),
                "TI_max_ms": round(float(ti[-1]), 3),
                "NS": int(par_float(par, ["NS"], 4)),
                "P1_us": par_float(par, ["P1(us)"]),
                "P2_us": par_float(par, ["P2(us)"]),
            },
            "composition": {
                "liver_mass_g": round(liver_g, 2),
                "added_lard_g": round(lard_g, 2),
                "added_lard_wt_pct": round(wt_pct, 2),
            },
            "paired_1d_ir": p_1d_ir,
            "paired_1d_cpmg": p_1d_cpmg,
            "monoexp_2d": mono_2d,
            "phase_diagnostics": bench["phase_diagnostics"],
            "nech_exceeds_pool_window": bench["nech_exceeds_pool_window"],
            "orig_vs_phys_max_pooled_diff": bench["orig_vs_phys_max_pooled_diff"],
            "orig_vs_phys_max_spec_diff": bench["orig_vs_phys_max_spec_diff"],
            "solvers": bench["solvers"],
        })

    liver_summary = summarize_liver_cohort(liver_samples)

    lard_2d_dir = os.path.join(archive_root, cfg["lard_2d_rel"])
    lard_1d_dir = os.path.join(archive_root, cfg["lard_1d_rel"])
    lard_xls = os.path.join(archive_root, cfg["lard_xls_rel"])

    lard_pea = load_pea_matrix(lard_2d_dir)
    lard_par = lard_pea["par"]
    te_lard = par_float(lard_par, ["TE(ms)", "TE"])
    ti_lard, _, Yr_lard_ircp = read_ircp_file(sorted(glob.glob(os.path.join(lard_2d_dir, "*.1.ircp")))[0])
    em_lard = np.arange(1, lard_pea["Yc"].shape[1] + 1, dtype=np.float64) * te_lard
    Yr_lard_pea, _ = phase_correct(lard_pea["Yc"])

    mono_lard = fit_separable_monoexp(Yr_lard_pea, ti_lard, em_lard, init_t1=150.0, init_t2=80.0)
    Yr_pea_n = Yr_lard_pea / (float(np.max(np.abs(Yr_lard_pea))) + 1e-12)
    Yr_ircp_n = Yr_lard_ircp / (float(np.max(np.abs(Yr_lard_ircp))) + 1e-12)
    n_sign_flipped = int(sum(
        (np.mean(Yr_pea_n[i, :5]) * np.mean(Yr_ircp_n[i, :5]) < 0.0)
        and (abs(np.mean(Yr_pea_n[i, :5])) > 0.05)
        for i in range(len(ti_lard))
    ))

    def bi2d_res(p):
        a1, lt1_1, lt2_1, a2, lt1_2, lt2_2, b = p
        m1 = a1 * np.outer(1.0 - b * np.exp(-ti_lard / (10.0 ** lt1_1)), np.exp(-em_lard / (10.0 ** lt2_1)))
        m2 = a2 * np.outer(1.0 - b * np.exp(-ti_lard / (10.0 ** lt1_2)), np.exp(-em_lard / (10.0 ** lt2_2)))
        return (m1 + m2 - Yr_pea_n).ravel()

    sol_bi2 = least_squares(
        bi2d_res,
        [0.55, np.log10(110), np.log10(60), 0.45, np.log10(220), np.log10(190), 2.0],
        bounds=([0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 1.5], [2.0, 3.5, 3.5, 2.0, 3.5, 3.5, 2.2]),
    )
    a1, lt1_1, lt2_1, a2, lt1_2, lt2_2, b_bi = [float(v) for v in sol_bi2.x]
    bi2d_lard = {
        "w1": round(a1 / (a1 + a2), 3),
        "T1_1_ms": round(10.0 ** lt1_1, 2),
        "T2_1_ms": round(10.0 ** lt2_1, 2),
        "w2": round(a2 / (a1 + a2), 3),
        "T1_2_ms": round(10.0 ** lt1_2, 2),
        "T2_2_ms": round(10.0 ** lt2_2, 2),
        "b": round(b_bi, 4),
        "NRMSE": round(float(np.sqrt(np.mean(bi2d_res(sol_bi2.x) ** 2))), 5),
    }

    t_1d_l, _, c_1d_l = read_col4(sorted(glob.glob(os.path.join(lard_1d_dir, "*.1.pea")))[0])
    y_1d_l = phase_correct_1d(c_1d_l, 0)
    y_1d_ln = y_1d_l / (float(np.max(np.abs(y_1d_l))) + 1e-12)
    sol_1d_l_mono = fit_1d_cpmg_monoexp(t_1d_l, c_1d_l)
    sol_1d_l_bi = least_squares(
        lambda p: p[0] * np.exp(-t_1d_l / p[1]) + p[2] * np.exp(-t_1d_l / p[3]) - y_1d_ln,
        [0.5, 40.0, 0.5, 150.0],
        bounds=([0.0, 1.0, 0.0, 10.0], [5.0, 500.0, 5.0, 5000.0]),
    )
    bw1, bt1, bw2, bt2 = [float(v) for v in sol_1d_l_bi.x]
    cpmg_1d_lard = {
        "mono_T2_ms": sol_1d_l_mono["T2_1D_ms"],
        "bi_w1": round(bw1 / (bw1 + bw2), 3),
        "bi_T2_1_ms": round(bt1, 2),
        "bi_w2": round(bw2 / (bw1 + bw2), 3),
        "bi_T2_2_ms": round(bt2, 2),
    }

    bench_lard_pea = evaluate_matrix_benchmark(lard_pea["Yc"], ti_lard, te_lard, refine_net)
    bench_lard_ircp = evaluate_matrix_benchmark(None, ti_lard, te_lard, refine_net, Yr_override=Yr_lard_ircp)

    lard_out = {
        "sample_id": "Lard_01",
        "shape": list(lard_pea["Yc"].shape),
        "acquisition": {
            "B0_MHz": par_float(lard_par, ["sf(MHz)"]),
            "TE_ms": te_lard,
            "TW_ms": par_float(lard_par, ["TW(ms)"]),
            "TI_min_ms": round(float(ti_lard[0]), 3),
            "TI_max_ms": round(float(ti_lard[-1]), 3),
            "NS": int(par_float(lard_par, ["NS"], 8)),
            "P1_us": par_float(lard_par, ["P1(us)"]),
            "P2_us": par_float(lard_par, ["P2(us)"]),
        },
        "pea_vs_ircp_audit": {
            "corr": round(float(np.corrcoef(Yr_pea_n.ravel(), Yr_ircp_n.ravel())[0, 1]), 6),
            "max_abs_norm_diff": round(float(np.max(np.abs(Yr_pea_n - Yr_ircp_n))), 6),
            "n_whole_row_sign_flips": n_sign_flipped,
        },
        "monoexp_2d": mono_lard,
        "biexp_2d": bi2d_lard,
        "paired_1d_cpmg": cpmg_1d_lard,
        "vendor_2d_output_xls": vendor_sheet3_peaks(lard_xls),
        "benchmark_pea_complex": bench_lard_pea,
        "benchmark_ircp_vendor": bench_lard_ircp,
    }

    c57_dir = os.path.join(archive_root, cfg["c57_2d_rel"])
    c57_prefix = cfg.get("c57_prefix", "C57H98012-S")
    c57_ircp = os.path.join(c57_dir, f"{c57_prefix}.1.ircp")
    if not os.path.isfile(c57_ircp):
        cand_ircps = sorted(glob.glob(os.path.join(c57_dir, "*.1.ircp")))
        assert cand_ircps, f"No *.1.ircp found in {c57_dir}"
        c57_ircp = cand_ircps[0]
        c57_prefix = os.path.basename(c57_ircp)[:-len(".1.ircp")]

    c57_pea = load_pea_matrix(c57_dir, prefix=c57_prefix)
    c57_par = c57_pea["par"]
    te_c57 = par_float(c57_par, ["TE(ms)", "TE"])
    ti_c57, _, _ = read_ircp_file(c57_ircp)
    em_c57 = np.arange(1, c57_pea["Yc"].shape[1] + 1, dtype=np.float64) * te_c57
    Yr_c57, _ = phase_correct(c57_pea["Yc"])
    mono_c57 = fit_separable_monoexp(Yr_c57, ti_c57, em_c57, init_t1=60.0, init_t2=15.0)
    bench_c57 = evaluate_matrix_benchmark(c57_pea["Yc"], ti_c57, te_c57, refine_net)

    c57_out = {
        "sample_id": "Std_C57H98O12_2022",
        "shape": list(c57_pea["Yc"].shape),
        "acquisition": {
            "B0_MHz": par_float(c57_par, ["sf(MHz)"]),
            "TE_ms": te_c57,
            "TW_ms": par_float(c57_par, ["TW(ms)"]),
            "TI_min_ms": round(float(ti_c57[0]), 3),
            "TI_max_ms": round(float(ti_c57[-1]), 3),
            "NS": int(par_float(c57_par, ["NS"], 8)),
            "P1_us": par_float(c57_par, ["P1(us)"]),
            "P2_us": par_float(c57_par, ["P2(us)"]),
        },
        "monoexp_2d": mono_c57,
        "benchmark_pea_complex": bench_c57,
    }

    payload = {
        "protocol": {
            "description": "Strict zero-even-leakage Odd->Even retrospective archive evaluation using frozen Paper 2 weights (v2_refine.pth, 19,393 params).",
            "odd_even_phasing": "Split complex Yc before phasing; ph_o from Yc_o[-1], ph_e from Yc_e[-1]; Yo_s = Re(Yc_o * exp(-i*ph_o)), Ye_on_o_phase = Re(Yc_e * exp(-i*ph_o)), Ye_s = Re(Yc_e * exp(-i*ph_e)).",
            "strict_nrmse_formula": "Normalize Yo_s and Ye_on_o_phase by scale_o = max|Yo_s|; fit alpha_o >= 0 solely on Yo_s; evaluate RMSE(alpha_o * K1 @ M_odd @ K2_even^T * DL1*DL2 - Ye_n).",
            "grid": {
                "N_T1": int(N_T1),
                "N_T2": int(N_T2),
                "T1_range_ms": [round(float(T1_GRID[0]), 2), round(float(T1_GRID[-1]), 2)],
                "T2_range_ms": [round(float(T2_GRID[0]), 2), round(float(T2_GRID[-1]), 2)],
                "DL1_dex": round(float(DL1), 6),
                "DL2_dex": round(float(DL2), 6),
            },
            "solvers": SOLVER_ORDER,
            "reprobundle_baseline_commit": "9d914b0",
        },
        "core_liver_lard_n11": {
            "summary": liver_summary,
            "samples": liver_samples,
        },
        "supplementary_lard_n1": lard_out,
        "separate_condition_c57_n1": c57_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved sanitized strict Odd->Even JSON: {out_json_path}")
    print_summary(payload)
    return 0


def verify_from_archive(archive_json_path):
    assert os.path.isfile(archive_json_path), f"Missing archive JSON: {archive_json_path}"
    with open(archive_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for req_key in ("protocol", "core_liver_lard_n11", "supplementary_lard_n1", "separate_condition_c57_n1"):
        assert req_key in payload, f"Missing top-level key in archive JSON: {req_key}"
    samples = payload["core_liver_lard_n11"]["samples"]
    assert len(samples) == 11, f"Expected 11 liver-lard samples, got {len(samples)}"
    recomputed = summarize_liver_cohort(samples)
    assert recomputed == payload["core_liver_lard_n11"]["summary"], "Recomputed summary mismatch vs archived summary"
    print(f"Verified offline archive JSON (n={len(samples)} core + Lard_01 + Std_C57H98O12_2022): {archive_json_path}")
    print_summary(payload)
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Strict Odd->Even retrospective 23 MHz workstation-archive evaluation / offline verification."
    )
    parser.add_argument(
        "--archive-root",
        default=os.environ.get("NMR2_HISTORICAL_ARCHIVE_DIR", ""),
        help="Optional root directory of restricted raw 23 MHz workstation archives.",
    )
    parser.add_argument(
        "--layout-json",
        default="",
        help="Optional JSON file overriding relative subdirectory patterns under --archive-root.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_ARCHIVE_JSON,
        help="Output JSON path when running with --archive-root.",
    )
    parser.add_argument(
        "--archive-json",
        default=DEFAULT_ARCHIVE_JSON,
        help="Sanitized archive JSON path used in default offline verification mode.",
    )
    args = parser.parse_args(argv)

    if args.archive_root:
        assert os.path.isdir(args.archive_root), f"Archive root not found: {args.archive_root}"
        layout = None
        if args.layout_json:
            with open(args.layout_json, "r", encoding="utf-8") as f:
                layout = json.load(f)
        return run_from_raw_archive(args.archive_root, args.out, layout=layout)

    return verify_from_archive(args.archive_json)


if __name__ == "__main__":
    raise SystemExit(main())
