# -*- coding: utf-8 -*-
"""
Monte Carlo literature composition label-noise sensitivity audit (n = 9 oils, N = 4,000 draws).

Tests Spearman correlation survival (same sign & asymptotic p < 0.05) across Gaussian
perturbations sigma in {1.0, 3.0, 5.0} percentage points added to (SFA, MUFA, PUFA) triplets
clipped >= 0.5% and renormalized to 100.0%, for both classical SVD-ridge (`e4_fingerprint.json`,
archived in `label_sensitivity.json`) and proposed `v2` (`stats_and_1d2d.json`, archived in `label_sensitivity_v2_stats.json`).
"""
import argparse
import json
import os
import re
import sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(HERE, "results_archive")
OILS = ["亚麻籽油", "大豆油", "菜籽油", "葵花籽油", "玉米油", "橄榄油", "芝麻油", "花生油", "红花籽油"]


def run_sensitivity(feature_map, pairs, n_draws=4000, seed=20261115):
    stats_path = os.path.join(ARCHIVE_DIR, "stats_and_1d2d.json")
    st = json.load(open(stats_path, encoding="utf-8"))
    raw = json.dumps(st, ensure_ascii=False)
    comp = {}
    pat = r'"(亚麻籽油|大豆油|菜籽油|葵花籽油|玉米油|橄榄油|芝麻油|花生油|红花籽油)"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    for m in re.finditer(pat, raw):
        comp[m.group(1)] = [float(m.group(2)), float(m.group(3)), float(m.group(4))]
    assert len(comp) == 9, f"Expected 9 oil compositions, got {len(comp)}"

    labels = {o: {"SFA": comp[o][0], "MUFA": comp[o][1], "PUFA": comp[o][2]} for o in OILS}
    feat_keys = sorted({f for f, _ in pairs})
    fvals = {f: np.array([feature_map[o][f] for o in OILS], dtype=float) for f in feat_keys}
    lab0 = {c: np.array([labels[o][c] for o in OILS], dtype=float) for c in ("PUFA", "SFA")}

    base = {}
    for f, c in pairs:
        res = spearmanr(fvals[f], lab0[c])
        base[f"{f}_vs_{c}"] = (float(res.statistic), float(res.pvalue))

    rng = np.random.default_rng(seed)
    out = {"base": {k: {"rho": r, "p": p} for k, (r, p) in base.items()}, "levels": {}}
    for sigma in (1.0, 3.0, 5.0):
        surv = {f"{f}_vs_{c}": [] for f, c in pairs}
        rho_gt05 = {f"{f}_vs_{c}": 0 for f, c in pairs}
        for _ in range(n_draws):
            tri = {}
            for o in OILS:
                t = np.array([labels[o][c] + rng.normal(0.0, sigma) for c in ("SFA", "MUFA", "PUFA")])
                t = np.clip(t, 0.5, None)
                tri[o] = t / t.sum() * 100.0
            lab = {
                "SFA": np.array([tri[o][0] for o in OILS]),
                "PUFA": np.array([tri[o][2] for o in OILS]),
            }
            for f, c in pairs:
                res = spearmanr(fvals[f], lab[c])
                key = f"{f}_vs_{c}"
                same_sign = float(res.statistic) * base[key][0] > 0
                surv[key].append(1 if (same_sign and float(res.pvalue) < 0.05) else 0)
                if abs(float(res.statistic)) > 0.5:
                    rho_gt05[key] += 1
        out["levels"][f"sigma_{sigma}pp"] = {
            f"{f}_vs_{c}": {
                "survival_p05": round(float(np.mean(surv[f"{f}_vs_{c}"])), 4),
                "rho_gt05": round(rho_gt05[f"{f}_vs_{c}"] / n_draws, 4),
            }
            for f, c in pairs
        }
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write label_sensitivity.json and label_sensitivity_v2_stats.json to working directory")
    args = parser.parse_args()

    e4 = json.load(open(os.path.join(ARCHIVE_DIR, "e4_fingerprint.json"), encoding="utf-8"))
    svd_pairs = [(f, c) for f in ("t2_fwhm", "t1_peak", "t1_fwhm", "t2_peak", "ratio") for c in ("PUFA", "SFA")]
    svd_out = run_sensitivity(e4["features"], svd_pairs)

    st = json.load(open(os.path.join(ARCHIVE_DIR, "stats_and_1d2d.json"), encoding="utf-8"))
    v2_feats = {o: st["sample_features_13"][o]["v2"] for o in OILS}
    v2_pairs = [("t1_fwhm", "PUFA"), ("mean_logT1", "PUFA"), ("t1_peak", "PUFA"), ("t1_peak", "SFA"), ("t2_fwhm", "PUFA")]
    v2_out = run_sensitivity(v2_feats, v2_pairs)

    print("== SVD-ridge Label-Noise Sensitivity (matches archived label_sensitivity.json) ==", flush=True)
    for lvl, blk in svd_out["levels"].items():
        print(f"  {lvl}: t2_fwhm_PUFA={blk['t2_fwhm_vs_PUFA']['survival_p05']:.4f}, t1_pk_PUFA={blk['t1_peak_vs_PUFA']['survival_p05']:.4f}, t1_fwhm_PUFA={blk['t1_fwhm_vs_PUFA']['survival_p05']:.4f}, t1_pk_SFA={blk['t1_peak_vs_SFA']['survival_p05']:.4f}", flush=True)

    print("== Proposed v2 Label-Noise Sensitivity (stats_and_1d2d.json / label_sensitivity_v2_stats.json) ==", flush=True)
    for lvl, blk in v2_out["levels"].items():
        print(f"  {lvl}: t1_fwhm_PUFA={blk['t1_fwhm_vs_PUFA']['survival_p05']:.4f}, mean_logT1_PUFA={blk['mean_logT1_vs_PUFA']['survival_p05']:.4f}, t1_pk_PUFA={blk['t1_peak_vs_PUFA']['survival_p05']:.4f}, t1_pk_SFA={blk['t1_peak_vs_SFA']['survival_p05']:.4f}", flush=True)

    if args.write:
        with open(os.path.join(HERE, "label_sensitivity.json"), "w", encoding="utf-8") as f:
            json.dump(svd_out, f, ensure_ascii=False, indent=2)
        with open(os.path.join(HERE, "label_sensitivity_v2_stats.json"), "w", encoding="utf-8") as f:
            json.dump(v2_out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
