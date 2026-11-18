# -*- coding: utf-8 -*-
"""
E3: TI 子采样敏感性——同一批 sim 样本，20 TI vs 10 TI（偶/奇两种抽法）的基线重建对比
"""
import io, sys, os, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import make_sample, TI_MS, N_T1, N_T2, DL1, DL2

_T1 = np.logspace(np.log10(8.0), np.log10(2000.0), N_T1)
_T2 = np.logspace(np.log10(0.5), np.log10(3000.0), N_T2)
LOG_T1g = np.log10(_T1)
LOG_T2g = np.log10(_T2)


def recon_with_ti(ti_grid, y, lam=1e-3):
    K1 = 1.0 - 2.0 * np.exp(-ti_grid[:, None] / _T1[None, :])
    K2 = np.exp((np.arange(1, y.shape[1] + 1) * 1.027)[:, None] / _T2[None, :] * -1.0)
    U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
    U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
    r1 = int((S1 > S1[0] * 1e-10).sum()); r2 = int((S2 > S2[0] * 1e-10).sum())
    Yc = U1[:, :r1].T @ np.real(y) @ U2[:, :r2]
    s = S1[:r1][:, None] * S2[:r2][None, :]
    C = s * Yc / (s ** 2 + (1e-3 * S1[0] * S2[0]) ** 2)
    M = np.clip(V1t[:r1].T @ C @ V2t[:r2], 0, None)
    a = M.sum() * DL1 * DL2
    return M / a if a > 1e-12 else M


def main():
    rng = np.random.default_rng(20261115)
    res = {"full20": [], "even10": [], "odd10": []}
    for _ in range(60):
        s = make_sample(rng, snr=float(rng.uniform(30, 300)))
        t = s["spec"] / (s["spec"].sum() * DL1 * DL2 + 1e-12)
        for key, idx in [("full20", np.arange(20)), ("even10", np.arange(0, 20, 2)),
                         ("odd10", np.arange(1, 20, 2))]:
            M = recon_with_ti(TI_MS[idx], s["y"][idx], lam=1e-3)
            p = M / (M.sum() * DL1 * DL2 + 1e-12)
            res[key].append(float(np.corrcoef(p.ravel(), t.ravel())[0, 1]))
    out = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in res.items()}
    print(json.dumps(out, indent=1), flush=True)
    with open(os.path.join(HERE, "e3_subsampling.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
