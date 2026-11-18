# -*- coding: utf-8 -*-
"""
baseline2d.py —— 2D T1-T2 物理基线（第二篇原型 Step 2）
经典快速 2D ILT（Venkataramanan 式）：K1/K2 各做 SVD，问题缩到小系数核，
NNLS+弱 Tikhonov 求解，再映射回谱域。合成样本自检：重建 vs 真值。
"""
import io, sys, os, time
import numpy as np
from scipy.optimize import nnls

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import (TI_MS, N_TI, NECH, N_T1, N_T2, K_T1, K_T2, make_sample,
                   DL1, DL2, LOG_T1, LOG_T2)

# ---------- SVD 截断 ----------
U1, S1, V1t = np.linalg.svd(K_T1, full_matrices=False)   # (20,k1),(k1,),(k1,48)
U2, S2, V2t = np.linalg.svd(K_T2, full_matrices=False)   # (2000,k2),(k2,),(k2,96)
r1 = int((S1 > S1.max() * 1e-10).sum())
r2 = int((S2 > S2.max() * 1e-10).sum())
print(f"K1 SVD 秩 {r1}/{min(N_TI, N_T1)}; K2 SVD 秩 {r2}/{min(NECH, N_T2)}")

U1c, S1c, V1c = U1[:, :r1], S1[:r1], V1t[:r1]            # V1c: (r1, nT1)
U2c, S2c, V2c = U2[:, :r2], S2[:r2], V2t[:r2]            # V2c: (r2, nT2)

# 小核：ŷ = S1c C S2cᵀ → vec(ŷ) = (S2c ⊗ S1c) vec(C)，(r1·r2) × (nT1·nT2)
K_small = (S2c[:, None] * S1c[None, :]).reshape(-1)      # (r1*r2,) 对角化形式
n_small = r1 * r2
print(f"小核方程数 {r1*r2}, 未知数 {N_T1*N_T2}")


def reconstruct_2d(y, mask=None, lam=1e-3):
    """y (20,2000) 复数 → 谱 M (nT1,nT2)。mask=None 视为全有效。
    核全实 → 虚部为纯噪声，取实部求解。
    SVD 域对角化：Ŷ_ij = S1_i·C_ij·S2_j → 闭式 ridge（逐系数维纳式）
    c_ij = s_ij·Ŷ_ij/(s_ij²+λ²·s_max²)；非负性施加在谱 M 上（ReLU+归一）。
    注意：不能对 C 施加非负（V 基混合符号，会压进错误锥）。"""
    Y = np.real(y)
    if mask is not None and (mask < 1).any():
        Y = Y * mask
    Ycore = U1c.T @ Y @ U2c                             # (r1, r2)
    s_prod = S1c[:, None] * S2c[None, :]                # (r1, r2) 对角增益
    smax = S1c[0] * S2c[0]
    denom = s_prod ** 2 + (lam * smax) ** 2
    C = np.where(denom > 0, s_prod * Ycore / np.where(denom > 0, denom, 1.0), 0.0)
    M = V1c.T @ C @ V2c                                 # (nT1, nT2)
    M = np.clip(M, 0, None)
    area = M.sum() * DL1 * DL2
    if area > 1e-12:
        M = M / area
    return M


def metrics(M, spec_true):
    p = M / (M.sum() * DL1 * DL2 + 1e-12)
    r = spec_true / (spec_true.sum() * DL1 * DL2 + 1e-12)
    pear = float(np.corrcoef(p.ravel(), r.ravel())[0, 1])
    # 峰位（log 域）
    ip = np.unravel_index(np.argmax(p), p.shape)
    it = np.unravel_index(np.argmax(r), r.shape)
    e_t1 = abs(LOG_T1[ip[0]] - LOG_T1[it[0]])
    e_t2 = abs(LOG_T2[ip[1]] - LOG_T2[it[1]])
    l2 = float(np.linalg.norm(p - r))
    return {"pearson": pear, "peak_err_logT1": e_t1, "peak_err_logT2": e_t2, "spec_l2": l2}


if __name__ == "__main__":
    rng = np.random.default_rng(20261112)
    print("== 可辨识性分析（论文图素材） ==")
    print("  K1 奇异值:", np.round(S1 / S1[0], 4))
    print("  K2 奇异值(前40):", np.round((S2 / S2[0])[:40], 4))
    print("== 合成样本重建自检 ==")
    for snr in (100, 30):
        for ncomp in (1, 3):
            s = make_sample(rng, snr=snr)
            M = reconstruct_2d(s["y"], s["mask"], lam=1e-3)
            m = metrics(M, s["spec"])
            print(f"  SNR={snr:>3} ncomp={ncomp}: pearson={m['pearson']:.4f} "
                  f"峰位误差 logT1={m['peak_err_logT1']:.3f} logT2={m['peak_err_logT2']:.3f} "
                  f"L2={m['spec_l2']:.4f}")
    # λ 敏感性
    s = make_sample(rng, snr=100)
    print("== λ 敏感性（SNR=100） ==")
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        M = reconstruct_2d(s["y"], s["mask"], lam=lam)
        m = metrics(M, s["spec"])
        print(f"  λ={lam:g}: pearson={m['pearson']:.4f} L2={m['spec_l2']:.4f}")
