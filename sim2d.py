# -*- coding: utf-8 -*-
"""
sim2d.py —— T1-T2 二维仿真生成器（第二篇原型 Step 1）
网格与真实采集一致：TI 20 点（D1S 对数 0.058–1000 ms）× 2000 回波（TE 1.027 ms）。
核：K2D((TI,TE),(T1,T2)) = [1−2exp(−TI/T1)] ⊗ exp(−TE/T2)   （Kronecker 可分离）
谱采样：物理合理的 2D 谱——少数 (log T1, log T2) 高斯斑，T1/T2 比值约束在油的实测范围。
"""
import numpy as np

# ---- 真实采集网格（与 .par/D1S 逐位一致）----
TI_MS = np.array([0.058, 0.097, 0.162, 0.271, 0.452, 0.756, 1.263, 2.11, 3.526,
                  5.891, 9.845, 16.451, 27.489, 45.934, 76.756, 128.26, 214.323,
                  358.134, 598.443, 1000.001])
N_TI = len(TI_MS)
NECH = 2000
TE_MS = 1.027
ECHO_MS = np.arange(1, NECH + 1) * TE_MS

# ---- 谱网格 ----
N_T1 = 48
N_T2 = 96
T1_GRID = np.logspace(np.log10(8.0), np.log10(2000.0), N_T1)     # 8–2000 ms
T2_GRID = np.logspace(np.log10(0.5), np.log10(3000.0), N_T2)     # 0.5–3000 ms
LOG_T1 = np.log10(T1_GRID)
LOG_T2 = np.log10(T2_GRID)
DL1 = LOG_T1[1] - LOG_T1[0]
DL2 = LOG_T2[1] - LOG_T2[0]

# ---- 可分离核 ----
K_T1 = 1.0 - 2.0 * np.exp(-TI_MS[:, None] / T1_GRID[None, :])        # (20, nT1)
K_T2 = np.exp(-ECHO_MS[:, None] / T2_GRID[None, :])                   # (2000, nT2)
# 注意：短 TI 的 K_T1 为负（反转态）✓；K_T2 首行≈exp(−TE/T2)~1


def kernel_t1():
    return K_T1.copy()


def kernel_t2():
    return K_T2.copy()


def sample_spectrum(rng, n_comp=None, ratio_range=(1.5, 3.5)):
    """采样一个 2D 谱 (nT1, nT2)：少数高斯斑，T1≈ratio·T2。
    物理范围按 13 样本实测（T1 43–628 ms）与油类 T2（~5–300 ms）约束，
    确保样本落在数据可辨识域内（TI_max=1000 ms）。"""
    n_comp = n_comp or int(rng.integers(1, 4))
    m = np.zeros((N_T1, N_T2))
    # 主斑：T2 峰位置 log-uniform（5–300 ms），T1 = ratio·T2
    t2_peak = 10.0 ** rng.uniform(np.log10(5.0), np.log10(300.0))
    ratio = rng.uniform(*ratio_range)
    t1_peak = ratio * t2_peak
    _add_blob(m, rng, t1_peak, t2_peak, amp=rng.uniform(0.6, 1.0),
              w1=rng.uniform(0.08, 0.25), w2=rng.uniform(0.1, 0.35))
    for _ in range(n_comp - 1):
        t2_k = 10.0 ** rng.uniform(np.log10(5.0), np.log10(300.0))
        ratio_k = rng.uniform(*ratio_range) * rng.uniform(0.85, 1.25)
        _add_blob(m, rng, ratio_k * t2_k, t2_k, amp=rng.uniform(0.15, 0.6),
                  w1=rng.uniform(0.1, 0.3), w2=rng.uniform(0.12, 0.4))
    m = np.clip(m, 0, None)
    area = m.sum() * DL1 * DL2
    if area > 0:
        m = m / area  # 单位 log 面积
    return m


def _add_blob(m, rng, t1_c, t2_c, amp, w1, w2):
    lt1, lt2 = np.log10(t1_c), np.log10(t2_c)
    g1 = np.exp(-0.5 * ((LOG_T1 - lt1) / w1) ** 2)
    g2 = np.exp(-0.5 * ((LOG_T2 - lt2) / w2) ** 2)
    m += amp * np.outer(g1, g2)


def make_sample(rng, snr=None, n_trunc_ti=None):
    """返回 dict：y(20,2000) 复数带噪、clean(20,2000)、spec 真值、mask(20,2000)。"""
    spec = sample_spectrum(rng)
    clean = K_T1 @ spec @ K_T2.T           # (20, 2000)
    y = clean.copy()
    mask = np.ones((N_TI, NECH))
    if snr is None:
        snr = rng.uniform(30, 300)
    sigma = np.abs(clean).max() / snr
    noise = (rng.normal(0, sigma, y.shape) + 1j * rng.normal(0, sigma, y.shape)) / np.sqrt(2)
    y = y + noise
    if n_trunc_ti:
        # 截断模拟：把最后 n_trunc_ti 个 TI 的回波串截短（掩码置零）
        keep = NECH - int(rng.integers(200, 1200))
        y[N_TI - n_trunc_ti:, keep:] = 0
        mask[N_TI - n_trunc_ti:, keep:] = 0
    return {"y": y.astype(np.complex128), "clean": clean, "spec": spec,
            "mask": mask, "snr": snr}


if __name__ == "__main__":
    rng = np.random.default_rng(20261112)
    s = make_sample(rng, snr=100)
    print("clean 形状:", s["clean"].shape, "| spec:", s["spec"].shape,
          "| 面积:", s["spec"].sum() * DL1 * DL2)
    print(f"clean 幅度: 首 TI 首 echo={s['clean'][0,0]:.4f}, 首 TI 末 echo={s['clean'][0,-1]:.4f}")
    print("K_T1[0]（TI=0.058ms）范围:", K_T1[0].min().round(3), "→", K_T1[0].max().round(3))
    print("K_T1[-1]（TI=1000ms）范围:", K_T1[-1].min().round(3), "→", K_T1[-1].max().round(3))
    # 物理自检：短 TI 应为负（反转态）
    assert K_T1[0].max() < 0, "短 TI 未反转——核符号错误!"
    # 非负性自检：T2 方向无负值（极短 T2 长回波处下溢为 0 属正常）
    assert K_T2.min() >= 0
    print("SMOKE OK：反转核符号正确、T2 核全正、2D 面形状正确")
