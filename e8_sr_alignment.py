# -*- coding: utf-8 -*-
"""E8-SR 对齐实验: 冻结 v2 (RefineNet 19,393 参数) 在 2MHz 砂岩 SR 数据上的零样本迁移
两个自洽变体（Gemini 任务书把"SR 核切换"与"信号符号对齐"混在一起，二选一才自洽）:
  变体A 信号对齐: g = 2*ys - ys[0:1,:] (行0=最长tau≈完全恢复作参考), 插值到训练 TI_MS 网格,
         用【原装 IR 前端】+ 冻结 RefineNet —— 网络输入保持在训练分布内
  变体B 核切换:  ys 原样输入, 前端 K1 换成 SR 核 —— 网络见到 SR 型前端输出
对照: SR-SVD 基线(训练网格 48×96, 线性域); r = corrcoef(v2输出, SR基线) 线性域。
只读数据; 输出 e8_sr_alignment.json / .png 于本目录。
"""
import io, sys, os, csv, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train2d_v2 import RefineNet, PhysicsFrontEnd, log_pool_np, _T1, _T2, DL1, DL2
from sim2d import TI_MS

BASE = os.path.join(os.environ.get("NMR2_ROCK_DATA_DIR", ""), "砂岩") if os.environ.get("NMR2_ROCK_DATA_DIR", "") else ""
T1G, T2G = np.asarray(_T1), np.asarray(_T2)
TI_MS = np.asarray(TI_MS, dtype=float)
DEVICE = torch.device("cpu")

def load_numeric(p):
    good = []
    for r in csv.reader(open(p, encoding="utf-8-sig")):
        if len(r) < 2:
            continue
        try:
            good.append([float(x) for x in r])
        except ValueError:
            pass
    return np.array(good)

def load_core(d):
    a = load_numeric(os.path.join(d, "data2.csv"))
    Y = a[:, 0::2] + 1j * a[:, 1::2]
    ir, ic = np.unravel_index(np.argmax(np.abs(Y)), Y.shape)
    Yr = np.real(Y * np.exp(-1j * np.angle(Y[ir, ic])))
    ys = Yr / (np.abs(Yr).max() + 1e-12)
    taus = np.logspace(np.log10(500.0), np.log10(0.1), ys.shape[0])   # 行0=最长tau
    te = 0.06
    return ys, taus, te

def sr_baseline(ys, taus, te, lam=1e-4):
    """SR 基线（训练网格, 线性域归一化）—— λ 在 1e-3/1e-4/1e-5 中取残差最小"""
    best = None
    for lam_try in (1e-3, 1e-4, 1e-5):
        K1 = 1 - np.exp(-taus[:, None] / T1G[None, :])
        K2 = np.exp(-(np.arange(1, ys.shape[1] + 1) * te)[:, None] / T2G[None, :])
        U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
        U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
        r1 = int((S1 > S1[0] * 1e-10).sum()); r2 = int((S2 > S2[0] * 1e-10).sum())
        Yc = U1[:, :r1].T @ ys @ U2[:, :r2]
        s = S1[:r1][:, None] * S2[:r2][None, :]
        smax = S1[0] * S2[0]
        C = s * Yc / (s ** 2 + (lam_try * smax) ** 2)
        Yfit = U1[:, :r1] @ (s * C) @ U2[:, :r2].T
        rel = float(np.linalg.norm(ys - Yfit) / (np.linalg.norm(ys) + 1e-12))
        if best is None or rel < best[1]:
            M = np.clip(V1t[:r1].T @ C @ V2t[:r2], 0, None)
            area = M.sum() * DL1 * DL2
            best = (M / (area + 1e-12), rel, lam_try)
    return best

def to_train_grid(g, taus, n_echo, te):
    """(变体A) 把 SR→IR 等效信号插值到训练 TI_MS 网格; 回波轴插值到训练 2000 回波时间轴"""
    order = np.argsort(taus)                       # tau 升序
    tt, gg = taus[order], g[order]
    gi = np.array([np.interp(np.log10(taus), np.log10(tt), gg[:, k]) for k in range(gg.shape[1])]).T
    gi = np.clip(gi, -1.0, 1.0)
    # 回波轴: 训练回波时间 (1..2000)*1.027ms; 数据窗口外置 0（采集截断）
    t_train = np.arange(1, 2001) * 1.027
    t_data = np.arange(1, n_echo + 1) * te
    out = np.zeros((len(TI_MS), 2000), dtype=np.float32)
    cov = t_train <= t_data[-1]
    for i in range(len(TI_MS)):
        out[i, cov] = np.interp(t_train[cov], t_data, gi[i])
    mx = np.abs(out).max()
    return out / (mx + 1e-12)


def main():
    if not BASE or not os.path.isdir(BASE):
        raise FileNotFoundError(
            "External 2.0 MHz sandstone dataset is not bundled under institutional policy; "
            "set NMR2_ROCK_DATA_DIR or inspect results_archive/e8_sr_alignment.json."
        )
    CK = torch.load(os.path.join(HERE, "v2_refine.pth"), map_location=DEVICE, weights_only=True)
    REFINE = RefineNet(); REFINE.load_state_dict(CK["refine"]); REFINE.eval()
    FRONT = PhysicsFrontEnd(); FRONT.eval()
    results = {"variant_A_signal_align": {}, "variant_B_kernel_swap": {}, "meta": {}}
    maps = {}
    sand_dir = os.path.join(BASE, "sandstone") if os.path.isdir(os.path.join(BASE, "sandstone")) else BASE
    core_dirs = sorted(
        d for d in os.listdir(sand_dir)
        if os.path.isfile(os.path.join(sand_dir, d, "T1T2b", "data2.csv"))
    )
    for idx, raw_core in enumerate(core_dirs, start=1):
        core = f"sandstone_{idx:02d}"
        ys, taus, te = load_core(os.path.join(sand_dir, raw_core, "T1T2b"))
        Mb, relb, lam_used = sr_baseline(ys, taus, te)
        # 变体A: 信号对齐 → 原装 IR 前端
        g = 2.0 * ys - ys[0:1, :]
        g_train = to_train_grid(g, taus, ys.shape[1], te)
        with torch.no_grad():
            base_ir = FRONT(torch.tensor(g_train[None], dtype=torch.float32))[0]
            pooled = torch.tensor(log_pool_np(g_train[None]), dtype=torch.float32)
            pred_log = REFINE(base_ir[None], pooled)[0].numpy()
        Ml_A = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
        rA = float(np.corrcoef(Ml_A.ravel(), Mb.ravel())[0, 1])
        # 变体B: SR 核切换 + 原始 ys
        K1sr = 1 - np.exp(-taus[:, None] / T1G[None, :])
        K2t = torch.tensor(np.exp(-(np.arange(1, ys.shape[1] + 1) * te)[:, None] / T2G[None, :]), dtype=torch.float32)
        U1, S1, V1t = torch.linalg.svd(torch.tensor(K1sr, dtype=torch.float32), full_matrices=False)
        U2, S2, V2t = torch.linalg.svd(K2t, full_matrices=False)
        r1 = int((S1 > S1[0] * 1e-10).sum()); r2 = int((S2 > S2[0] * 1e-10).sum())
        ytt = torch.tensor(ys[None], dtype=torch.float32)
        with torch.no_grad():
            Yc = torch.einsum("ti,btj->bij", U1[:, :r1], ytt)
            Yc = torch.einsum("bij,jk->bik", Yc, U2[:, :r2])
            ss = S1[:r1][:, None] * S2[:r2][None, :]
            sm = S1[0] * S2[0]
            C = ss * Yc / (ss ** 2 + (1e-4 * sm) ** 2).clamp(min=1e-12)
            Msr = torch.relu(V1t[:r1].t() @ C @ V2t[:r2])
            area = (Msr.sum() * DL1 * DL2).clamp(min=1e-9)
            base_sr = torch.log1p(Msr / area * 20.0)
            pooled = torch.tensor(log_pool_np(ys[None]), dtype=torch.float32)
            pred_log = REFINE(base_sr, pooled)[0].numpy()
        Ml_B = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
        rB = float(np.corrcoef(Ml_B.ravel(), Mb.ravel())[0, 1])
        lt1, lt2 = np.log10(T1G), np.log10(T2G)
        pk = lambda M: (10 ** lt1[int(np.argmax(M.sum(1)))], 10 ** lt2[int(np.argmax(M.sum(0)))])
        results["variant_A_signal_align"][core] = {"r_vs_SRbaseline": round(rA, 4),
                                                   "SRbaseline_resid": round(relb, 4), "lam": lam_used,
                                                   "peaks_v2": [round(x, 3) for x in pk(Ml_A)],
                                                   "peaks_SRbase": [round(x, 3) for x in pk(Mb)]}
        results["variant_B_kernel_swap"][core] = {"r_vs_SRbaseline": round(rB, 4),
                                                  "peaks_v2": [round(x, 3) for x in pk(Ml_B)]}
        maps[core] = (Mb, Ml_A, Ml_B)
        print(f"{core:<11} SR基线残差={relb:.4f}(λ={lam_used:g})  变体A r={rA:.4f}  变体B r={rB:.4f}  "
              f"A峰={results['variant_A_signal_align'][core]['peaks_v2']} 基线峰={results['variant_A_signal_align'][core]['peaks_SRbase']}")
    
    ra = [v["r_vs_SRbaseline"] for v in results["variant_A_signal_align"].values()]
    rb = [v["r_vs_SRbaseline"] for v in results["variant_B_kernel_swap"].values()]
    results["meta"] = {"mean_A": round(float(np.mean(ra)), 4), "mean_B": round(float(np.mean(rb)), 4),
                       "note": "对照: e8 中未对齐零样本一致性(砂岩均值)=0.1976(混域旧口径)/见 e8_cross_material.json 线性域现值"}
    print(f"均值: 变体A={results['meta']['mean_A']:.4f}  变体B={results['meta']['mean_B']:.4f}")
    
    with open(os.path.join(HERE, "results_archive", "e8_sr_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, core in zip(axs.ravel(), sorted(maps.keys())):
        Mb, Ma, Mbv = maps[core]
        ax.plot(Mb.sum(0), label="SR-SVD 基线", color="#2471a3")
        ax.plot(Ma.sum(0), label="v2 变体A(信号对齐)", color="#c0392b", ls="--")
        ax.plot(Mbv.sum(0), label="v2 变体B(SR核)", color="#1e8449", ls=":")
        ax.set_xscale("log")
        ax.set_title(f"{core}  T2 边缘谱", fontsize=10)
        ax.legend(fontsize=7)
    plt.suptitle("E8-SR 对齐: 冻结 v2 在 2MHz 砂岩 SR 数据上的零样本迁移")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "e8_sr_alignment.png"), dpi=150)
    print("e8_sr_alignment.json / .png 已存")

if __name__ == "__main__":
    main()
