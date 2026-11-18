# -*- coding: utf-8 -*-
"""
E6: SNR 鲁棒性曲线——学习式 vs 物理基线，SNR 5→500
每个 SNR 50 个新鲜样本；学习式用已训好的 v2 模型（EMA 权重，seed3）。
"""
import io, sys, os, json, glob
import numpy as np
import torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import make_sample, N_TI, NECH, N_T1, N_T2, DL1, DL2, TI_MS
from train2d_v2 import PhysicsFrontEnd, RefineNet, log_pool_np
from baseline2d import reconstruct_2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 v2 模型（用 seed3 的最终 EMA 权重）
front = PhysicsFrontEnd().to(DEVICE)
front.eval()
ckpt = torch.load(os.path.join(HERE, "v2_refine.pth"), map_location=DEVICE, weights_only=True)
refine = RefineNet().to(DEVICE)
refine.load_state_dict(ckpt["refine"])
refine.eval()

SNRS = [5, 10, 20, 50, 100, 200, 500]
N_PER = 50


def pearson_flat(a, b):
    pa = a / (a.sum() + 1e-12)
    pb = b / (b.sum() + 1e-12)
    return float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])


def main():
    rng = np.random.default_rng(20261115)
    results = {}
    for snr in SNRS:
        r_learn, r_base = [], []
        for _ in range(N_PER):
            # 手动指定 SNR（覆盖 make_sample 的随机值）
            spec = None
            while spec is None:
                try:
                    from sim2d import sample_spectrum
                    spec_raw = None  # make_sample 内部采样，改为手动构造
                    # 简化：直接调 make_sample 但事后按 clean 重算实际 SNR 并归到目标
                    s = make_sample(rng, snr=snr)
                    spec = s["spec"]
                except Exception:
                    spec = None
            # baseline
            b = reconstruct_2d(s["y"], s["mask"], lam=1e-3)
            r_base.append(pearson_flat(b, s["spec"]))
            # learned
            ys = np.real(s["y"]) / (np.abs(np.real(s["y"])).max() + 1e-9)
            yt = torch.tensor(ys[None], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                base_t = front(yt)
                pooled = torch.tensor(np.nan_to_num(log_pool_np(ys[None]), nan=0.0),
                                      dtype=torch.float32, device=DEVICE)
                pred_log = refine(base_t, pooled)[0].cpu().numpy()
            pred = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
            r_learn.append(pearson_flat(pred, s["spec"]))
        results[snr] = {"learned": round(float(np.mean(r_learn)), 4),
                        "baseline": round(float(np.mean(r_base)), 4),
                        "learned_std": round(float(np.std(r_learn)), 4),
                        "baseline_std": round(float(np.std(r_base)), 4)}
        print(f"  SNR={snr:>3}: learned={results[snr]['learned']:.4f}±{results[snr]['learned_std']:.4f} | "
              f"baseline={results[snr]['baseline']:.4f}±{results[snr]['baseline_std']:.4f}", flush=True)

    with open(os.path.join(HERE, "e6_snr_robustness.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("e6_snr_robustness.json 已存", flush=True)

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    snr_list = sorted(results.keys())
    ax.errorbar(snr_list, [results[s]["learned"] for s in snr_list],
                yerr=[results[s]["learned_std"] for s in snr_list],
                fmt="o-", color="tab:blue", capsize=3,
                label="Learned (physics front-end + refine)", ms=4)
    ax.errorbar(snr_list, [results[s]["baseline"] for s in snr_list],
                yerr=[results[s]["baseline_std"] for s in snr_list],
                fmt="s--", color="tab:orange", capsize=3,
                label="Physics baseline (SVD-ridge)", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("SNR")
    ax.set_ylabel("spectrum pearson r")
    ax.set_title("SNR robustness (n = 50 per point)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "figures", "e6_snr_robustness.png"), dpi=300)
    plt.close()
    print("e6_snr_robustness.png 已存", flush=True)


if __name__ == "__main__":
    main()
