# -*- coding: utf-8 -*-
"""
E7: 组分数鲁棒性——学习式 vs 物理基线，1–5 个斑，每个组分数 50 个样本
"""
import io, sys, os, json
import numpy as np
import torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import make_sample, N_TI, NECH, DL1, DL2
from train2d_v2 import PhysicsFrontEnd, RefineNet, log_pool_np
from baseline2d import reconstruct_2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
front = PhysicsFrontEnd().to(DEVICE); front.eval()
ckpt = torch.load(os.path.join(HERE, "v2_refine.pth"), map_location=DEVICE, weights_only=True)
refine = RefineNet().to(DEVICE); refine.load_state_dict(ckpt["refine"]); refine.eval()


def pearson_flat(a, b):
    pa = a / (a.sum() + 1e-12); pb = b / (b.sum() + 1e-12)
    return float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1])


def make_with_ncomp(rng, ncomp, snr):
    """强制指定斑数生成样本。"""
    from sim2d import sample_spectrum, K_T1, K_T2
    # monkey-patch: 覆盖 sample_spectrum 的 n_comp 逻辑
    orig = sample_spectrum
    spec = orig(rng, n_comp=ncomp)
    clean = K_T1 @ spec @ K_T2.T
    sigma = np.abs(clean).max() / snr
    noise = (rng.normal(0, sigma, clean.shape) +
             1j * rng.normal(0, sigma, clean.shape)) / np.sqrt(2)
    y = clean + noise
    mask = np.ones(clean.shape)
    return {"y": (y).astype(np.complex128), "clean": clean, "spec": spec, "mask": mask}


def main():
    rng = np.random.default_rng(20261116)
    results = {}
    for ncomp in range(1, 6):
        r_l, r_b = [], []
        for _ in range(50):
            s = make_with_ncomp(rng, ncomp, snr=100)
            b = reconstruct_2d(s["y"], s["mask"], lam=1e-3)
            r_b.append(pearson_flat(b, s["spec"]))
            ys = np.real(s["y"]) / (np.abs(np.real(s["y"])).max() + 1e-9)
            yt = torch.tensor(ys[None], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                base_t = front(yt)
                pooled = torch.tensor(np.nan_to_num(log_pool_np(ys[None]), nan=0.0),
                                      dtype=torch.float32, device=DEVICE)
                pred_log = refine(base_t, pooled)[0].cpu().numpy()
            pred = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
            r_l.append(pearson_flat(pred, s["spec"]))
        results[ncomp] = {"learned": round(float(np.mean(r_l)), 4),
                          "baseline": round(float(np.mean(r_b)), 4),
                          "learned_std": round(float(np.std(r_l)), 4),
                          "baseline_std": round(float(np.std(r_b)), 4)}
        print(f"  ncomp={ncomp}: learned={results[ncomp]['learned']:.4f}±{results[ncomp]['learned_std']:.4f} | "
              f"baseline={results[ncomp]['baseline']:.4f}±{results[ncomp]['baseline_std']:.4f}", flush=True)

    with open(os.path.join(HERE, "e7_component_robustness.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("e7_component_robustness.json 已存", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ncs = sorted(results.keys())
    ax.errorbar(ncs, [results[n]["learned"] for n in ncs],
                yerr=[results[n]["learned_std"] for n in ncs],
                fmt="o-", color="tab:blue", capsize=3, ms=4,
                label="Learned (physics front-end + refine)")
    ax.errorbar(ncs, [results[n]["baseline"] for n in ncs],
                yerr=[results[n]["baseline_std"] for n in ncs],
                fmt="s--", color="tab:orange", capsize=3, ms=4,
                label="Physics baseline (SVD-ridge)")
    ax.set_xlabel("Number of spectral components")
    ax.set_ylabel("spectrum pearson r")
    ax.set_title("Component count robustness (SNR = 100, n = 50 per point)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "figures", "e7_component_robustness.png"), dpi=300)
    plt.close()
    print("e7_component_robustness.png 已存", flush=True)


if __name__ == "__main__":
    main()
