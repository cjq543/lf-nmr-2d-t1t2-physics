# -*- coding: utf-8 -*-
"""
E1: 带符号 vs 幅度截断消融（第二篇方法学卖点之二）
两模式同架构同预算：signed=Re(复数)/max；magnitude=|复数|/max（仪器只存幅度的常规做法）。
指标：sim test 200 的 pearson + T1 峰位误差（|Δlog10 T1|，真值已知）。
"""
import io, sys, os, time, json, copy
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import make_sample, N_TI, NECH, N_T1, N_T2, TI_MS, TE_MS, DL1, DL2, LOG_T1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 4000
BATCH = 32

_edges = np.unique(np.logspace(0, np.log10(NECH), 129).astype(int) - 1)
_edges[-1] = NECH
POOLIDX = [_edges[i:i + 2] for i in range(len(_edges) - 1)]

_T1 = np.logspace(np.log10(8.0), np.log10(2000.0), N_T1)
_T2 = np.logspace(np.log10(0.5), np.log10(3000.0), N_T2)
K_T1 = 1.0 - 2.0 * np.exp(-TI_MS[:, None] / _T1[None, :])
K_T2 = np.exp((np.arange(1, NECH + 1) * TE_MS)[:, None] / _T2[None, :] * -1.0)


class PhysicsFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("K1", torch.tensor(K_T1, dtype=torch.float32))
        self.register_buffer("K2", torch.tensor(K_T2, dtype=torch.float32))
        U1, S1, V1t = torch.linalg.svd(self.K1, full_matrices=False)
        U2, S2, V2t = torch.linalg.svd(self.K2, full_matrices=False)
        r1 = int((S1 > S1[0] * 1e-10).sum()); r2 = int((S2 > S2[0] * 1e-10).sum())
        self.register_buffer("U1c", U1[:, :r1]); self.register_buffer("S1c", S1[:r1])
        self.register_buffer("V1c", V1t[:r1])
        self.register_buffer("U2c", U2[:, :r2]); self.register_buffer("S2c", S2[:r2])
        self.register_buffer("V2c", V2t[:r2])

    def forward(self, y):
        Ycore = self.U1c.t() @ y @ self.U2c
        s = self.S1c[:, None] * self.S2c[None, :]
        smax = self.S1c[0] * self.S2c[0]
        C = s * Ycore / (s ** 2 + (1e-3 * smax) ** 2).clamp(min=1e-12)
        M = torch.relu(self.V1c.t() @ C @ self.V2c)
        area = (M.sum(dim=(1, 2), keepdim=True) * DL1 * DL2).clamp(min=1e-9)
        return torch.log1p(M / area * 20.0)


def log_pool(y):
    out = np.empty((y.shape[0], y.shape[1], len(POOLIDX)), dtype=np.float32)
    for j, (a, b) in enumerate(POOLIDX):
        if b > a:
            out[:, :, j] = y[:, :, a:b].mean(axis=2)
    return out


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 32
        self.refine = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, base_map, pooled):
        p = torch.nn.functional.interpolate(pooled.unsqueeze(1), size=(N_T1, N_T2),
                                            mode="bilinear", align_corners=False)
        x = torch.cat([base_map.unsqueeze(1), p], dim=1)
        return torch.nn.functional.softplus(self.refine(x).squeeze(1))


def t1_peak_err(pred_log, spec):
    pred = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
    t = spec / (spec.sum() * DL1 * DL2 + 1e-12)
    p = pred / (pred.sum() * DL1 * DL2 + 1e-12)
    ip = int(np.argmax(p.sum(1))); it = int(np.argmax(t.sum(1)))
    return abs(LOG_T1[ip] - LOG_T1[it])


def run_mode(mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    front = PhysicsFrontEnd().to(DEVICE)
    net = RefineNet().to(DEVICE)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=8e-5)
    rng = np.random.default_rng(20261113 + seed)

    def to_input(s):
        y = np.real(s["y"]) if mode == "signed" else np.abs(s["y"])
        return y / (np.abs(y).max() + 1e-9)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        ys, specs, xs = [], [], []
        for _ in range(BATCH):
            s = make_sample(rng)
            ys.append(to_input(s))
            specs.append(s["spec"])
        yt = torch.tensor(np.stack(ys)[:, None], dtype=torch.float32, device=DEVICE)
        st = torch.tensor(np.stack(specs), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            base = front(yt[:, 0])
        pooled = torch.tensor(log_pool(np.stack(ys)), dtype=torch.float32, device=DEVICE)
        target = torch.log1p(st * 20.0)
        pred = net(base, pooled)
        rel = torch.sqrt(((pred - target) ** 2).mean(dim=(1, 2))) / \
              (torch.sqrt((target ** 2).mean(dim=(1, 2))) + 1e-9)
        loss = rel.mean()
        if not torch.isfinite(loss):
            opt.zero_grad(); continue
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pn, alpha=0.001)
        if step % 1000 == 0:
            print(f"  [{mode} seed{seed} step {step}] relMSE={float(loss):.4f} ({time.time()-t0:.0f}s)",
                  flush=True)

    net.load_state_dict(ema.state_dict()); net.eval()
    rng_e = np.random.default_rng(20261114 + seed)
    pear, terr = [], []
    with torch.no_grad():
        for _ in range(200):
            s = make_sample(rng_e)
            x = to_input(s)
            yt = torch.tensor(x[None, None], dtype=torch.float32, device=DEVICE)
            base = front(yt[:, 0])[0].cpu().numpy()
            pooled = torch.tensor(log_pool(x[None]), dtype=torch.float32, device=DEVICE)
            baset = torch.tensor(base[None], dtype=torch.float32, device=DEVICE)
            pred_log = net(baset, pooled)[0].cpu().numpy()
            pred = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
            t = s["spec"] / (s["spec"].sum() * DL1 * DL2 + 1e-12)
            p = pred / (pred.sum() * DL1 * DL2 + 1e-12)
            pear.append(float(np.corrcoef(p.ravel(), t.ravel())[0, 1]))
            ip = int(np.argmax(p.sum(1))); it = int(np.argmax(t.sum(1)))
            terr.append(abs(LOG_T1[ip] - LOG_T1[it]))
    return {"pearson": float(np.mean(pear)), "t1_err_mean": float(np.mean(terr)),
            "t1_err_median": float(np.median(terr))}


def main():
    # 支持单运行模式（CUDA 偶发崩溃隔离）：e1_sign_ablation.py <mode> <seed>
    if len(sys.argv) >= 3:
        mode, seed = sys.argv[1], int(sys.argv[2])
        r = run_mode(mode, seed)
        with open(os.path.join(HERE, f"e1_{mode}_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print(f"[{mode} seed{seed}] pearson={r['pearson']:.4f} "
              f"T1峰位误差 mean={r['t1_err_mean']:.3f} median={r['t1_err_median']:.3f}", flush=True)
        return
    out = {}
    for mode in ("signed", "magnitude"):
        for seed in (1, 2):
            r = run_mode(mode, seed)
            out[f"{mode}_seed{seed}"] = r
            print(f"[{mode} seed{seed}] pearson={r['pearson']:.4f} "
                  f"T1峰位误差 mean={r['t1_err_mean']:.3f} median={r['t1_err_median']:.3f}",
                  flush=True)
    for mode in ("signed", "magnitude"):
        ks = [k for k in out if k.startswith(mode)]
        out[f"{mode}_mean"] = {"pearson": float(np.mean([out[k]["pearson"] for k in ks])),
                               "t1_err_mean": float(np.mean([out[k]["t1_err_mean"] for k in ks]))}
        print(f"[{mode}] 两种子均值: {out[f'{mode}_mean']}", flush=True)
    with open(os.path.join(HERE, "e1_sign_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("e1_sign_ablation.json 已存", flush=True)


if __name__ == "__main__":
    main()
