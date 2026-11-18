# -*- coding: utf-8 -*-
"""
train2d_v2.py —— 物理前端 + 学习修正（model-based deep learning）
v1 的病根：从头学反演（任务太难）+ 输入分布 sim/real 不配（迁移崩）。
v2 修法：SVD-ridge 物理重建做成可微 torch 模块（第一层），网络只学修正残差；
        EMA 权重 + 相对损失（去 SNR 方差）+ 3 种子。
"""
import io, sys, os, time, json, glob, copy
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import make_sample, N_TI, NECH, N_T1, N_T2, TI_MS, TE_MS, DL1, DL2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 4000
BATCH = 32
N_POOL = 128

_edges = np.unique(np.logspace(0, np.log10(NECH), N_POOL + 1).astype(int) - 1)
_edges[-1] = NECH
POOLIDX = [_edges[i:i + 2] for i in range(len(_edges) - 1)]
N_POOL = len(POOLIDX)

# 谱网格（与 sim2d 一致）
_T1 = np.logspace(np.log10(8.0), np.log10(2000.0), N_T1)
_T2 = np.logspace(np.log10(0.5), np.log10(3000.0), N_T2)
K_T1 = 1.0 - 2.0 * np.exp(-TI_MS[:, None] / _T1[None, :])
K_T2 = np.exp((np.arange(1, NECH + 1) * TE_MS)[:, None] / _T2[None, :] * -1.0)


class PhysicsFrontEnd(nn.Module):
    """可微 SVD-ridge 2D 重建（按固定 20-TI 网格）。输出 log1p 谱图。"""

    def __init__(self, lam=1e-3):
        super().__init__()
        self.register_buffer("K1", torch.tensor(K_T1, dtype=torch.float32))
        self.register_buffer("K2", torch.tensor(K_T2, dtype=torch.float32))
        self.lam = lam
        # 预分解
        U1, S1, V1t = torch.linalg.svd(self.K1, full_matrices=False)
        U2, S2, V2t = torch.linalg.svd(self.K2, full_matrices=False)
        tol = S1[0] * 1e-10
        r1 = int((S1 > tol).sum()); r2 = int((S2 > S2[0] * 1e-10).sum())
        self.register_buffer("U1c", U1[:, :r1]); self.register_buffer("S1c", S1[:r1])
        self.register_buffer("V1c", V1t[:r1])
        self.register_buffer("U2c", U2[:, :r2]); self.register_buffer("S2c", S2[:r2])
        self.register_buffer("V2c", V2t[:r2])

    def forward(self, y):                     # y (B,20,2000) 实数
        Ycore = self.U1c.t() @ y @ self.U2c   # (B,r1,r2)
        s = self.S1c[:, None] * self.S2c[None, :]
        smax = self.S1c[0] * self.S2c[0]
        C = s * Ycore / (s ** 2 + (self.lam * smax) ** 2).clamp(min=1e-12)
        M = torch.relu(self.V1c.t() @ C @ self.V2c)
        area = (M.sum(dim=(1, 2), keepdim=True) * DL1 * DL2).clamp(min=1e-9)
        return torch.log1p(M / area * 20.0)   # log1p 压动态范围


class RefineNet(nn.Module):
    """修正网络：输入 (2,48,96)=log1p基线图 + 池化数据图，输出修正谱。"""

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
        # pooled (B,20,128) → 上采样到 (48,96) 再拼通道
        p = torch.nn.functional.interpolate(pooled.unsqueeze(1), size=(N_T1, N_T2),
                                            mode="bilinear", align_corners=False)
        x = torch.cat([base_map.unsqueeze(1), p], dim=1)
        return torch.nn.functional.softplus(self.refine(x).squeeze(1))


def log_pool_np(y):
    """按训练回波时间轴对数分箱池化。y 的回波数 < NECH 时（如 600 回波参照样），
    覆盖不到的分箱显式置 0（表示采集截断），不再产生 NaN 告警。"""
    out = np.zeros((y.shape[0], y.shape[1], N_POOL), dtype=np.float32)
    n = y.shape[2]
    for j, (a, b) in enumerate(POOLIDX):
        aa, bb = min(a, n), min(b, n)
        if bb > aa:
            out[:, :, j] = y[:, :, aa:bb].mean(axis=2)
    return out


def prep_batch(rng, batch, front):
    ys, specs = [], []
    for _ in range(batch):
        s = make_sample(rng)
        ys.append(np.real(s["y"]) / (np.abs(np.real(s["y"])).max() + 1e-9))
        specs.append(s["spec"])
    yt = torch.tensor(np.stack(ys), dtype=torch.float32, device=DEVICE)
    st = torch.tensor(np.stack(specs), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        base = front(yt)
    pooled = torch.tensor(log_pool_np(np.stack(ys)), dtype=torch.float32, device=DEVICE)
    return yt, base, pooled, st


def main():
    print(f"[v2] device={DEVICE}", flush=True)
    front = PhysicsFrontEnd().to(DEVICE)
    results = {}
    for seed in (1, 2, 3):
        torch.manual_seed(seed); np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        net = RefineNet().to(DEVICE)
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=8e-5)
        rng = np.random.default_rng(20261112 + seed)
        ema_decay = 0.999
        t0 = time.time()
        for step in range(1, STEPS + 1):
            yt, base, pooled, spec = prep_batch(rng, BATCH, front)
            target = torch.log1p(spec * 20.0)          # 与基线图同域（log1p, 0~7）
            pred = net(base, pooled)
            rel = torch.sqrt(((pred - target) ** 2).mean(dim=(1, 2))) / \
                  (torch.sqrt((target ** 2).mean(dim=(1, 2))) + 1e-9)
            loss = rel.mean()
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(ema_decay).add_(pn, alpha=1 - ema_decay)
            if step % 1000 == 0:
                print(f"  [seed{seed} step {step}] relMSE={float(loss):.4f} ({time.time()-t0:.0f}s)",
                      flush=True)

        # ---------- 评测（EMA 权重） ----------
        net.load_state_dict(ema.state_dict())
        net.eval()

        @torch.no_grad()
        def eval_sim(n=200, seed_off=100):
            r_l, r_b = [], []
            rng_e = np.random.default_rng(20261112 + seed_off)
            from baseline2d import reconstruct_2d
            for _ in range(n):
                s = make_sample(rng_e)
                ys = np.real(s["y"]) / (np.abs(np.real(s["y"])).max() + 1e-9)
                yt = torch.tensor(ys[None], dtype=torch.float32, device=DEVICE)
                base = front(yt)[0]
                pooled = torch.tensor(log_pool_np(ys[None]), dtype=torch.float32,
                                      device=DEVICE)
                pred_log = net(base[None], pooled)[0].cpu().numpy()
                pred = np.expm1(np.clip(pred_log, 0, 20)) / 20.0   # log1p 域还原
                bnp = reconstruct_2d(s["y"], s["mask"], lam=1e-3)
                for M, acc in [(pred, r_l), (bnp, r_b)]:
                    p = M / (M.sum() * DL1 * DL2 + 1e-12)
                    t = s["spec"] / (s["spec"].sum() * DL1 * DL2 + 1e-12)
                    acc.append(float(np.corrcoef(p.ravel(), t.ravel())[0, 1]))
            return float(np.mean(r_l)), float(np.mean(r_b))

        r_l, r_b = eval_sim()
        print(f"  [seed{seed}] sim test: 学习修正={r_l:.4f} | 物理基线={r_b:.4f}", flush=True)
        results[f"seed{seed}"] = {"learned": r_l, "baseline": r_b}

    mean_l = float(np.mean([results[k]["learned"] for k in results]))
    mean_b = float(np.mean([results[k]["baseline"] for k in results]))
    results["summary"] = {"learned_mean": mean_l, "baseline_mean": mean_b}
    torch.save({"refine": net.state_dict(), "results": results},
               os.path.join(HERE, "v2_refine.pth"))
    with open(os.path.join(HERE, "v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[v2] 三种子: 学习修正={mean_l:.4f} vs 物理基线={mean_b:.4f}", flush=True)


if __name__ == "__main__":
    main()
