# -*- coding: utf-8 -*-
"""
train2d.py —— 学习式 2D T1-T2 反演器原型（第二篇原型 Step 4）
- 数据：sim2d 在线无限生成（SNR 30–300, 1–3 斑, 对数回波池化 128）
- 模型：CNN 编码器 (1,20,128) → FC → (48,96) → conv 精修 → softplus
- 损失：MSE on sqrt(spec)（动态范围稳定）
- 评测：sim test 与 2D 物理基线对照；真实 13 样本与基线图一致性
"""
import io, sys, os, time, json, glob
import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import make_sample, N_TI, NECH, N_T1, N_T2, TI_MS, TE_MS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 4000
BATCH = 32
N_POOL = 128  # 对数回波池化后长度

# 对数回池边界（echo 1..2000 对应 te=1.027..2054ms，按 log 均匀分 128 桶）
_edges = np.unique(np.logspace(0, np.log10(NECH), N_POOL + 1).astype(int) - 1)
_edges[-1] = NECH
POOLIDX = [_edges[i:i + 2] for i in range(len(_edges) - 1)]
N_POOL = len(POOLIDX)


def log_pool(y):
    """(B,20,2000) → (B,20,N_POOL) 分桶均值。"""
    out = np.empty((y.shape[0], y.shape[1], N_POOL), dtype=np.float32)
    for j, (a, b) in enumerate(POOLIDX):
        out[:, :, j] = y[:, :, a:b].mean(axis=2)
    return out


class Inv2DNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 32
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, (3, 7), stride=(1, 2), padding=(1, 3)), nn.ReLU(),
            nn.Conv2d(ch, ch, (3, 5), stride=(2, 2), padding=(1, 2)), nn.ReLU(),
            nn.Conv2d(ch, ch, (3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(ch, ch, (3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(ch * 16, 512)
        self.head = nn.Sequential(
            nn.Linear(512, N_T1 * N_T2),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        z = self.enc(x)
        z = torch.flatten(z, 1)
        z = torch.relu(self.fc(z))
        m = self.head(z).view(-1, 1, N_T1, N_T2)
        m = self.refine(m)
        return torch.nn.functional.softplus(m.squeeze(1))


def prep_batch(rng, batch):
    ys, specs = [], []
    for _ in range(batch):
        s = make_sample(rng)
        ys.append(np.real(s["y"]) / (np.abs(np.real(s["y"])).max() + 1e-9))
        specs.append(s["spec"])
    x = log_pool(np.stack(ys))                     # (B,20,N_POOL)
    return (torch.tensor(x[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(specs), dtype=torch.float32, device=DEVICE))


def main():
    print(f"[C-2D-learned] device={DEVICE}", flush=True)
    torch.manual_seed(1); np.random.seed(1)
    model = Inv2DNet().to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {nparam:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-4)
    rng = np.random.default_rng(20261112)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, spec = prep_batch(rng, BATCH)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(torch.sqrt(pred + 1e-9),
                                            torch.sqrt(spec + 1e-9))
        if not torch.isfinite(loss):
            opt.zero_grad(); continue
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == STEPS:
            print(f"  [step {step:>4}] sqrt-MSE={float(loss):.4f} ({time.time()-t0:.0f}s)",
                  flush=True)

    # ---------- 评测：sim test 200 ----------
    torch.save({"model_state_dict": model.state_dict(), "params": nparam, "steps": STEPS},
               os.path.join(HERE, "c_learned_model.pth"))

    @torch.no_grad()
    def eval_sim(n=200):
        rs_l, rs_b = [], []
        from baseline2d import reconstruct_2d, metrics
        for _ in range(n):
            s = make_sample(rng)
            x = log_pool(np.real(s["y"])[None] / (np.abs(np.real(s["y"])).max() + 1e-9))
            xt = torch.tensor(x[:, None], dtype=torch.float32, device=DEVICE)
            pred = model(xt).cpu().numpy()[0]
            base = reconstruct_2d(s["y"], s["mask"], lam=1e-3)
            rs_l.append(metrics(pred, s["spec"])["pearson"])
            rs_b.append(metrics(base, s["spec"])["pearson"])
        return float(np.mean(rs_l)), float(np.mean(rs_b))

    r_learn, r_base = eval_sim(200)
    print(f"sim test(200): 学习式 pearson={r_learn:.4f} | 物理基线={r_base:.4f}", flush=True)

    # ---------- 真实 13 样本 ----------
    try:
        from realdata2d import load_sample, phase_correct, LIB
    except ImportError:
        load_sample = phase_correct = None
        LIB = os.environ.get("NMR23_DATA_DIR", "")
    out = []
    dirs = sorted(glob.glob(os.path.join(LIB, "*__IR-CPMG"))) if (load_sample and LIB and os.path.isdir(LIB)) else []
    if not dirs:
        print("Restricted 23 MHz raw waveforms not attached (set NMR23_DATA_DIR to evaluate); saving synthetic summary only.", flush=True)
    for d in dirs:
        s = load_sample(d)
        Yr = phase_correct(s["Yc"])
        x = np.real(Yr) / (np.abs(np.real(Yr)).max() + 1e-9)
        xt = torch.tensor(log_pool(x[None])[:, None], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred = model(xt).cpu().numpy()[0]
        p1 = pred.sum(1); p2 = pred.sum(0)
        t1p = 10 ** (np.log10(8.0) + (np.log10(2000.0) - np.log10(8.0)) * int(np.argmax(p1)) / (N_T1 - 1))
        t2p = 10 ** (np.log10(0.5) + (np.log10(3000.0) - np.log10(0.5)) * int(np.argmax(p2)) / (N_T2 - 1))
        out.append({"name": s["name"], "learned_T1_peak": round(float(t1p), 1),
                    "learned_T2_peak": round(float(t2p), 1)})
        print(f"  {s['name']:<14} 学习式 T1峰={t1p:>7.1f} T2峰={t2p:>7.1f}", flush=True)

    (os.path.join(HERE, "c_learned_summary.json")).write_text if False else None
    with open(os.path.join(HERE, "c_learned_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"sim_test_learned": r_learn, "sim_test_baseline": r_base,
                   "real_peaks": out, "params": nparam, "steps": STEPS},
                  f, ensure_ascii=False, indent=2)
    print("c_learned_summary.json 已存", flush=True)


if __name__ == "__main__":
    main()
