# -*- coding: utf-8 -*-
"""
E9: Wide-grid Saturation-Recovery (SR) porous-rock retraining & evaluation (v2rock).

Public reproduction module for Paper #2 (Measurement Science and Technology).
Requires external 2.0 MHz Magritek datasets via NMR2_ROCK_DATA_DIR only for re-running
raw H2/H4 core evaluations; offline synthetic benchmark (H1) and archived results run standalone.
"""
import argparse
import copy
import csv
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    pass  # configured cleanly inside main() when invoked as script

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from train2d_v2 import RefineNet, log_pool_np

T1E = np.logspace(np.log10(0.05), np.log10(5000.0), 48)
T2E = np.logspace(np.log10(0.02), np.log10(5000.0), 96)
TAUS = np.logspace(np.log10(500.0), np.log10(0.1), 20)
NECH, TE = 2000, 0.06
DL1 = np.log(10) * (np.log10(T1E[-1]) - np.log10(T1E[0])) / len(T1E)
DL2 = np.log(10) * (np.log10(T2E[-1]) - np.log10(T2E[0])) / len(T2E)
LAM = 1e-4

K1 = 1.0 - np.exp(-TAUS[:, None] / T1E[None, :])
K2 = np.exp(-(np.arange(1, NECH + 1) * TE)[:, None] / T2E[None, :])
U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
r1 = int((S1 > S1[0] * 1e-10).sum())
r2 = int((S2 > S2[0] * 1e-10).sum())
U1t, S1c, V1c = U1[:, :r1], S1[:r1], V1t[:r1]
U2t, S2c, V2c = U2[:, :r2], S2[:r2], V2t[:r2]
SS = S1c[:, None] * S2c[None, :]
SMAX = S1c[0] * S2c[0]


class RockFrontEnd(nn.Module):
    """Differentiable frozen SR SVD-ridge front-end on expanded wide grid (lam = 1e-4)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("U1", torch.tensor(U1t, dtype=torch.float32))
        self.register_buffer("S1", torch.tensor(S1c, dtype=torch.float32))
        self.register_buffer("V1", torch.tensor(V1c, dtype=torch.float32))
        self.register_buffer("U2", torch.tensor(U2t, dtype=torch.float32))
        self.register_buffer("S2", torch.tensor(S2c, dtype=torch.float32))
        self.register_buffer("V2", torch.tensor(V2c, dtype=torch.float32))
        self.register_buffer("ss", self.S1[:, None] * self.S2[None, :])
        self.smax = float(S1c[0] * S2c[0])

    def forward(self, y):
        yc = torch.einsum("ti,btj->bij", self.U1, y)
        yc = torch.einsum("bij,jk->bik", yc, self.U2)
        c = self.ss * yc / (self.ss ** 2 + (LAM * self.smax) ** 2).clamp(min=1e-12)
        m = torch.relu(torch.einsum("ir,bij,jq->brq", self.V1, c, self.V2))
        area = (m.sum(dim=(1, 2)) * DL1 * DL2).clamp(min=1e-9)
        return torch.log1p(m / area[:, None, None] * 20.0)


def rock_sample(rng):
    """Generate bimodal synthetic porous-medium spectrum with 10-200 density scaling."""
    spec = np.zeros((len(T1E), len(T2E)))

    def blob(lt1, lt2, slt1, slt2, w):
        g = np.exp(
            -0.5 * ((np.log10(T1E)[:, None] - lt1) / slt1) ** 2
            - 0.5 * ((np.log10(T2E)[None, :] - lt2) / slt2) ** 2
        )
        return w * g / (g.sum() + 1e-12)

    comps = []
    lt2 = rng.uniform(-1.0, 3.5)
    ratio = 10 ** rng.uniform(0.18, 0.7)
    lt1 = np.log10(10 ** lt2 * ratio)
    comps.append((lt1, lt2, rng.uniform(0.12, 0.35), rng.uniform(0.10, 0.30)))
    if rng.uniform() < 0.75:
        lt1f = rng.uniform(-0.3, 0.9)
        rf = 10 ** rng.uniform(0.18, 0.9)
        comps.append((lt1f, lt1f - np.log10(rf), rng.uniform(0.10, 0.30), rng.uniform(0.08, 0.25)))
    if rng.uniform() < 0.2:
        comps.append((rng.uniform(1.3, 2.4), rng.uniform(-0.5, 1.5), rng.uniform(0.10, 0.25), rng.uniform(0.10, 0.25)))
    w = rng.dirichlet(np.ones(len(comps)) * 2.0)
    for c, wi in zip(comps, w):
        spec += blob(c[0], c[1], c[2], c[3], wi)
    spec = spec / (spec.max() + 1e-12) * 10 ** rng.uniform(1.0, 2.3)
    return spec


def prep_batch(rng, batch, device):
    ys, specs = [], []
    for _ in range(batch):
        spec = rock_sample(rng)
        clean = K1 @ spec @ K2.T
        snr = 10 ** rng.uniform(np.log10(5), 2.0)
        y = clean + rng.normal(0, clean.max() / snr, clean.shape)
        ys.append(y / (np.abs(y).max() + 1e-9))
        specs.append(spec)
    return (
        torch.tensor(np.array(ys), dtype=torch.float32, device=device),
        torch.tensor(np.array(specs), dtype=torch.float32, device=device),
    )


def train_seed(seed, front, device, steps=4000, batch=32):
    ckt = os.path.join(HERE, f"e9_v2rock_seed{seed}.pth")
    if os.path.isfile(ckt):
        return ckt
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    net = RefineNet().to(device)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=8e-5)
    rng = np.random.default_rng(20261115 + seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        yt, spec = prep_batch(rng, batch, device)
        with torch.no_grad():
            base = front(yt)
        pooled = torch.tensor(log_pool_np(yt.detach().cpu().numpy()), dtype=torch.float32, device=device)
        target = torch.log1p(spec * 20.0)
        pred = net(base, pooled)
        rel = torch.sqrt(((pred - target) ** 2).mean(dim=(1, 2))) / (
            torch.sqrt((target ** 2).mean(dim=(1, 2))) + 1e-9
        )
        loss = rel.mean()
        if not torch.isfinite(loss):
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pn, alpha=1 - 0.999)
        if step % 500 == 0:
            print(f"  [seed{seed} step {step}] relMSE={float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"refine": ema.state_dict()}, ckt)
    return ckt


def eval_sim(front, refine_ema, device, n=200, seed_off=100):
    rng = np.random.default_rng(20261115 + seed_off)
    r_l, r_b, pe = [], [], {"v2rock": [], "baseline": []}
    lt1, lt2 = np.log10(T1E), np.log10(T2E)
    for _ in range(n):
        spec = rock_sample(rng)
        clean = K1 @ spec @ K2.T
        snr = 10 ** rng.uniform(np.log10(5), 2.0)
        y = clean + rng.normal(0, clean.max() / snr, clean.shape)
        ys = y / (np.abs(y).max() + 1e-9)
        yt = torch.tensor(ys[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            base = front(yt)
            pooled = torch.tensor(log_pool_np(ys[None]), dtype=torch.float32, device=device)
            pred_log = refine_ema(base, pooled)[0].cpu().numpy()
        pred = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
        yc = U1t.T @ ys @ U2t
        c = SS * yc / (SS ** 2 + (LAM * SMAX) ** 2)
        mb = np.clip(V1c.T @ c @ V2c, 0, None)
        mb = mb / (mb.sum() * DL1 * DL2 + 1e-12)
        p = pred / (pred.sum() * DL1 * DL2 + 1e-12)
        t = spec / (spec.sum() * DL1 * DL2 + 1e-12)
        r_l.append(float(np.corrcoef(p.ravel(), t.ravel())[0, 1]))
        r_b.append(float(np.corrcoef(mb.ravel(), t.ravel())[0, 1]))
        for nm, mm in [("v2rock", pred), ("baseline", mb)]:
            e1 = abs(lt1[int(np.argmax(mm.sum(1)))] - lt1[int(np.argmax(spec.sum(1)))])
            e2 = abs(lt2[int(np.argmax(mm.sum(0)))] - lt2[int(np.argmax(spec.sum(0)))])
            pe[nm].append((e1, e2))
    res = {"v2rock_pearson": round(float(np.mean(r_l)), 4), "baseline_pearson": round(float(np.mean(r_b)), 4)}
    for nm in ("v2rock", "baseline"):
        a = np.array(pe[nm])
        res[f"{nm}_t1err"] = [round(float(a[:, 0].mean()), 4), round(float(np.median(a[:, 0])), 4)]
        res[f"{nm}_t2err"] = [round(float(a[:, 1].mean()), 4), round(float(np.median(a[:, 1])), 4)]
    return res


def load_num(p):
    good = []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) < 2:
                continue
            try:
                good.append([float(x) for x in r])
            except ValueError:
                pass
    return np.array(good)


def load_core(d):
    a = load_num(os.path.join(d, "data2.csv"))
    y_c = a[:, 0::2] + 1j * a[:, 1::2]
    ir, ic = np.unravel_index(np.argmax(np.abs(y_c)), y_c.shape)
    yr = np.real(y_c * np.exp(-1j * np.angle(y_c[ir, ic])))
    ys = yr / (np.abs(yr).max() + 1e-12)
    taus = np.logspace(np.log10(500.0), np.log10(0.1), ys.shape[0])
    return ys, taus


def rock_frontend_map(ys, taus, refine_ema, device):
    """Per-dataset wide-grid SR SVD-ridge front-end + frozen v2rock RefineNet."""
    k1d = 1 - np.exp(-taus[:, None] / T1E[None, :])
    k2d = np.exp(-(np.arange(1, ys.shape[1] + 1) * TE)[:, None] / T2E[None, :])
    u1d, s1d, v1td = np.linalg.svd(k1d, full_matrices=False)
    u2d, s2d, v2td = np.linalg.svd(k2d, full_matrices=False)
    rr1 = int((s1d > s1d[0] * 1e-10).sum())
    rr2 = int((s2d > s2d[0] * 1e-10).sum())
    yc = u1d[:, :rr1].T @ ys @ u2d[:, :rr2]
    s = s1d[:rr1][:, None] * s2d[:rr2][None, :]
    smax = s1d[0] * s2d[0]
    c = s * yc / (s ** 2 + (LAM * smax) ** 2)
    yfit = u1d[:, :rr1] @ (s * c) @ u2d[:, :rr2].T
    relb = float(np.linalg.norm(ys - yfit) / (np.linalg.norm(ys) + 1e-12))
    mb = np.clip(v1td[:rr1].T @ c @ v2td[:rr2], 0, None)
    mb = mb / (mb.sum() * DL1 * DL2 + 1e-12)
    t_train = np.arange(1, NECH + 1) * TE
    t_data = np.arange(1, ys.shape[1] + 1) * TE
    g = np.zeros((ys.shape[0], NECH), dtype=np.float32)
    cov = t_train <= t_data[-1]
    for i in range(ys.shape[0]):
        g[i, cov] = np.interp(t_train[cov], t_data, ys[i])
    with torch.no_grad():
        base_log = torch.log1p(torch.tensor(mb[None], dtype=torch.float32, device=device) * 20.0)
        pooled = torch.tensor(log_pool_np(g[None]), dtype=torch.float32, device=device)
        pred_log = refine_ema(base_log, pooled)[0].cpu().numpy()
    ml = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
    return mb, ml, relb


def fast_frac(m):
    return float(m[np.log10(T1E) < np.log10(8.0), :].sum() / (m.sum() + 1e-12) * 100)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write evaluation JSON to working directory")
    args = parser.parse_args()

    device = torch.device("cpu")
    front = RockFrontEnd().to(device).eval()
    for p in front.parameters():
        p.requires_grad_(False)
    for seed in (1, 2, 3):
        train_seed(seed, front, device)

    refine_ema = RefineNet().to(device)
    ck = torch.load(os.path.join(HERE, "e9_v2rock_seed1.pth"), map_location=device, weights_only=True)
    refine_ema.load_state_dict(ck["refine"])
    refine_ema.eval()

    print("== E9 H1 Synthetic Wide-Grid SR Benchmark (n=200, SNR=5-100, CPU) ==", flush=True)
    h1 = eval_sim(front, refine_ema, device)
    print(" ", json.dumps(h1), flush=True)

    base = os.environ.get("NMR2_ROCK_DATA_DIR", "")
    if base and os.path.isdir(base):
        sand = {}
        sand_root = os.path.join(base, "sandstone") if os.path.isdir(os.path.join(base, "sandstone")) else base
        core_dirs = sorted(
            d for d in os.listdir(sand_root)
            if os.path.isfile(os.path.join(sand_root, d, "T1T2b", "data2.csv"))
        )
        for idx, raw_core in enumerate(core_dirs, start=1):
            core = f"sandstone_{idx:02d}"
            dpath = os.path.join(sand_root, raw_core, "T1T2b")
            ys, taus = load_core(dpath)
            mb, ml, relb = rock_frontend_map(ys, taus, refine_ema, device)
            r = float(np.corrcoef(ml.ravel(), mb.ravel())[0, 1])
            sand[core] = {
                "r_v2rock_vs_baseline": round(r, 4),
                "baseline_resid": round(relb, 4),
                "fast_frac_v2rock": round(fast_frac(ml), 1),
                "fast_frac_baseline": round(fast_frac(mb), 1),
            }
        out = {"hypotheses": {"H1": h1, "H2_sandstone": sand}}
        if args.write:
            with open(os.path.join(HERE, "e9_rock_retrain.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    else:
        arch = json.load(open(os.path.join(HERE, "results_archive", "e9_rock_retrain.json"), encoding="utf-8"))
        print("== External 2.0 MHz data not mounted (NMR2_ROCK_DATA_DIR unset); showing archived cohort summaries ==", flush=True)
        print("  H2_sandstone_summary:", json.dumps(arch["hypotheses"]["H2_sandstone_summary"]), flush=True)
        print("  H4_coal_summary:", json.dumps(arch["hypotheses"]["H4_coal_summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
