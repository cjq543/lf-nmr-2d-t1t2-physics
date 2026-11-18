# -*- coding: utf-8 -*-
"""Table 1 补列: 合成测试集 (n=200, seed_off=100) 上 baseline/v2 的 T1 峰位误差 (dex)
完全复刻 train2d_v2.eval_sim 的数据流, 峰位在线性域边缘谱 argmax。"""
import io, sys, os, json
import numpy as np
import torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train2d_v2 import RefineNet, PhysicsFrontEnd, log_pool_np, _T1, _T2, DL1, DL2
from sim2d import make_sample
from baseline2d import reconstruct_2d

DEVICE = torch.device("cpu")
T1G, T2G = np.asarray(_T1), np.asarray(_T2)
lt1, lt2 = np.log10(T1G), np.log10(T2G)
ck = torch.load(os.path.join(HERE, "v2_refine.pth"), map_location=DEVICE, weights_only=True)
refine = RefineNet(); refine.load_state_dict(ck["refine"]); refine.eval()
front = PhysicsFrontEnd(); front.eval()

def peak_err(M, spec):
    p1, p2 = M.sum(1), M.sum(0)
    t1, t2 = spec.sum(1), spec.sum(0)
    e1 = abs(lt1[int(np.argmax(p1))] - lt1[int(np.argmax(t1))])
    e2 = abs(lt2[int(np.argmax(p2))] - lt2[int(np.argmax(t2))])
    return e1, e2

rng_e = np.random.default_rng(20261112 + 100)
rows = {"baseline": [], "v2": []}
r_l, r_b = [], []
for i in range(200):
    s = make_sample(rng_e)
    ys = np.real(s["y"]) / (np.abs(np.real(s["y"])).max() + 1e-9)
    yt = torch.tensor(ys[None], dtype=torch.float32)
    with torch.no_grad():
        base = front(yt)[0]
        pooled = torch.tensor(log_pool_np(ys[None]), dtype=torch.float32)
        pred_log = refine(base[None], pooled)[0].numpy()
    pred = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
    bnp = reconstruct_2d(s["y"], s["mask"], lam=1e-3)
    for name, M in [("v2", pred), ("baseline", bnp)]:
        e1, e2 = peak_err(M, s["spec"])
        rows[name].append((e1, e2))
    pl = pred / (pred.sum() * DL1 * DL2 + 1e-12)
    tb = bnp / (bnp.sum() * DL1 * DL2 + 1e-12)
    tt = s["spec"] / (s["spec"].sum() * DL1 * DL2 + 1e-12)
    r_l.append(float(np.corrcoef(pl.ravel(), tt.ravel())[0, 1]))
    r_b.append(float(np.corrcoef(tb.ravel(), tt.ravel())[0, 1]))

out = {}
for k, v in rows.items():
    a = np.array(v)
    out[k] = {"t1_peak_err_dex_mean": round(float(a[:, 0].mean()), 4),
              "t1_peak_err_dex_median": round(float(np.median(a[:, 0])), 4),
              "t2_peak_err_dex_mean": round(float(a[:, 1].mean()), 4),
              "t2_peak_err_dex_median": round(float(np.median(a[:, 1])), 4)}
out["pearson_check"] = {"v2": round(float(np.mean(r_l)), 4), "baseline": round(float(np.mean(r_b)), 4)}
print(json.dumps(out, indent=2))
with open(os.path.join(HERE, "table1_peak_err.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("table1_peak_err.json 已存")
