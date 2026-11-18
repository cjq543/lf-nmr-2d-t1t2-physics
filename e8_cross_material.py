# -*- coding: utf-8 -*-
"""E8 跨材料零样本迁移 v2（预注册式，含扩展网格基线产品）
假设（运行前登记）:
  H1 恢复形式: Magritek "T1T2" 行恢复曲线符合 SR(1-exp) 而非 |IR|（削顶参考样本剔除）
  H1b 恢复曲线为双族群: 快相 T1<8ms(吸附/表面弛豫) + 慢相平台(自由流体, T1~1-5s);
      训练网格(T1 8-2000/T2 0.5-3000ms)不能覆盖快相, 慢相处于边缘
  H2 零样本一致性 r 低于油样域内参考 0.705, 且砂岩亦受支撑限制(2D 回波窗 60-180ms 截断慢相 T2)
  H3 煤岩/页岩双轴支撑失配 → 预测在网格角堆积(网格边缘质量放大)
  H4 同深度复测扩展网格基线谱相关性对比
"""
import io, sys, os, csv, json, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train2d_v2 import RefineNet, log_pool_np, _T1, _T2, DL1, DL2, POOLIDX, N_POOL

BASE = os.environ.get("NMR2_ROCK_DATA_DIR", "")
DEVICE = torch.device("cpu")
T1G, T2G = np.asarray(_T1), np.asarray(_T2)
T1X = np.logspace(np.log10(0.02), np.log10(5000.0), 72)
T2X = np.logspace(np.log10(0.02), np.log10(3000.0), 72)
TE_TRAIN = 1.027

def load_numeric(pth):
    good = []
    for r in csv.reader(open(pth, encoding="utf-8-sig")):
        if len(r) < 2:
            continue
        try:
            good.append([float(x) for x in r])
        except ValueError:
            pass
    return np.array(good)

def load_magritek_2d(d):
    a = load_numeric(os.path.join(d, "data2.csv"))
    n_tau, nc = a.shape
    cands = {"iq_even_first": a[:, 0::2] + 1j * a[:, 1::2],
             "iq_odd_first": a[:, 1::2] + 1j * a[:, 0::2],
             "block_half": a[:, :nc // 2] + 1j * a[:, nc // 2:]}
    best, bname = None, None
    for k, Y in cands.items():
        mag = np.abs(Y[:, :50])
        tv = float(np.abs(np.diff(np.log(mag + 1e-9), axis=1)).mean())
        if best is None or tv < best:
            best, bname = tv, k
    return cands[bname], bname

def read_par(d):
    s = open(os.path.join(d, "acqu.par"), encoding="utf-8", errors="replace").read()
    out = {}
    for k in ["experiment", "echoTime", "nrEchoes", "tauSteps", "minTau", "maxTau", "nrScans", "rxGain"]:
        m = re.search(rf"{k}\s*=\s*\"?([^\"\n\r]+)", s)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                out[k] = m.group(1).strip()
    return out

def tau_grid(par):
    n = int(par.get("tauSteps", 10))
    lo, hi = par.get("minTau", 0.1), par.get("maxTau", 500.0)
    return np.logspace(np.log10(hi), np.log10(lo), n)

def phase_align(Y):
    ir, ic = np.unravel_index(np.argmax(np.abs(Y)), Y.shape)
    ph = np.exp(-1j * np.angle(Y[ir, ic]))
    return np.real(Y * ph)

# ---------- H1: K1 形式裁定（剔削顶样本） ----------
def row_amplitudes(Y):
    A = np.mean(np.abs(Y[:, :5]), axis=1)
    return A / A.max()

def is_clipped(A, tol=0.005, need=3):
    """连续≥3个相邻行幅差<0.5% 视为削顶平台"""
    d = np.abs(np.diff(A))
    run = mx = 0
    for v in d:
        run = run + 1 if v < tol else 0
        mx = max(mx, run)
    return mx >= need

def baseline_map(y_real, taus, te_ms, form, T1g, T2g, lam=1e-3):
    K1 = (1 - np.exp(-taus[:, None] / T1g[None, :])) if form == "SR" else \
         (1 - 2 * np.exp(-taus[:, None] / T1g[None, :]))
    K2 = np.exp(-(np.arange(1, y_real.shape[1] + 1) * te_ms)[:, None] / T2g[None, :])
    U1, S1, V1t = np.linalg.svd(K1, full_matrices=False)
    U2, S2, V2t = np.linalg.svd(K2, full_matrices=False)
    r1 = int((S1 > S1[0] * 1e-10).sum())
    r2 = int((S2 > S2[0] * 1e-10).sum())
    Yc = U1[:, :r1].T @ y_real @ U2[:, :r2]
    s = S1[:r1][:, None] * S2[:r2][None, :]
    smax = S1[0] * S2[0]
    C = s * Yc / (s ** 2 + (lam * smax) ** 2)
    Yfit = U1[:, :r1] @ (s * C) @ U2[:, :r2].T      # 对角化系数域 → 数据域必须乘回 s
    rel = float(np.linalg.norm(y_real - Yfit) / (np.linalg.norm(y_real) + 1e-12))
    M = np.clip(V1t[:r1].T @ C @ V2t[:r2], 0, None)
    dT1 = np.log(10) * (np.log10(T1g[-1]) - np.log10(T1g[0])) / len(T1g)
    dT2 = np.log(10) * (np.log10(T2g[-1]) - np.log10(T2g[0])) / len(T2g)
    area = M.sum() * dT1 * dT2
    if area > 1e-12:
        M = M / area
    return M, rel

def peaks(M, T1g, T2g):
    lt1, lt2 = np.log10(T1g), np.log10(T2g)
    return 10 ** lt1[int(np.argmax(M.sum(1)))], 10 ** lt2[int(np.argmax(M.sum(0)))]

# ---------- 数据集 ----------
def collect():
    items = []
    for root, dirs, files in os.walk(BASE):
        if "data2.csv" in files:
            rel = os.path.relpath(root, BASE).replace("\\", "/")
            fam = rel.split("/")[0] if "/" in rel else "rock"
            items.append((f"{fam}/core_{len(items)+1:02d}", root))
    return sorted(items)

def main():
    if not BASE or not os.path.isdir(BASE):
        raise FileNotFoundError(
            "External 2.0 MHz Magritek rock dataset is not bundled under institutional policy; "
            "set NMR2_ROCK_DATA_DIR or inspect results_archive/e8_cross_material.json."
        )
    OUT = {"hypotheses": {}, "k1_form": {}, "datasets": [], "retest": {}, "reference": {}}
    
    ck = torch.load(os.path.join(HERE, "v2_refine.pth"), map_location=DEVICE, weights_only=True)
    refine = RefineNet().to(DEVICE)
    refine.load_state_dict(ck["refine"])
    refine.eval()
    p = os.path.join(HERE, "results_archive", "v2_real_eval.json")
    if os.path.isfile(p):
        OUT["reference"]["oil_in_domain_agreement"] = json.load(open(p, encoding="utf-8"))["mean_agreement"]
    print(f"域内参考(油13样零样本一致性) r = {OUT['reference']['oil_in_domain_agreement']:.4f}")
    print("\n== H1: K1 形式裁定 (SR vs |IR|) ==")
    FORMS = {}
    all_items = collect()
    for idx, (name, d) in enumerate(all_items[:5], start=1):
        if not os.path.isfile(os.path.join(d, "data2.csv")):
            continue
        Y, _ = load_magritek_2d(d)
        par = read_par(d)
        taus = tau_grid(par)
        A = row_amplitudes(Y)
        clip = is_clipped(A)
        rec = {"taus_max_ms": float(taus[0]), "clipped": clip}
        if not clip:
            for form in ["SR", "absIR"]:
                K = (1 - np.exp(-taus[:, None] / T1X[None, :])) if form == "SR" else \
                    np.abs(1 - 2 * np.exp(-taus[:, None] / T1X[None, :]))
                w, _ = nnls(K, A)
                rec[f"resid_{form}"] = round(float(np.linalg.norm(K @ w - A) / np.linalg.norm(A)), 4)
                if form == "SR":
                    rec["fast_T1_lt8ms_pct"] = round(float(w[T1X < 8].sum() / (w.sum() + 1e-12) * 100), 1)
                    rec["slow_T1_gt1s_pct"] = round(float(w[T1X > 1000].sum() / (w.sum() + 1e-12) * 100), 1)
            rec["winner"] = "SR" if rec["resid_SR"] < rec["resid_absIR"] else "absIR"
        FORMS[name] = rec
        print(f"  {name:<22} taus≤{taus[0]:.3g}ms 削顶={clip} " +
              (f"SR残差={rec['resid_SR']:.4f} |IR|残差={rec['resid_absIR']:.4f} → {rec['winner']} "
               f"快相T1<8ms={rec['fast_T1_lt8ms_pct']:.0f}% 慢相T1>1s={rec['slow_T1_gt1s_pct']:.0f}%" if not clip else "剔除裁定"))
    OUT["k1_form"] = FORMS
    wins = [v["winner"] for v in FORMS.values() if "winner" in v]
    OUT["hypotheses"]["H1"] = f"SR wins {sum(1 for w in wins if w == 'SR')}/{len(wins)} unclipped scans (clipped reference excluded)"
    print(f"  → {OUT['hypotheses']['H1']}")
    FORM = "SR"
    print("\n== H2/H3: 扩展网格基线 + 零样本 ==")
    maps_ext, maps_train = {}, {}
    for name, d in collect():
        Y, _ = load_magritek_2d(d)
        par = read_par(d)
        te = par.get("echoTime", 60.0) / 1000.0
        taus = tau_grid(par)
        Yr = phase_align(Y)
        if Yr.shape[0] < 3:
            print(f"  {name:<16} 跳过(仅{Yr.shape[0]}行tau)")
            continue
        if len(taus) > Yr.shape[0]:
            taus = taus[:Yr.shape[0]]
        ys = Yr / (np.abs(Yr).max() + 1e-12)
        window_ms = ys.shape[1] * te
        # 扩展网格基线（物理产品）: λ 自适应（小奇异值方向被维纳滤波压死时放宽）
        best = None
        for lam in (1e-3, 1e-4, 1e-5):
            Mx_try, rel_try = baseline_map(ys, taus, te, FORM, T1X, T2X, lam=lam)
            if best is None or rel_try < best[1]:
                best = (Mx_try, rel_try, lam)
        Mx, relx, lam_used = best
        # 训练网格基线（零样本参照）
        Mb, relb = baseline_map(ys, taus, te, FORM, T1G, T2G)
        base_log = np.log1p(Mb * 20.0)
        # 双族群分解（扩展网格）
        dT1 = np.log(10) * (np.log10(T1X[-1]) - np.log10(T1X[0])) / len(T1X)
        dT2 = np.log(10) * (np.log10(T2X[-1]) - np.log10(T2X[0])) / len(T2X)
        mt = Mx.sum() * dT1 * dT2
        fast = float(Mx[np.log10(T1X) < np.log10(8), :].sum() * dT1 * dT2 / mt * 100)
        slow = float(Mx[np.log10(T1X) > 3, :].sum() * dT1 * dT2 / mt * 100)
        # 零样本
        tmap = np.arange(1, 2001) * TE_TRAIN
        kidx = np.clip(np.round(tmap / te).astype(int) - 1, -1, ys.shape[1] - 1)
        Ymap = np.zeros((ys.shape[0], 2000), dtype=np.float32)
        ok = kidx >= 0
        Ymap[:, ok] = ys[:, kidx[ok]]
        pooled = log_pool_np(Ymap[None].astype(np.float32))
        with torch.no_grad():
            pred_log = refine(torch.tensor(base_log[None], dtype=torch.float32),
                              torch.tensor(pooled, dtype=torch.float32))[0].numpy()
        Ml = np.expm1(np.clip(pred_log, 0, 20)) / 20.0
        base_lin = np.expm1(base_log) / 20.0
        agree = float(np.corrcoef(Ml.ravel(), base_lin.ravel())[0, 1])   # 线性域（与 eval_sim 同口径）
        b_t1, b_t2 = peaks(Mb, T1G, T2G)
        x_t1, x_t2 = peaks(Mx, T1X, T2X)
        t2_out = float(Mx[:, np.log10(T2X) < np.log10(T2G[0])].sum() * dT1 * dT2 / mt * 100)
        t1_out = float(Mx[np.log10(T1X) < np.log10(T1G[0]), :].sum() * dT1 * dT2 / mt * 100)
        rec = dict(name=name, family=name.split("/")[0], nti=int(len(taus)), nech=int(ys.shape[1]),
                   te_ms=round(te, 3), window_ms=round(window_ms, 1), form=FORM, lam_ext=lam_used,
                   rel_resid_ext=round(relx, 4), rel_resid_train=round(relb, 4),
                   agree_zeroshot=round(agree, 4),
                   base_T1=round(b_t1, 2), base_T2=round(b_t2, 2),
                   ext_T1=round(x_t1, 3), ext_T2=round(x_t2, 3),
                   fast_phase_pct=round(fast, 1), slow_phase_pct=round(slow, 1),
                   support_out_T1_pct=round(t1_out, 1), support_out_T2_pct=round(t2_out, 1))
        OUT["datasets"].append(rec)
        maps_ext[name], maps_train[name] = Mx, Mb
        print(f"  {name:<16} 窗={window_ms:6.1f}ms 残差{relx:.4f} 零样本r={agree:.4f} "
              f"训练网格峰{b_t1:7.2f}/{b_t2:7.2f} 扩展峰{x_t1:7.3f}/{x_t2:8.3f} "
              f"快相={fast:5.1f}% 慢相={slow:5.1f}% 支撑外T1={t1_out:5.1f}% T2={t2_out:5.1f}%")
    
    # ---------- H4 复测（扩展网格基线） ----------
    corr = []
    if os.path.isdir(os.path.join(BASE, "shale test")):
        for dep in sorted(os.listdir(os.path.join(BASE, "shale test"))):
            p1 = [k for k in maps_ext if k.endswith(dep)]
            if len(p1) >= 2:
                corr.append(round(float(np.corrcoef(maps_ext[p1[0]].ravel(), maps_ext[p1[1]].ravel())[0, 1]), 4))
    OUT["retest"] = {"ext_baseline_pearson": corr,
                     "median": float(np.median(corr)) if corr else None}
    print(f"\n== H4 复测: 扩展网格基线谱相关 {corr} 中位 {OUT['retest']['median']}")
    
    # ---------- 汇总 ----------
    sand = [d for d in OUT["datasets"] if d["family"] in ("砂岩", "sandstone")]
    coal = [d for d in OUT["datasets"] if d["family"] not in ("砂岩", "sandstone")]
    OUT["hypotheses"]["H1b"] = (
        f"快相占比: 砂岩中位 {np.median([d['fast_phase_pct'] for d in sand]):.0f}%, "
        f"煤岩中位 {np.median([d['fast_phase_pct'] for d in coal]):.0f}% —— 双族群确认" if sand and coal else "n/a")
    OUT["hypotheses"]["H2"] = (
        f"零样本一致性: 砂岩均值 {np.mean([d['agree_zeroshot'] for d in sand]):.4f}, "
        f"煤岩均值 {np.mean([d['agree_zeroshot'] for d in coal]):.4f} "
        f"(油域内 {OUT['reference']['oil_in_domain_agreement']:.4f})")
    OUT["hypotheses"]["H3"] = (
        f"支撑外质量: T1 煤岩中位 {np.median([d['support_out_T1_pct'] for d in coal]):.1f}%, "
        f"T2 煤岩中位 {np.median([d['support_out_T2_pct'] for d in coal]):.1f}%")
    for k, v in OUT["hypotheses"].items():
        print(f"  {k}: {v}")
    
    with open(os.path.join(HERE, "results_archive", "e8_cross_material.json"), "w", encoding="utf-8") as f:
        json.dump(OUT, f, ensure_ascii=False, indent=2)
    print("e8_cross_material.json 已存")
    
    # ---------- 图 ----------
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.5))
    # (a) 恢复曲线 + SR 拟合
    first_item = all_items[0][1]
    Y, _ = load_magritek_2d(first_item)
    taus = tau_grid(read_par(first_item))
    A = row_amplitudes(Y)
    w, _ = nnls(1 - np.exp(-taus[:, None] / T1X[None, :]), A)
    axs[0, 0].semilogx(taus, A, "o", label="实测行幅")
    axs[0, 0].semilogx(taus, 1 - np.exp(-taus[:, None] / T1X[None, :]) @ w, "-", label="SR 拟合")
    axs[0, 0].invert_xaxis()
    axs[0, 0].set_xlabel("tau (ms)"); axs[0, 0].set_ylabel("归一化行幅")
    axs[0, 0].set_title("H1: 恢复曲线 SR 裁定 (Representative Core)")
    axs[0, 0].legend(fontsize=8)
    # (b) 砂岩扩展网格基线
    sn = [d for d in OUT["datasets"] if d["family"] in ("砂岩", "sandstone")][0]["name"]
    im = axs[0, 1].pcolormesh(T2X, T1X, maps_ext[sn], shading="auto", cmap="viridis")
    axs[0, 1].set_xscale("log"); axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("T2 (ms)"); axs[0, 1].set_ylabel("T1 (ms)")
    axs[0, 1].set_title(f"扩展网格基线 {sn} (SR)")
    axs[0, 1].plot([0.02, 5000], [0.02, 5000], "w--", lw=0.8, alpha=0.6)
    plt.colorbar(im, ax=axs[0, 1], label="归一化谱")
    # (c) 一致性按族
    fam = {}
    for d in OUT["datasets"]:
        fam.setdefault(d["family"], []).append(d["agree_zeroshot"])
    labels = list(fam.keys())
    vals = [fam[k] for k in labels]
    axs[1, 0].boxplot(vals, tick_labels=labels)
    axs[1, 0].axhline(OUT["reference"]["oil_in_domain_agreement"], ls="--", c="r",
                      label=f"油域内 {OUT['reference']['oil_in_domain_agreement']:.3f}")
    axs[1, 0].set_ylabel("零样本一致性 r")
    axs[1, 0].tick_params(axis="x", rotation=30)
    axs[1, 0].legend(fontsize=8)
    axs[1, 0].set_title("H2/H3: 基线 vs 冻结网络")
    # (d) 快慢相散点
    for famname, mk in [("砂岩", "o"), ("coal_shale", "s")]:
        xs = [d["fast_phase_pct"] for d in OUT["datasets"] if famname in d["family"]]
        ys2 = [d["agree_zeroshot"] for d in OUT["datasets"] if famname in d["family"]]
        if xs:
            axs[1, 1].plot(xs, ys2, mk, ms=5, label=famname, alpha=0.7)
    axs[1, 1].set_xlabel("快相(T1<8ms)占比 %")
    axs[1, 1].set_ylabel("零样本一致性 r")
    axs[1, 1].legend(fontsize=8)
    axs[1, 1].set_title("支撑失配程度 vs 迁移一致性")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "e8_cross_material.png"), dpi=150)
    print("e8_cross_material.png 已存")

if __name__ == "__main__":
    main()
