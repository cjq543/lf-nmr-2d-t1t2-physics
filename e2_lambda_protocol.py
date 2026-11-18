# -*- coding: utf-8 -*-
"""
E2: 2D λ 协议（承接 1D 纪律：dev 选定→冻结→test 报告）
dev 100 样本选 λ → test 300 新鲜样本出分。附敏感性曲线数据。
"""
import io, sys, os, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim2d import make_sample
from baseline2d import reconstruct_2d, metrics

LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def pearson(M, spec):
    p = M / (M.sum() * 2.5e-3 + 1e-12)  # DL1*DL2≈2.5e-3，归一系数不影响相关
    t = spec / (spec.sum() * 2.5e-3 + 1e-12)
    return float(np.corrcoef(p.ravel(), t.ravel())[0, 1])


def main():
    rng = np.random.default_rng(20261113)
    # dev
    dev = [make_sample(rng, snr=float(rng.uniform(30, 300))) for _ in range(100)]
    dev_scores = []
    for lam in LAMBDAS:
        rs = [pearson(reconstruct_2d(s["y"], s["mask"], lam=lam), s["spec"]) for s in dev]
        dev_scores.append(float(np.mean(rs)))
        print(f"  dev λ={lam:g}: r={dev_scores[-1]:.4f}", flush=True)
    best_i = int(np.argmax(dev_scores))
    lam_frozen = LAMBDAS[best_i]
    # test（新鲜种子）
    test = [make_sample(rng, snr=float(rng.uniform(30, 300))) for _ in range(300)]
    test_scores = {}
    for lam in LAMBDAS:
        rs = [pearson(reconstruct_2d(s["y"], s["mask"], lam=lam), s["spec"]) for s in test]
        test_scores[lam] = float(np.mean(rs))
    out = {"lambdas": LAMBDAS, "dev_scores": dev_scores, "frozen_lambda": lam_frozen,
           "test_scores": test_scores, "test_at_frozen": test_scores[lam_frozen]}
    print(f"冻结 λ={lam_frozen}; test 各 λ: "
          f"{ {f'{k:g}': round(v,4) for k,v in test_scores.items()} }", flush=True)
    with open(os.path.join(HERE, "e2_lambda_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("e2_lambda_protocol.json 已存", flush=True)


if __name__ == "__main__":
    main()
