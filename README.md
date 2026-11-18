# ReproBundle for Paper #2 (*Measurement Science and Technology*)

**Manuscript**: *Physics-Front-End Deep Learning for 2D $T_1$–$T_2$ Relaxation Correlation Spectroscopy on a Compact Low-Field NMR Analyzer*  
**Authors**: Jie Huang, Hao Wei, Di Wang, Caidie Qian, Dingwen Zhou, Can Liang, Zijian Jia*  
**Institutions**: School of Health Science and Engineering, University of Shanghai for Science and Technology, Shanghai 200093, China; Changzhou Institute of Technology, Changzhou, Jiangsu 213032, China  
**Correspondence**: `jiazijian1989@usst.edu.cn`  
**Repository**: https://github.com/cjq543/lf-nmr-2d-t1t2-physics

---

## Overview

This standalone reproduction bundle provides the deterministic 23 MHz IR-CPMG forward simulator, untouched frozen physics-front-end CNN weights (`v2_refine.pth`, `seed3`), neural and classical 2D inverse Laplace solvers, synthetic ablation benchmarks, and sanitized derived JSON metric/feature archives supporting the *Measurement Science and Technology* (*MST*) manuscript.

- **Model footprint & numerical implementation**: Frozen zero-parameter SVD-ridge physics front-end (`PhysicsFrontEnd`, $\lambda = 10^{-3}$) + compact 4-layer `softplus` CNN refiner (`RefineNet`, **19,393** trainable parameters; direct non-skip log-spectral refinement rather than an additive skip branch) vs. **2,663,777** (`Inv2DNet`) and **73,548,416** (`SimpleViTGrid`, DLEMLR architecture). In `train2d_v2.py`, `PhysicsFrontEnd` casts `K1, K2` to `float32` prior to `torch.linalg.svd`, retaining all `20×96` economy modes (whereas `float64` `1e-10` retains `13×34 = 442` separable / `315` joint modes; verified in `results_archive/frontend_precision_sensitivity.json` within $\le 3.05\times 10^{-4}$ relative $L_2$).
- **Development-exposure disclosure**: Neural weights (`v2_refine.pth`) were trained solely on synthetic IR-CPMG signals (`sim2d.py`); the 13 real 23 MHz liquid acquisitions informed prior support bounds and `v1->v2` architecture diagnostics and therefore constitute a **retrospective** real-data transfer cohort rather than a blind external cohort.
- **Verified offline CPU benchmark (`n = 200`, `rng = 20261112 + 100`)**: `v2` achieves `r = 0.9648 ± 0.0324` (`seed3`, matching the 3-seed mean `0.9648`), median `(T1, T2)` peak errors `(0.0000, 0.0000)` dex, single-thread CPU latency ~4 ms/sample, outperforming fixed SVD-ridge (`r = 0.8942`, `198/200` paired wins).

---

## Quick Verification (CPU-Only, Offline)

Install dependencies (`python >= 3.10` recommended):

```bash
pip install -r requirements.txt
```

Run the single-thread CPU verification and 200-sample synthetic benchmark:

```bash
python -B eval_synthetic_benchmark.py
```

Run the full integrity, checksum (`MANIFEST_SHA256.csv`), module-import, and privacy/leak audit:

```bash
python -B verify_reprobundle.py
```

---

## Bundle Contents

### Core Physics, Neural Models & Evaluation
- `sim2d.py` — 23 MHz IR-CPMG forward operator (`TI`: 20 points `0.058–1000.001 ms`, `2000` echoes, `TE = 1.027 ms`, `48 × 96` spectral grid) and deterministic `make_sample()`.
- `train2d_v2.py` — `PhysicsFrontEnd(lam=1e-3)` + `RefineNet` (`19,393` params) training/inference implementation.
- `v2_refine.pth` — Untouched frozen PyTorch checkpoint (`80,768` bytes, `MD5: 804d0c5a3d1186f8782102c19398b209`).
- `train2d.py` — Pure data-driven CNN control (`Inv2DNet`, `2,663,777` params).
- `dlemlr_model.py` — 23 MHz physical-kernel-channel `SimpleViTGrid` baseline (`73,548,416` params).
- `eval_synthetic_benchmark.py` — Self-contained offline verification of frozen weights, parameter counts, held-out 200-sample `v2` vs. SVD-ridge parity, and classical solver spot-checks.
- `verify_reprobundle.py` — Automated integrity, SHA-256 manifest, leak/exclusion, and runtime suite checker.

### Classical Solvers & Ablation / Hardening Scripts
- `baseline2d.py` — Fixed closed-form Kronecker SVD-ridge (`lam=1e-3`).
- `adaptive_lambda_baseline.py` — Kronecker SVD-ridge GCV, fixed Song (2002) BRD-NNLS (`lam=1e-3`), and adaptive BRD-GCV.
- `sparse_l1_baseline.py` — Non-negative FISTA-$\ell_1$ and Guo (2019)-style two-stage $\ell_1$+NNLS.
- `e1_sign_ablation.py`, `e2_lambda_protocol.py`, `e2b_peak_err_table1.py`, `e3_subsampling.py`, `e6_snr_robustness.py`, `e7_component_robustness.py` — Deterministic synthetic ablation & robustness protocols.
- `stats_and_1d2d.py`, `run_hardening_benchmarks.py`, `label_sensitivity.py` — SVD effective-rank audit, $9!$ exact permutation tests (plus whole-pipeline $9!$ nested-LOO & $2^9$ paired 1D/2D residual audit in `results_archive/nested_loo_exact_permutation.json`), Holm-Bonferroni correction, nested LOO regression, H1/H2/H3/H4 hardening framework, and Monte Carlo `N=4,000` Gaussian label-noise sensitivity (`label_sensitivity.py` executes completely offline on `results_archive/`; raw 23 MHz acquisitions required only to re-derive raw tables).
- `e8_cross_material.py`, `e8_sr_alignment.py`, `e9_rock_retrain.py` — Cross-instrument 2.0 MHz porous-rock boundary, SR-alignment negative control, and dedicated wide-grid SR rock retraining (`e9_v2rock_seed{1,2,3}.pth`; synthetic wide-grid `H1` benchmark executes offline, while raw `H2/H4` core re-evaluation uses external `NMR2_ROCK_DATA_DIR`, and `results_archive/` stores anonymized cohort-level summary statistics without per-record external core/wood identifiers or raw filenames).

### Sanitized Derived Archives (`results_archive/`)
- `v2_summary.json`, `table1_peak_err.json`, `e1_sign_ablation.json`, `e2_lambda_protocol.json`, `e3_subsampling.json`, `e6_snr_robustness.json`, `e7_component_robustness.json`
- `v2_real_eval.json`, `e4_fingerprint.json`, `all_samples_features.json`, `adaptive_baseline.json`, `sparse_l1_baseline.json`, `frontend_precision_sensitivity.json`
- `stats_and_1d2d.json`, `hardening_results.json`, `nested_loo_exact_permutation.json`, `real_uncertainty_and_standards.json`, `label_sensitivity.json`, `label_sensitivity_v2_stats.json`, `dlemlr_results.json`, `dlemlr_fair_audit.json`
- `vendor_xls_audit.json`, `e8_cross_material.json`, `e8_sr_alignment.json`, `e9_rock_retrain.json`

---

## Acknowledgments, Funding & Companion Reference

The authors acknowledge **Le Liu** for 23 MHz experimental NMR acquisitions, **University of Shanghai for Science and Technology** for 23 MHz instrument and compute access, and co-author **Can Liang** (**Changzhou Institute of Technology**, `常州工学院`) for access to the 2.0 MHz Magritek low-field NMR platform and porous-rock core samples. This work was supported by the Changzhou Science and Technology Project (`CJ20253073`). For companion 1D edge-denoising work (`Ref. [11]`), see our accompanying *Journal of Magnetic Resonance* manuscript. See `DATA_POLICY_AND_PROVENANCE.md` for full data governance terms.
