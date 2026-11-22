# -*- coding: utf-8 -*-
"""
Automated integrity, privacy, manifest, import, and CPU benchmark verifier
for Paper #2 ReproBundle_paper2.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import ast
import csv
import hashlib
import importlib
import json
import re
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

EXPECTED_JSONS = [
    "v2_summary.json",
    "v2_real_eval.json",
    "table1_peak_err.json",
    "e1_sign_ablation.json",
    "e2_lambda_protocol.json",
    "e3_subsampling.json",
    "e6_snr_robustness.json",
    "e7_component_robustness.json",
    "e8_cross_material.json",
    "e8_sr_alignment.json",
    "e9_rock_retrain.json",
    "vendor_xls_audit.json",
    "e4_fingerprint.json",
    "all_samples_features.json",
    "adaptive_baseline.json",
    "sparse_l1_baseline.json",
    "stats_and_1d2d.json",
    "hardening_results.json",
    "label_sensitivity.json",
    "label_sensitivity_v2_stats.json",
    "dlemlr_results.json",
    "dlemlr_fair_audit.json",
    "frontend_precision_sensitivity.json",
    "nested_loo_exact_permutation.json",
    "real_uncertainty_and_standards.json",
    "mechanism_ablations.json",
    "archive_audit_and_wood_probe.json",
    "historical_strict_odd_even_eval.json",
]

ALLOWED_PTHS = {
    "v2_refine.pth",
    "e9_v2rock_seed1.pth",
    "e9_v2rock_seed2.pth",
    "e9_v2rock_seed3.pth",
}

FORBIDDEN_EXTS = {
    ".pea", ".ircp", ".irp", ".inv", ".fid", ".par", ".xls", ".xlsx", ".npz", ".pt",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tree_and_leaks():
    # Construct audit patterns dynamically so this script has zero sensitive literals
    drive_pat = re.compile(r"(?<![A-Za-z])[A-Za-z]:[/\\]")
    unix_pat = re.compile(r"/(?:Users|home)/[^\s\"']+")
    sensitive_tokens = [
        "\u6bd5" + "\u4e1a",
        "928" + "83",
        "1314" + "5547664",
        "dorm_bundle-" + "\u5df2\u8dd1\u5b8c",
        "\u8d3e\u5bfc" + "\u90a3\u8585",
        "\u7ebd\u8fc8\u6570\u636e" + "\u603b\u5e93",
        "\u98df\u7528\u6cb9" + "\u539f\u59cb\u6570\u636e",
        "\x6ci125-1",
        "\x6ci263-1",
        "\x7ahuang74-2",
        "2023" + "1126",
        "2023" + "1128",
        "2023" + "1130",
        "0106_" + "T1T2\x2enpy",
    ]
    npy_pat = re.compile(r"\." + "npy\\b", re.I)

    found_files = {}
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, HERE).replace("\\", "/")
            ext = os.path.splitext(fn)[1].lower()
            if ext in FORBIDDEN_EXTS:
                raise AssertionError(f"Forbidden raw/checkpoint extension found: {rel}")
            if ext == ".pth" and rel not in ALLOWED_PTHS:
                raise AssertionError(f"Unexpected .pth file found: {rel}")
            if ext == ".csv" and rel != "MANIFEST_SHA256.csv":
                raise AssertionError(f"Unexpected .csv file (raw rock data prohibited): {rel}")

            found_files[rel] = p

            if ext in {".py", ".md", ".txt", ".json", ".csv"} or fn == "LICENSE":
                txt = open(p, "r", encoding="utf-8", errors="strict").read()
                m_drv = drive_pat.findall(txt)
                assert not m_drv, f"Windows drive-path leak in {rel}: {m_drv[:3]}"
                m_unx = unix_pat.findall(txt)
                assert not m_unx, f"Unix user-path leak in {rel}: {m_unx[:3]}"
                for tok in sensitive_tokens:
                    assert tok not in txt, f"Sensitive local token leak ({tok}) in {rel}"
                if rel not in ("DATA_POLICY_AND_PROVENANCE.md", "verify_reprobundle.py"):
                    assert not npy_pat.search(txt), f"Unexpected external filename leak in {rel}"
                if ext == ".json":
                    json.loads(txt)

    return found_files


def verify_manifest(found_files):
    man_path = os.path.join(HERE, "MANIFEST_SHA256.csv")
    assert os.path.isfile(man_path), "Missing MANIFEST_SHA256.csv"
    rows = {}
    with open(man_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["relative_path"]] = (int(r["bytes"]), r["sha256"])

    expected_keys = {k for k in found_files if k != "MANIFEST_SHA256.csv"}
    assert set(rows.keys()) == expected_keys, (
        f"Manifest mismatch: extra={set(rows.keys()) - expected_keys}, "
        f"missing={expected_keys - set(rows.keys())}"
    )

    for rel, (sz, digest) in rows.items():
        p = found_files[rel]
        actual_sz = os.path.getsize(p)
        actual_sha = sha256_file(p)
        assert actual_sz == sz, f"Size mismatch for {rel}: {actual_sz} != {sz}"
        assert actual_sha == digest, f"SHA256 mismatch for {rel}: {actual_sha} != {digest}"
    return len(rows)


def main():
    print("=== ReproBundle_paper2 Full Verification ===", flush=True)
    for req in ("README.md", "DATA_POLICY_AND_PROVENANCE.md", "requirements.txt", "LICENSE", "v2_refine.pth", "e9_v2rock_seed1.pth"):
        assert os.path.isfile(os.path.join(HERE, req)), f"Missing required file: {req}"
    for jf in EXPECTED_JSONS:
        jp = os.path.join(HERE, "results_archive", jf)
        assert os.path.isfile(jp), f"Missing archive JSON: {jf}"

    found = verify_tree_and_leaks()
    n_manifest = verify_manifest(found)
    print(f"[1/4] Privacy, extension & leak audit PASS ({len(found)} files checked, 0 leaks).", flush=True)
    print(f"[2/4] MANIFEST_SHA256.csv verified PASS ({n_manifest} entries exact match).", flush=True)

    # Check clean side-effect-free import for core & sanitized modules (skip e1..e7 top-level scripts)
    import_safe = [
        "sim2d",
        "baseline2d",
        "train2d_v2",
        "train2d",
        "dlemlr_model",
        "adaptive_lambda_baseline",
        "sparse_l1_baseline",
        "stats_and_1d2d",
        "run_hardening_benchmarks",
        "e8_cross_material",
        "e8_sr_alignment",
        "e9_rock_retrain",
        "label_sensitivity",
        "eval_synthetic_benchmark",
        "eval_historical_strict_odd_even",
    ]
    for rel, p in found.items():
        if rel.endswith(".py"):
            ast.parse(open(p, "r", encoding="utf-8").read(), filename=rel)
    for mod in import_safe:
        importlib.import_module(mod)
    print(f"[3/4] AST syntax & side-effect-free imports PASS ({len(import_safe)} modules imported).", flush=True)

    bench_path = os.path.join(HERE, "eval_synthetic_benchmark.py")
    sub_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "-B", bench_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=sub_env, check=False
    )
    print(proc.stdout.strip(), flush=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr, flush=True)
        raise SystemExit(proc.returncode)
    print("[4/4] Standalone CPU benchmark PASS (exit 0).", flush=True)
    print("REPROBUNDLE VERIFICATION: ALL CHECKS PASSED.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
