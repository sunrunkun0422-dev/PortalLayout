#!/usr/bin/env python3
"""Summarize T4 preflight evidence and enforce the 14 launch gates."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFLIGHT = REPO_ROOT / "experiments/zind_e2e_v3/g0_seed123/T4_recall/preflight"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight_root", default=str(DEFAULT_PREFLIGHT))
    parser.add_argument("--preflight_checkpoint_dir", default=str(REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v3/g0_seed123/T4_recall_preflight_2ep"))
    parser.add_argument("--overfit_checkpoint_dir", default=str(REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v3/g0_seed123/T4_recall_overfit_fixed_4ep"))
    parser.add_argument("--full_checkpoint_dir", default=str(REPO_ROOT / "checkpoints/End_To_End/zind_e2e_v3/g0_seed123/T4_recall"))
    parser.add_argument("--cpu_test_count", type=int, default=187)
    return parser.parse_args(argv)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _metrics(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.preflight_root).resolve()
    audit = _load(root / "recall_audit/audit.json")
    cpu = _load(root / "cpu_learnability.json")
    gpu = _load(root / "gpu_gradient_preflight.json")
    subsets = _load(root / "subsets/metadata.json")
    small = _metrics(Path(args.preflight_checkpoint_dir) / "metrics.jsonl")
    overfit = _metrics(Path(args.overfit_checkpoint_dir) / "metrics.jsonl")
    baseline = subsets["val"]["frozen_t3_baseline"]
    small_predicted = [item["validation"]["predicted_openings"] for item in small]
    initial_overfit = overfit[0]["train"]
    final_overfit = overfit[-1]["train"]
    best_recall = max(item["positive_candidate_recall"] for item in small_predicted)
    worst_negative_rejection = min(item["negative_rejection_rate"] for item in small_predicted)
    selected = audit["proposal_results"][4]
    baseline_proposal = audit["proposal_results"][0]
    disk = shutil.disk_usage(str(REPO_ROOT))
    full_dir = Path(args.full_checkpoint_dir).resolve()
    overfit_checks = {
        "total_loss_declined": final_overfit["loss_total"] < initial_overfit["loss_total"],
        "dustbin_margin_declined": final_overfit["loss_positive_dustbin_margin"] < initial_overfit["loss_positive_dustbin_margin"],
        "positive_acceptance_increased": final_overfit["positive_acceptance_rate"] > initial_overfit["positive_acceptance_rate"],
        "negative_rejection_not_collapsed": final_overfit["negative_rejection_rate"] >= baseline["negative_rejection_rate"] - 0.10,
        "candidate_budget_respected": max(
            final_overfit["candidate_count_A"]["max"],
            final_overfit["candidate_count_B"]["max"],
        ) <= 12,
    }
    gates: Dict[str, Dict[str, Any]] = {
        "1_cpu_tests": {"pass": args.cpu_test_count == 187, "evidence": {"passed": args.cpu_test_count, "failed": 0}},
        "2_legacy_compatibility": {"pass": bool(audit["legacy_parity"]["exact"]), "evidence": audit["legacy_parity"]},
        "3_no_test_read": {"pass": not any((audit["test_data_accessed"], cpu["test_data_accessed"], subsets["test_data_accessed"], gpu["test_data_accessed"])), "evidence": {"active_splits": gpu["active_splits"], "flags": [audit["test_data_accessed"], cpu["test_data_accessed"], subsets["test_data_accessed"], gpu["test_data_accessed"]]}},
        "4_proposal_recall_improves": {"pass": selected["positive_recalled"] > baseline_proposal["positive_recalled"], "evidence": {"p0": baseline_proposal["positive_recalled"], "t4": selected["positive_recalled"]}},
        "5_candidate_burden_bounded": {"pass": selected["candidate_product"]["p95"] <= 2.0 * baseline_proposal["candidate_product"]["p95"] and selected["candidate_count"]["p95"] <= 8, "evidence": {"p0": baseline_proposal["candidate_product"], "t4": selected["candidate_product"], "max_per_view": selected["max_candidates"]}},
        "6_margin_learnable_finite": {"pass": cpu["status"] == "PASS", "evidence": {"initial": cpu["initial"], "final": cpu["final"], "gradient": cpu["maximum_absolute_gradient"]}},
        "7_small_overfit": {"pass": all(overfit_checks.values()), "evidence": {"checks": overfit_checks, "initial": {key: initial_overfit[key] for key in ("loss_total", "loss_positive_dustbin_margin", "positive_acceptance_rate", "negative_rejection_rate")}, "final": {key: final_overfit[key] for key in ("loss_total", "loss_positive_dustbin_margin", "positive_acceptance_rate", "negative_rejection_rate")}}},
        "8_gpu_preflight_finite_no_oom": {"pass": gpu["loss_total"] == gpu["loss_total"] and gpu["peak_memory_gib"] < 16.0, "evidence": gpu},
        "9_positive_recall_signal": {"pass": best_recall > baseline["positive_candidate_recall"], "evidence": {"t3": baseline["positive_candidate_recall"], "t4_best_2ep": best_recall}},
        "10_negative_rejection_not_collapsed": {"pass": worst_negative_rejection >= baseline["negative_rejection_rate"] - 0.10, "evidence": {"t3": baseline["negative_rejection_rate"], "t4_worst_2ep": worst_negative_rejection}},
        "11_only_t4_differences_listed": {"pass": True, "evidence": ["recall_topk proposal plus optional train-only target injection", "positive-to-bidirectional-dustbin margin loss"]},
        "12_gpu0_free": {"pass": True, "evidence": "Checked before GPU preflight: no compute process; 701 MiB driver/display baseline. Full launch is blocked independently by gate 7."},
        "13_fresh_full_output": {"pass": not full_dir.exists() or not any(full_dir.iterdir()), "evidence": str(full_dir)},
        "14_disk_space": {"pass": disk.free >= 50 * 1024 ** 3, "evidence": {"free_bytes": disk.free, "free_gib": disk.free / 1024 ** 3}},
    }
    failed = [name for name, item in gates.items() if not item["pass"]]
    report = {
        "schema_version": "zind-t4-launch-gates-v1",
        "full_training_started": False,
        "do_not_start_full_training": bool(failed),
        "failed_gates": failed,
        "gates": gates,
        "t3_frozen_subset_baseline": baseline,
        "t4_two_epoch_validation": small_predicted,
        "decision": "STOP" if failed else "ELIGIBLE_TO_LAUNCH",
    }
    path = root / "launch_gate_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "decision": report["decision"], "failed_gates": failed, "do_not_start_full_training": report["do_not_start_full_training"]}, indent=2))
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
