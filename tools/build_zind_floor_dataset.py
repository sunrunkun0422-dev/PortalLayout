#!/usr/bin/env python3
"""Build RGB-only, variable-N ZInD floor samples and separate offline labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portallayout.datasets.zind_floor_builder import build_dataset, chosen_houses, source_observations, write_json
from portallayout.datasets.zind_floor_dataset import read_json, safe_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="read-only original ZInD data root")
    p.add_argument("--partition", required=True, help="official zind_partition.json")
    p.add_argument("--output", help="new output directory; existing output requires exact --resume")
    p.add_argument("--config", default=str(REPO_ROOT / "configs/zind_floorset_v1.json"))
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "val", "test"])
    p.add_argument("--houses", nargs="+")
    p.add_argument("--floors", nargs="+")
    p.add_argument("--max-houses", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--house-workers", type=int, default=1, help="independent house processes; workers is image threads per house")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--materialize-pairs", action="store_true", help="write every pair; default uses documented lazy labels")
    p.add_argument("--dry-run", action="store_true", help="annotation inventory only; no image decode/build")
    p.add_argument("--audit-report", help="optional JSON output for --dry-run")
    args = p.parse_args(argv)
    if args.workers < 1 or args.house_workers < 1 or (args.max_houses is not None and args.max_houses < 1):
        p.error("workers and max-houses must be positive")
    if not args.dry_run and not args.output:
        p.error("--output is required for a build")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = read_json(args.config)
    if not 0 <= config["dev_fraction"] < 1:
        raise ValueError("invalid dev_fraction")
    if args.dry_run:
        selected, lookup = chosen_houses(read_json(args.partition), args.splits, args.houses, args.max_houses)
        report = {"phase": "dry_run", "selected_splits": args.splits, "selected_houses": len(selected),
                  "available_houses": 0, "annotated_floors": 0, "annotated_observations": 0,
                  "missing_houses": [], "images_decoded": False,
                  "val_test_source_accessed": bool(set(args.splits) & {"val", "test"})}
        for house in selected:
            path = safe_path(args.root, house + "/zind_data.json")
            if not path.is_file():
                report["missing_houses"].append(house)
                continue
            payload = read_json(path)
            records = list(source_observations(payload, house, args.floors))
            report["available_houses"] += 1
            report["annotated_floors"] += sum(not args.floors or fid in args.floors for fid in payload.get("merger", {}))
            report["annotated_observations"] += len(records)
        if args.audit_report:
            write_json(args.audit_report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    stats = build_dataset(args.root, args.partition, args.output, config, args.splits, args.houses,
                          args.floors, args.workers, args.resume, args.materialize_pairs, args.max_houses, args.house_workers)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
