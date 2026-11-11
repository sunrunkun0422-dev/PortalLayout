"""Grouped, dependency-light entry point for the existing research tools."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Keep algorithms in their original modules so saved models and old commands work.
COMMANDS = {
    "data": {
        "build": ("tools/build_zind_floor_dataset.py", "Build ZInD floor data"),
        "validate": ("tools/validate_zind_floor_dataset.py", "Validate floor data"),
    },
    "openings": {
        "predict": ("tools/predict_floor_observations.py", "Cache RGB frontend predictions"),
        "encode": ("tools/encode_floor_openings.py", "Encode opening representations"),
        "pairs": ("tools/run_opening_pairs.py", "Build labels, train, match, or calibrate pairs"),
        "report": ("tools/report_opening_matches.py", "Offline matching diagnostics"),
    },
    "floor": {
        "run": ("tools/run_floor_layout.py", "Prepare, train, calibrate, infer, or evaluate floors"),
        "preflight": ("tools/preflight_floor_training.py", "Check floor training prerequisites"),
    },
    "evidence": {
        "build": ("tools/build_method_advantage_evidence.py", "Recompute frozen-artifact evidence"),
    },
    "legacy": {
        "layout": ("main.py", "Original Bi-Layout training and evaluation"),
        "cross-scene": ("tools/estimate_cross_scene_layout.py", "Two-room geometric alignment"),
        "train-pairs": ("tools/train_zind_end_to_end.py", "Earlier pairwise joint training"),
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m portallayout",
        description="PortalLayout — opening-guided cross-room layout reconstruction",
        epilog="Run from the repository root. Arguments after the action are forwarded unchanged.",
    )
    groups = parser.add_subparsers(dest="group", required=True)
    for group, actions in COMMANDS.items():
        group_parser = groups.add_parser(group)
        action_parsers = group_parser.add_subparsers(dest="action", required=True)
        for action, (_, description) in actions.items():
            action_parser = action_parsers.add_parser(action, help=description, add_help=False)
            action_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    # Parse only the routing prefix; --help and all other tool flags belong to
    # the original parser, including flags before its own positional arguments.
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] not in COMMANDS or args[1] not in COMMANDS[args[0]]:
        parser.parse_args(args)
        return 0
    script = ROOT / COMMANDS[args[0]][args[1]][0]
    if Path.cwd().resolve() != ROOT:
        parser.error("run from the repository root: {}".format(ROOT))
    return subprocess.call([sys.executable, str(script), *args[2:]])


if __name__ == "__main__":
    raise SystemExit(main())
