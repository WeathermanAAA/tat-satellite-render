#!/usr/bin/env python3
"""Command-line entrypoint for the staged true-colour validation harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.truecolor.gates import ResearchIntegrityError, run_baseline, verify_research_root
from validation.truecolor.report import UnsafeArtifactPath, build_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tat_truecolor_validate",
        description="Research-grounded TAT true-colour validation harness",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser(
        "baseline", help="record current HEAD without re-blessing known failures"
    )
    baseline.add_argument(
        "--research-root",
        type=Path,
        required=True,
        help="explicit extracted canonical TAT_TRUECOLOR_RESEARCH root",
    )
    baseline.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="external review-artifact directory (must not be inside any Git worktree)",
    )
    baseline.add_argument(
        "--enforce-through",
        choices=("M0",),
        default=None,
        help="enforce the implemented M0 measurement/safety contracts",
    )
    baseline.add_argument(
        "--rayleigh-data-dir",
        type=Path,
        default=None,
        help=(
            "explicit external writable directory used by both rsr_dir and "
            "rayleigh_dir; required with --enforce-through M0"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.enforce_through == "M0" and arguments.rayleigh_data_dir is None:
        parser.error("--rayleigh-data-dir is required with --enforce-through M0")
    try:
        research = verify_research_root(arguments.research_root)
        run = run_baseline(
            research,
            enforce_through=arguments.enforce_through,
            rayleigh_data_directory=arguments.rayleigh_data_dir,
        )
        report = build_report(run, enforce_through=arguments.enforce_through)
        written = write_report(report, arguments.artifact_dir)
    except (ResearchIntegrityError, UnsafeArtifactPath, FileExistsError, OSError, RuntimeError) as exc:
        print(f"tat_truecolor_validate: {exc}", file=sys.stderr)
        return 2

    summary = {
        "report": str(written.report_path),
        "sha256": written.report_sha256,
        "outcomes": report["summary"]["outcomes"],
        "blocking_gate_count": report["summary"]["blocking_gate_count"],
        "m0_enforcement": report["summary"]["m0_enforcement"],
    }
    print(json.dumps(summary, sort_keys=True))
    if arguments.enforce_through and not run.enforcement.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
