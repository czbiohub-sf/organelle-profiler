"""Run organelle-profiler feature extraction across many experiments using
the 4-wave orchestrator.

Mirrors :mod:`run_top_cells_pipeline` but **without** the top-cells CSV filter
— every cell in each experiment's phenotyping_v3 store is processed (subject
to the ``--modality`` / ``--organelles`` filters).

Wave orchestration:
    Wave 1 — GPU (morphology + localization), all N experiments queued in parallel
    Wave 2 — SPMD CPU (network analysis), per-experiment arrays in parallel
    Wave 3 — Merge (combines partials), one job per experiment
    Wave 4 — Aggregate (final AnnData), one job per experiment

Hard-stops the entire batch on any failure and writes a rerun manifest.

Examples::

    # Default: paper-v1 set, all modalities
    python -m organelle_profiler.feature_extraction.run_fe_all_experiments \\
        --paper-v1 \\
        --output-base $OPS_BASE_PATH/analysis/fe_paper_v1 \\
        --run-name fe_paper_v1_2026apr

    # Specific experiments, fluorescent only
    python -m organelle_profiler.feature_extraction.run_fe_all_experiments \\
        --experiments ops0094 ops0143 \\
        --modality fluorescent \\
        --output-base /tmp/fe_out --run-name spotcheck

    # Resume: rerun only the failures from a previous batch's GPU phase
    python -m organelle_profiler.feature_extraction.run_fe_all_experiments \\
        --experiments ops0094 ops0144 \\
        --resume-from gpu \\
        --output-base ... --run-name ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from organelle_profiler.paths import BASE_PATH


def _resolve_paper_v1_experiments() -> List[str]:
    from cyclops_utils.data.bad_experiments import is_excluded
    base = Path(f"{BASE_PATH}")
    all_exps = sorted(d.name for d in base.iterdir() if d.name.startswith("ops0") and d.is_dir())
    paper_exclude = (
        "bad", "iss_only", "do_not_run", "non_standard",
        "positive_control", "need_rescue",
    )
    return [e for e in all_exps if not is_excluded(e, categories=paper_exclude)]


def _resolve_experiment_names(raw: List[str]) -> List[str]:
    from cyclops_utils.data.filesystem import resolve_experiment_name
    out = []
    for name in raw:
        resolved = resolve_experiment_name(name, allow_interactive=False) or name
        out.append(resolved)
    return out


def _build_fe_args(
    experiment: str,
    output_base: Optional[Path],
    modality: str,
    organelles: Optional[str],
    *,
    resume_from: Optional[str],
    force: bool = False,
) -> argparse.Namespace:
    """Build the Namespace passed to ``submit_feature_extraction_jobs``.

    The wave runner mutates per-phase fields (resume_from / stop_after / no_wait)
    on a copy of this Namespace; ``resume_from`` here is the **starting** phase
    when the user supplied ``--resume-from`` (carried through to the first wave;
    later waves clear it).
    """
    return argparse.Namespace(
        experiment=experiment,
        cells_csv=None,                  # no top-cells filter — process all cells
        output_base=str(output_base) if output_base else None,
        modality=modality,
        organelles=organelles,
        wells=None,
        cells_per_batch=None,
        sequential=False,
        max_concurrent=None,
        partition=None,
        timeout=None,
        mem=None,
        cpus=None,
        gpus=None,
        agg_local=False,
        agg_mem="256G",
        preview=None,
        split=True,
        resume_from=resume_from,
        stop_after=None,
        no_wait=True,
        spmd_ntasks=1024,
        dry_run=False,
        quiet=False,
        force=force,
        checkpoint=False,
        escalate=False,
        aggregate_only=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Experiment selection
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--paper-v1", action="store_true",
        help="Process the v1-paper experiment set (default if no other selector given). "
             "Uses the same is_excluded filter as report_pipeline_status.",
    )
    grp.add_argument(
        "--experiments", "-e", nargs="+", default=None,
        help="Explicit experiment names or shorthand (e.g. '94' / 'ops0094_20251217').",
    )
    parser.add_argument(
        "--exclude-experiments", nargs="+", default=None,
        help="Experiments to exclude from the resolved set (shorthand or full name; "
             "matched on the ops-key prefix). Useful when running --paper-v1 minus a few "
             "exps that already finished in a separate run, e.g. --exclude-experiments 94 144.",
    )

    # Run config
    parser.add_argument(
        "--run-name", required=True,
        help="Short name for this batch (used to scope the failure CSV and, if "
             "--output-base is set, the output directory).",
    )
    parser.add_argument(
        "--output-base", type=Path, default=None,
        help="Optional root directory for FE outputs. When set, each experiment lands at "
             "<output-base>/<run-name>/<exp>/feature_extraction/. "
             "When unset (default), outputs go to each experiment's natural fast-ops "
             "location: <exp>/3-assembly/feature_extraction/. Pass --output-base only "
             "when you want a side-by-side run that doesn't touch the canonical per-exp dirs.",
    )

    # Feature-extraction filters
    parser.add_argument(
        "--modality", default="all", choices=["phase", "fluorescent", "all"],
        help="Which channel modality to extract (default: all).",
    )
    parser.add_argument(
        "--organelles", default=None,
        help="Optional comma-separated list of organelles to restrict to (overrides modality preset).",
    )

    # Wave-runner controls
    parser.add_argument(
        "--resume-from", choices=["gpu", "spmd", "merge", "aggregate"], default=None,
        help="When supplied (typically after a partial-failure rerun), the wave runner skips "
             "phases earlier than this. Used together with --experiments to resubmit just "
             "the failed subset.",
    )
    parser.add_argument(
        "--stop-after", choices=["gpu", "spmd", "merge", "aggregate"], default=None,
        help="Stop after this phase (don't run later phases). "
             "E.g. --resume-from spmd --stop-after spmd runs only the SPMD phase.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip experiments that already have feature_extraction outputs under the run dir.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe each experiment's stale batch_results/_batch_config and start GPU phase fresh. "
             "Required when source cell counts have changed since the last run.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Before each wave, delete stale completion markers from previous runs so the "
             "phase reruns from scratch. Without this flag, phases that appear already-complete "
             "(e.g. SPMD rank*_done markers from an old run) are skipped.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment list and per-phase plan, then exit. No SLURM activity.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve experiment list
    # ------------------------------------------------------------------
    if args.experiments:
        experiments = _resolve_experiment_names(args.experiments)
    else:
        # Default to paper-v1 unless --experiments was given.
        experiments = _resolve_paper_v1_experiments()
        if not args.paper_v1:
            print(f"(no --experiments given; defaulting to --paper-v1)")

    # Apply --exclude-experiments after resolution. Match by ops-key prefix so
    # users can type '94' / 'ops0094' / 'ops0094_20251217' and they all hit the
    # same target.
    if args.exclude_experiments:
        from cyclops_utils.data.filesystem import extract_ops_key

        def _to_ops_key(raw: str) -> str:
            # Bare digits "94" -> "ops0094"; ops-keys / full names handled by extract_ops_key.
            s = str(raw).strip()
            if s.isdigit():
                return f"ops{int(s):04d}"
            return extract_ops_key(s) or s

        exclude_keys = {_to_ops_key(r) for r in args.exclude_experiments}
        before = len(experiments)
        experiments = [
            e for e in experiments
            if (extract_ops_key(e) or e) not in exclude_keys
        ]
        n_dropped = before - len(experiments)
        print(f"--exclude-experiments: dropped {n_dropped} experiment(s) "
              f"(matched against {sorted(exclude_keys)})")

    if not experiments:
        print("No experiments to process; exiting.")
        return 1

    # Run directory holds the failure CSV + any housekeeping. It only doubles
    # as the output root when --output-base is set; otherwise FE outputs go to
    # each experiment's natural fast-ops dir.
    if args.output_base:
        run_dir = args.output_base / args.run_name
    else:
        run_dir = Path("slurm_logs/fe_all_runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        from cyclops_utils.data.experiment import OpsDataset
        kept = []
        for exp in experiments:
            if args.output_base:
                out_marker = args.output_base / args.run_name / exp / "feature_extraction"
            else:
                out_marker = OpsDataset(exp).results_fast / "feature_extraction"
            if out_marker.exists() and any(out_marker.glob("*.parquet")):
                print(f"  [skip] {exp}: outputs already present at {out_marker}")
                continue
            kept.append(exp)
        experiments = kept
        if not experiments:
            print("All experiments already done. Exiting.")
            return 0

    print(f"\n{'='*72}")
    print(f"FE wave runner — {len(experiments)} experiments")
    print(f"  run dir:   {run_dir}")
    print(f"  modality:  {args.modality}")
    if args.organelles:
        print(f"  organelles: {args.organelles}")
    if args.resume_from:
        print(f"  resume_from (first wave): {args.resume_from}")
    if args.output_base:
        print(f"  output_base: {args.output_base} → outputs at <output-base>/{args.run_name}/<exp>/feature_extraction/")
    else:
        print(f"  output_base: <unset> → outputs at <exp>/3-assembly/feature_extraction/ (canonical fast-ops)")
    print(f"{'='*72}\n")
    for exp in experiments:
        print(f"  - {exp}")
    print()

    if args.dry_run:
        from organelle_profiler.feature_extraction.multi_experiment_wave_runner import PHASES
        target_phases = (
            PHASES[PHASES.index(args.resume_from):] if args.resume_from else PHASES
        )
        print(f"DRY RUN — would walk {len(target_phases)} wave(s) for {len(experiments)} experiments:")
        for i, p in enumerate(target_phases, 1):
            print(f"  Wave {i}/{len(target_phases)}  {p.upper()}  → {len(experiments)} array(s)")
        print(f"\nFailure manifest path on real run: {run_dir / '_wave_failures.csv'}")
        print(f"\nNo SLURM activity. Re-run without --dry-run to submit.")
        return 0

    # ------------------------------------------------------------------
    # Hand off to the wave runner
    # ------------------------------------------------------------------
    from organelle_profiler.feature_extraction.multi_experiment_wave_runner import (
        run_waves, PHASES,
    )

    # If --resume-from / --stop-after supplied, slice the phase list.
    start_idx = PHASES.index(args.resume_from) if args.resume_from else 0
    stop_idx = PHASES.index(args.stop_after) + 1 if args.stop_after else len(PHASES)
    target_phases = PHASES[start_idx:stop_idx]
    if args.resume_from or args.stop_after:
        skipped = PHASES[:start_idx]
        if skipped:
            print(f"--resume-from {args.resume_from!r}: skipping phases {skipped}")
        print(f"Phases to run: {target_phases}\n")

    def _build(exp: str) -> argparse.Namespace:
        # When --output-base is None, pass None down so FE writes to the natural
        # per-experiment fast-ops dir.
        return _build_fe_args(
            experiment=exp,
            output_base=run_dir if args.output_base else None,
            modality=args.modality,
            organelles=args.organelles,
            resume_from=None,
            force=args.force,
        )

    result = run_waves(
        experiments=experiments,
        build_args=_build,
        run_dir=run_dir,
        label=args.run_name,
        phases=target_phases,
        clean=args.clean,
    )

    print("\nAll requested phases complete.")
    return 0 if result.get("all_completed") else 1


if __name__ == "__main__":
    sys.exit(main())
