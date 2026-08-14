"""Wave-based multi-experiment feature-extraction runner.

Drives :func:`submit_feature_extraction_jobs` for many experiments **per
phase**: all GPU arrays go in together, then all SPMD CPU arrays, then all
merge jobs, then all aggregate jobs. Between waves we use
:func:`wait_for_multiple_job_arrays` so progress is reported as a unified
cohort instead of one-experiment-at-a-time.

Behavioural contract (per user request):
  * Submission is all-or-nothing per wave: if any experiment fails to queue,
    every array already queued in this wave is ``scancel``-ed and the run
    aborts.
  * Runtime SLURM failures (any task in any array exits non-zero) **stop the
    entire run** — we don't proceed to later waves.
  * On any failure we write ``<run_dir>/_wave_failures.csv`` with columns
    (experiment, phase, base_job_id) and print a copy-pasteable rerun command
    that retries just the failed experiments from the failed phase.

Public entry point::

    from organelle_profiler.feature_extraction.multi_experiment_wave_runner import run_waves
    run_waves(experiments, build_args, run_dir=Path("..."))

``build_args(exp)`` is a callable returning the ``argparse.Namespace`` that
:func:`submit_feature_extraction_jobs` expects. The runner mutates copies of
that Namespace per wave to set ``resume_from`` / ``stop_after`` / ``no_wait``.
"""
from __future__ import annotations

import copy
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable

PHASES = ("gpu", "spmd", "merge", "aggregate")


def _clean_stale_phase_outputs(
    experiments: list[str],
    phase: str,
    build_args: Callable[[str], "object"],
) -> None:
    """Remove stale completion markers for *phase* so it reruns cleanly.

    Each phase has its own output/marker pattern inside ``_batch_results``:
      gpu       — ``batch_*_cells.parquet``, ``_batch_config.json``
      spmd      — ``_partials_combined/`` (rank*_done markers + rank parquets)
      merge     — ``batch_*_cells.parquet`` (final merged; same glob as gpu but
                   only relevant when rerunning merge after spmd changed)
      aggregate — no markers to clean (always overwrites AnnData)
    """
    from organelle_profiler.feature_extraction.feature_extraction_slurm import (
        _resolve_fe_output_dir,
    )

    if phase == "aggregate":
        return  # nothing to clean

    cleaned = 0
    for exp in experiments:
        args = build_args(exp)
        output_dir = _resolve_fe_output_dir(exp, args)
        batch_results = output_dir / "_batch_results"

        if phase == "spmd":
            partials = batch_results / "_partials_combined"
            if partials.exists():
                shutil.rmtree(partials)
                cleaned += 1
        elif phase == "gpu":
            for p in batch_results.glob("batch_*_cells.parquet"):
                p.unlink()
                cleaned += 1
            cfg = batch_results / "_batch_config.json"
            if cfg.exists():
                cfg.unlink()
            # Also wipe the per-cell pickle cache so GPU's resume logic doesn't
            # short-circuit on stale partials when cells_df has changed since
            # the last run.
            for partials in batch_results.glob("_gpu_partials_*"):
                if partials.is_dir():
                    shutil.rmtree(partials)
                    cleaned += 1
        elif phase == "merge":
            # Merge writes final batch_*_cells.parquet — same as gpu output
            for p in batch_results.glob("batch_*_cells.parquet"):
                p.unlink()
                cleaned += 1

    if cleaned:
        print(f"  Cleaned stale {phase} outputs for {cleaned} experiment(s)")


def _scancel(base_job_ids: Iterable[str]) -> None:
    """Best-effort scancel of every base job ID."""
    for jid in base_job_ids:
        if not jid:
            continue
        try:
            subprocess.run(["scancel", str(jid)], check=False, timeout=10)
            print(f"  scancel {jid}")
        except Exception as e:
            print(f"  WARN: scancel failed for {jid}: {e}")


def _write_failure_csv(
    run_dir: Path,
    failures: list[tuple[str, str, str]],
) -> Path:
    """Write (experiment, phase, base_job_id) rows for easy rerun."""
    path = run_dir / "_wave_failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "phase", "base_job_id"])
        for row in failures:
            w.writerow(row)
    return path


def _print_rerun_command(failed_exps: list[str], phase: str) -> None:
    print(f"\n{'='*72}")
    print("To rerun the failed experiments from this phase:")
    print(
        f"  python -m organelle_profiler.feature_extraction.run_fe_all_experiments \\\n"
        f"    --experiments {' '.join(failed_exps)} \\\n"
        f"    --resume-from {phase}\n"
    )
    print("(append other original CLI flags as needed: --modality, --output-base, etc.)")
    print(f"{'='*72}\n")


def _phase_kwargs(phase: str) -> dict:
    """Map wave name to the (resume_from, stop_after) pair for that wave.

    Each wave runs **only** its own phase: resume_from sets the start, stop_after
    the end. Both equal to the phase name.
    """
    return {"resume_from": phase, "stop_after": phase, "no_wait": True, "split": True}


def _extract_array_entry(
    exp: str,
    submit_result: dict,
    phase: str,
) -> dict | None:
    """Pull the ``submit_parallel_jobs``-shape result for the wave just submitted.

    The phases nest results differently: GPU returns ``submit_parallel_jobs``
    output directly; SPMD/Merge return ``{"spmd_result": ..., "merge_result": ...}``.
    """
    if not submit_result.get("success"):
        return None

    # GPU wave — submit_feature_extraction_jobs returns gpu_result top-level
    if phase == "gpu":
        gpu_result = submit_result.get("gpu_result") or submit_result
        if not gpu_result.get("submitted_jobs") or not gpu_result.get("base_job_id"):
            return None
        return {
            "label": exp,
            "submitted_jobs": gpu_result["submitted_jobs"],
            "base_job_id": gpu_result["base_job_id"],
            "slurm_params": gpu_result.get("slurm_params") or {},
        }

    # SPMD / Merge — both nested under spmd_result
    if phase in ("spmd", "merge"):
        spmd_outer = submit_result.get("spmd_result") or {}
        # SPMD wave: the inner submit_parallel_jobs result for the SPMD array is
        # at spmd_result["spmd_result"] (when submit_merge_after=False, no inner merge).
        # Merge wave: spmd_result IS the merge submit_parallel_jobs result.
        target = spmd_outer.get("spmd_result") if phase == "spmd" else spmd_outer
        if not target or not target.get("submitted_jobs") or not target.get("base_job_id"):
            return None
        return {
            "label": exp,
            "submitted_jobs": target["submitted_jobs"],
            "base_job_id": target["base_job_id"],
            "slurm_params": target.get("slurm_params") or {},
        }

    if phase == "aggregate":
        agg = submit_result.get("agg_result") or {}
        if not agg.get("submitted_jobs") or not agg.get("base_job_id"):
            return None
        return {
            "label": exp,
            "submitted_jobs": agg["submitted_jobs"],
            "base_job_id": agg["base_job_id"],
            "slurm_params": agg.get("slurm_params") or {},
        }

    return None


def run_waves(
    experiments: list[str],
    build_args: Callable[[str], "object"],
    run_dir: Path,
    *,
    label: str = "fe_batch",
    phases: tuple[str, ...] = PHASES,
    clean: bool = False,
) -> dict:
    """Drive an N-experiment feature-extraction batch wave-by-wave.

    Parameters
    ----------
    experiments
        Ordered list of experiment names to process.
    build_args
        Callable taking an experiment name and returning an ``argparse.Namespace``
        suitable for :func:`submit_feature_extraction_jobs`. The runner copies
        the Namespace per wave and sets ``resume_from`` / ``stop_after`` /
        ``no_wait`` / ``split`` on the copy — the underlying defaults you set
        in ``build_args`` (modality, output_base, etc.) are preserved.
    run_dir
        Where to write the failure CSV on abort.
    label
        Free-form tag passed to ``wait_for_multiple_job_arrays`` for log lines.
    phases
        Phases to actually run, in order. Defaults to all four
        (``gpu``, ``spmd``, ``merge``, ``aggregate``); pass a tail slice to
        resume e.g. ``("merge", "aggregate")``.
    """
    from organelle_profiler.feature_extraction.feature_extraction_slurm import (
        submit_feature_extraction_jobs,
    )
    from cyclops_utils.hpc.slurm_batch_utils import wait_for_multiple_job_arrays

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    survivors = list(experiments)
    summary: dict = {"phases": {}}

    for phase_idx, phase in enumerate(phases, start=1):
        print(f"\n{'='*72}")
        print(f"WAVE {phase_idx}/{len(phases)} — {phase.upper()}  ({len(survivors)} experiments)")
        print(f"{'='*72}\n")

        # ---- 0. Clean stale completion markers from previous runs (opt-in)
        if clean:
            _clean_stale_phase_outputs(survivors, phase, build_args)

        # ---- 1. Submission pass: queue every experiment's wave-`phase` array
        wave_arrays: list[dict] = []
        submission_failures: list[tuple[str, str]] = []

        for exp in survivors:
            args = copy.copy(build_args(exp))
            for k, v in _phase_kwargs(phase).items():
                setattr(args, k, v)

            print(f"\n[{exp}] preparing wave-{phase} submission")
            try:
                result = submit_feature_extraction_jobs(exp, args)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"\n[{exp}] EXCEPTION during prep:\n{tb}", flush=True)
                submission_failures.append((exp, f"exception: {type(e).__name__}: {e}"))
                break

            if not result.get("success"):
                submission_failures.append((exp, str(result.get("error") or "submit returned success=False")))
                break

            entry = _extract_array_entry(exp, result, phase)
            if entry is None:
                # No jobs queued — phase already complete for this experiment.
                # Skip it; don't abort the batch.
                print(f"[{exp}] wave-{phase}: nothing to submit (already complete), skipping")
                continue
            wave_arrays.append(entry)

        # If anything failed during submission, roll back THIS wave + abort run
        if submission_failures:
            failed_exp = submission_failures[0][0]
            print(f"\n[{failed_exp}] submission failure: {submission_failures[0][1]}")
            print(f"Rolling back {len(wave_arrays)} already-queued array(s) in this wave...")
            _scancel(a["base_job_id"] for a in wave_arrays)
            failures = [(failed_exp, phase, "")]
            csv_path = _write_failure_csv(run_dir, failures)
            print(f"Wrote {csv_path}")
            _print_rerun_command([failed_exp], phase)
            raise RuntimeError(
                f"Wave {phase} submission failed at {failed_exp}: aborted batch."
            )

        if not wave_arrays:
            print(f"\nNo arrays queued for wave {phase} — nothing to wait on (treating as no-op).")
            summary["phases"][phase] = {"queued": 0, "completed": 0, "failed": []}
            continue

        # ---- 2. Wait for the whole wave (one progress bar across N exps)
        wait_result = wait_for_multiple_job_arrays(
            job_arrays=wave_arrays,
            experiment=f"{label}_{phase}",
            verbose=True,
            print_resource_summary=True,
            print_success=True,
        )

        array_results = wait_result.get("array_results", {}) or {}
        runtime_failed_exps: list[str] = []
        for exp_label, results in array_results.items():
            if results.get("failed"):
                runtime_failed_exps.append(exp_label)

        summary["phases"][phase] = {
            "queued": len(wave_arrays),
            "completed": len(wave_arrays) - len(runtime_failed_exps),
            "failed": runtime_failed_exps,
        }

        # ---- 3. Hard-stop on any runtime failure
        if runtime_failed_exps:
            print(
                f"\nWave {phase} runtime failures: "
                f"{len(runtime_failed_exps)}/{len(wave_arrays)} experiments had failed tasks"
            )
            failures = []
            for exp in runtime_failed_exps:
                base_job_id = next(
                    (a["base_job_id"] for a in wave_arrays if a["label"] == exp), ""
                )
                failures.append((exp, phase, base_job_id))
            csv_path = _write_failure_csv(run_dir, failures)
            print(f"Wrote {csv_path}")
            _print_rerun_command(runtime_failed_exps, phase)
            raise RuntimeError(
                f"Wave {phase} had {len(runtime_failed_exps)} runtime failure(s): "
                f"aborting batch as requested."
            )

        # All survivors made it through this wave; carry them to the next.
        # (No drop-out: "stop on any failure" semantics mean we either continue
        # with the same set or have already raised.)

    print(f"\n{'='*72}")
    print(f"All {len(phases)} waves complete for {len(experiments)} experiment(s).")
    print(f"{'='*72}\n")
    summary["all_completed"] = True
    return summary
