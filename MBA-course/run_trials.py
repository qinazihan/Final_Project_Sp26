#!/usr/bin/env python3
"""
Run MBA Agent v4 with different model combinations (auto-approve mode).
Saves results to results/trials/ for comparison.

Usage:
  python run_trials.py                        # run all trials
  python run_trials.py mixed_best gemini      # run specific trials
"""
import os, sys, yaml, time, shutil, subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
AGENT_SCRIPT = SCRIPT_DIR / "mba_agent_v4.py"
RESULTS_DIR = SCRIPT_DIR / "results" / "trials"

# ── Trial configurations ──
TRIALS = {
    "baseline_gpt4o": {
        "intake":      "openai/gpt-4o",
        "researcher":  "openai/gpt-4o",
        "critic":      "openai/gpt-4o",
        "synthesizer": "openai/gpt-4o",
    },
    "mixed_best": {
        "intake":      "openai/gpt-4o",
        "researcher":  "google/gemini-3-flash-preview",
        "critic":      "anthropic/claude-sonnet-4.5",
        "synthesizer": "anthropic/claude-sonnet-4.5",
    },
    "gemini_heavy": {
        "intake":      "google/gemini-2.0-flash-001",
        "researcher":  "google/gemini-3-flash-preview",
        "critic":      "google/gemini-3-pro-preview",
        "synthesizer": "google/gemini-3-pro-preview",
    },
}


def run_trial(trial_name: str, models: dict):
    """Run agent with a temp config. Original config.yaml is never touched."""
    print(f"\n{'='*80}")
    print(f"  TRIAL: {trial_name}")
    print(f"{'='*80}")
    for role, model in models.items():
        print(f"  {role:12s} -> {model}")
    print()

    # Build temp config from original + patched models
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for role, model in models.items():
        cfg["agents"][role] = {"model": model}

    tmp_config = SCRIPT_DIR / f".config_trial_{trial_name}.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Prepare trial output directory
    trial_dir = RESULTS_DIR / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "log.txt"

    env = os.environ.copy()
    env["MBA_AUTO_APPROVE"] = "true"
    env["MBA_CONFIG_PATH"] = str(tmp_config)

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(AGENT_SCRIPT)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(SCRIPT_DIR),
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
            proc.wait(timeout=600)

        elapsed = time.time() - t0
        exit_code = proc.returncode
        print(f"\n  Trial '{trial_name}' finished in {elapsed:.0f}s (exit {exit_code})")

        # Move MD + DOCX files into trial dir
        results_base = SCRIPT_DIR / "results"
        for pattern in ["mba_v4_report_*.md", "mba_v4_report_*.docx"]:
            files = sorted(results_base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                dest = trial_dir / files[0].name
                shutil.move(str(files[0]), str(dest))
                print(f"  Saved: {dest.name}")

        # Write metadata
        with open(trial_dir / "meta.yaml", "w") as f:
            yaml.dump({
                "trial": trial_name,
                "models": models,
                "elapsed_seconds": round(elapsed, 1),
                "exit_code": exit_code,
            }, f, default_flow_style=False)

        return {"name": trial_name, "elapsed": elapsed, "success": exit_code == 0}

    except subprocess.TimeoutExpired:
        proc.kill()
        elapsed = time.time() - t0
        print(f"\n  TIMEOUT after {elapsed:.0f}s")
        return {"name": trial_name, "elapsed": elapsed, "success": False}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ERROR: {e}")
        return {"name": trial_name, "elapsed": elapsed, "success": False}
    finally:
        tmp_config.unlink(missing_ok=True)


def main():
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        trials = {k: v for k, v in TRIALS.items() if k in selected}
        if not trials:
            print(f"Unknown trials: {selected}")
            print(f"Available: {list(TRIALS.keys())}")
            sys.exit(1)
    else:
        trials = TRIALS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"  MBA Agent v4 — Model Trials ({len(trials)} trials)")
    print(f"{'='*80}")
    print(f"  Results -> {RESULTS_DIR}")
    print(f"  Trials:  {', '.join(trials.keys())}")
    print(f"  Config:  {CONFIG_PATH} (read-only)")
    print()

    results = []
    for name, models in trials.items():
        r = run_trial(name, models)
        results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print("  TRIAL SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Trial':<25} {'Time':>8}  {'Status':<6}")
    print(f"  {'-'*45}")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  {r['name']:<25} {r['elapsed']:>7.0f}s  {status:<6}")
    print(f"{'='*80}")
    print(f"  Results: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
