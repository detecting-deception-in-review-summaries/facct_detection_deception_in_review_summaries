"""Regenerate every figure, table and report in `results/`.

Runs the analyses in order: the notebooks are executed in place (so their
stored outputs are refreshed), and the scripts' printed reports are saved to
`results/reports/`. The Google Maps steps are skipped when the case-study data,
which is available on request, is not present.

Usage:
    uv run python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORTS = RESULTS / "reports"
GOOGLE_MAPS_DATA = ROOT / "google_maps_experiment" / "google_maps_dataset.json"

# (description, path, extra arguments); notebooks end in .ipynb, the rest are scripts.
STEPS = [
    ("Score distributions and metric comparison", "src/plot_scores.ipynb", []),
    ("Ranking for manual review", "src/manual_audit.ipynb", []),
    ("Market monitoring", "src/market_monitoring.ipynb", []),
    ("Modification percentages", "modification_validation/print_modifications_percentages.py", []),
    ("Modification validation", "modification_validation/modification_validation_analysis.py", []),
    ("Google Maps platform verdict", "google_maps_experiment/google_maps_analysis.ipynb", []),
    ("Google Maps annotation analysis",
     "google_maps_experiment/google_maps_annotation_analysis.py", []),
]


def command(path: str, arguments: list[str]) -> list[str]:
    if path.endswith(".ipynb"):
        return [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=-1", path]
    return [sys.executable, path, *arguments]


def run_step(description: str, path: str, arguments: list[str]) -> str:
    """Run one step; return a one-line summary, or exit with its output on failure."""
    if path.startswith("google_maps_experiment/") and not GOOGLE_MAPS_DATA.exists():
        return f"skipped  {description} (google_maps_dataset.json is available on request)"

    started = time.time()
    result = subprocess.run(command(path, arguments), cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - started
    if result.returncode != 0:
        tqdm.write(f"\nFAILED   {description} ({path})\n")
        tqdm.write(result.stdout[-3000:] + result.stderr[-3000:])
        sys.exit(result.returncode)

    if path.endswith(".py"):
        report = REPORTS / f"{Path(path).stem}.txt"
        report.write_text(result.stdout, encoding="utf-8")
        where = f"report: {report.relative_to(ROOT)}"
    else:
        where = f"outputs stored in {path}"
    return f"done     {description} in {elapsed:.0f}s ({where})"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tqdm(STEPS, unit="step", bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}]") as bar:
        for description, path, arguments in bar:
            bar.set_description(description)
            tqdm.write(run_step(description, path, arguments))

    files = sorted(p.relative_to(ROOT) for p in RESULTS.rglob("*") if p.is_file())
    print(f"\nFinished in {(time.time() - started) / 60:.1f} min. {len(files)} files in results/:")
    for kind in ("figures", "tables", "reports"):
        listed = [f for f in files if f.parts[1] == kind and f.suffix != ".png"]
        print(f"\n  {kind}/")
        for f in listed:
            print(f"    {Path(*f.parts[2:])}")


if __name__ == "__main__":
    main()
