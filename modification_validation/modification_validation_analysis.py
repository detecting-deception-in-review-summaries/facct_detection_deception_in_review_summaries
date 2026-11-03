"""Inter-annotator agreement for the validation of the controlled modifications.

Reads `modification_validation_dataset.json`: a sample of modified summaries,
each labelled by three annotators (A1-A3) as a correct (yes) or incorrect (no)
application of the requested modification. It reports

  1) each annotator's labels, and Krippendorff's alpha and Gwet's AC1 over all
     annotators together, and
  2) the share of items marked correct, after consolidating each item to one
     label by majority vote - no annotator is privileged, and an even split
     counts as incorrect.

Unsure and empty labels are missing (`null`) and are left out, rather than
being folded into yes or no.

Usage:
    uv run python modification_validation/modification_validation_analysis.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nltk.metrics.agreement import AnnotationTask

DATASET_JSON = Path(__file__).resolve().parent / "modification_validation_dataset.json"

LABEL_WORDS = {1: "yes", 0: "no"}
CLASS_NAMES = {1: "correct modification", 0: "incorrect modification"}


def krippendorff_alpha(ratings: list[list[int]]) -> float:
    """Nominal Krippendorff's alpha (NLTK) over the units rated by two or more annotators."""
    triples = [(rater, unit, value)
               for unit, values in enumerate(r for r in ratings if len(r) >= 2)
               for rater, value in enumerate(values)]
    return AnnotationTask(triples).alpha()


def gwet_ac1(ratings: list[list[int]], categories: tuple[int, ...] = (0, 1)) -> float:
    """Gwet's AC1 (Gwet 2008) for any number of raters, with missing ratings allowed.

    No major Python library provides it; this follows Gwet's definition and
    matches the reference implementation (irrCAC) on the reported tables.
    """
    rated = [unit for unit in ratings if unit]
    units = [unit for unit in rated if len(unit) >= 2]
    p_a = sum(
        sum(count * (count - 1) for count in Counter(unit).values()) / (len(unit) * (len(unit) - 1))
        for unit in units
    ) / len(units)
    # Chance agreement is estimated from every rated row, as Gwet (2008) defines it.
    pi = {category: sum(Counter(unit)[category] / len(unit) for unit in rated) / len(rated)
          for category in categories}
    p_e = sum(pi[c] * (1 - pi[c]) for c in categories) / (len(categories) - 1)
    return (p_a - p_e) / (1 - p_e)


def majority_label(item: dict) -> tuple[int, bool]:
    """(label, tied): the majority vote of the annotators who labelled the item.

    An even split, possible only when an even number labelled it, counts as
    incorrect, the conservative reading.
    """
    counts = Counter(label for label in item["labels"].values() if label is not None)
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    return (winners[0], False) if len(winners) == 1 else (0, True)


def report_agreement(data: dict) -> None:
    annotators, items = data["annotators"], data["items"]
    ratings = [[item["labels"][a] for a in annotators if item["labels"].get(a) is not None]
               for item in items]

    print("=" * 72)
    print(f"AGREEMENT ON `correct`  "
          f"(yes = 1 = {CLASS_NAMES[1]}, no = 0 = {CLASS_NAMES[0]}; unsure/blank excluded)")
    print("=" * 72)
    print(f"Annotators: {', '.join(annotators)}")
    print(f"Rows: {len(items)} total, "
          f"{sum(len(unit) >= 2 for unit in ratings)} labelled yes/no by 2+ annotators")
    for annotator in annotators:
        counts = Counter(LABEL_WORDS.get(item["labels"][annotator], "blank") for item in items)
        print(f"  {annotator}: {', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}")

    print("\nALL ANNOTATORS TOGETHER")
    print(f"  Krippendorff's alpha (nominal): {krippendorff_alpha(ratings):.3f}")
    print(f"  Gwet's AC1:                     {gwet_ac1(ratings):.3f}")


def report_correct_rate(data: dict) -> None:
    """Share of items whose majority label is correct."""
    resolved = [majority_label(item) for item in data["items"]]
    total = len(resolved)
    correct = sum(label == 1 for label, _ in resolved)
    ties = sum(tied for _, tied in resolved)

    print("\n" + "=" * 72)
    print("CORRECT RATE (majority label per row)")
    print("=" * 72)
    print(f"  {CLASS_NAMES[1].capitalize()} (1): {correct} of {total} rows = {correct / total:.1%}")
    print(f"  {CLASS_NAMES[0].capitalize()} (0): {total - correct} of {total} rows "
          f"= {(total - correct) / total:.1%}")
    if ties:
        print(f"  Even splits with no majority, counted as {CLASS_NAMES[0]}: {ties}")


def main() -> None:
    data = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    report_agreement(data)
    report_correct_rate(data)


if __name__ == "__main__":
    main()
