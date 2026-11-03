"""Annotation analysis of the Google Maps case study.

Reads `google_maps_dataset.json` and reports:

  1. Agreement between the three annotators on which reviews are relevant to
     each summary clause, and on whether they support or contradict it.
  2. The time the annotators spent per restaurant.
  3. Agreement on the 1-5 Consistency and Relevance ratings of each summary,
     and how those ratings correlate with G-Eval.
  4. The contradiction burden of each summary (how strongly its reviews
     contradict it) against G-Eval Consistency.
  5. The aspects most of whose relevant reviews contradict them, with the
     restaurants ranked by G-Eval Consistency.

Annotators are combined by majority vote: a review is relevant to a clause when
most annotators marked it so, and its stance is the label most annotators gave
(support, contradict or neutral), with a tie going to neutral.

Usage:
    uv run python google_maps_experiment/google_maps_annotation_analysis.py
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import binary_distance, interval_distance
from scipy import stats

DATASET_JSON = Path(__file__).resolve().parent / "google_maps_dataset.json"

FLAGS = ("relevant", "supporting", "contradicting")
SCORES = ("consistency", "relevance")

PERMUTATIONS = 20_000  # for the permutation test on Kendall's tau-b
SEED = 42
THRESHOLD = 0.5  # an aspect is flagged when more than this share of its relevant reviews contradict it
SCREEN_SHARE = 0.25  # the share of lowest-scoring summaries an auditor would read


# =============================================================================
# Loading
# =============================================================================


def place_label(index: int) -> str:
    """The anonymized label the reports print in place of a restaurant's name:
    its position in the file, counted from 001 as in the review IDs."""
    return f"R{index:03d}"


def load(path: Path = DATASET_JSON) -> tuple[dict, list[dict]]:
    """The dataset's settings, and one entry per restaurant, in file order.

    Each restaurant keeps its annotated reviews only, with every clause's marks
    as {annotator: set of flags}.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    criterion = data["annotation"]["metric_criterion"]
    settings = {
        "annotators": data["annotation"]["annotators"],
        "criterion": criterion,
        "judge": data["judge"]["judge_name"],
    }

    restaurants = []
    for index, place in enumerate(data["places"], start=1):
        reviews = []
        for review in place["reviews"]:
            if "clauses" not in review:
                continue
            marks = {
                int(number): {annotator: set() if flags == "not_relevant" else set(flags)
                              for annotator, flags in by_annotator.items()}
                for number, by_annotator in review["clauses"].items()
            }
            reviews.append({"id": review["review_id"], "marks": marks})

        restaurants.append({
            "name": place_label(index),
            "summary": place["summary"],
            "clauses": place["clauses"],
            "geval": {score: place["geval"][score][criterion] for score in SCORES},
            "annotators": place["annotators"],
            "reviews": reviews,
        })
    return settings, restaurants


# =============================================================================
# Statistics
# =============================================================================


def krippendorff_alpha(units: list[list[int]], distance=binary_distance) -> float:
    """Krippendorff's alpha (NLTK) over units, each the list of ratings it received.

    `binary_distance` treats the ratings as nominal; `interval_distance` as the
    1-5 scale they are.
    """
    triples = [(rater, unit, value)
               for unit, ratings in enumerate(units)
               for rater, value in enumerate(ratings)]
    return AnnotationTask(triples, distance=distance).alpha()


def gwet_ac1(units: list[list[int]], categories: tuple[int, ...] = (0, 1)) -> float:
    """Gwet's AC1 (Gwet 2008) for any number of raters per unit.

    No major Python library provides it; this follows Gwet's definition and
    matches the reference implementation (irrCAC) on every table it reports.
    """
    p_a = sum(
        sum(count * (count - 1) for count in Counter(unit).values()) / (len(unit) * (len(unit) - 1))
        for unit in units
    ) / len(units)
    pi = {category: sum(Counter(unit)[category] / len(unit) for unit in units) / len(units)
          for category in categories}
    p_e = sum(pi[c] * (1 - pi[c]) for c in categories) / (len(categories) - 1)
    return (p_a - p_e) / (1 - p_e)


def permutation_p(xs: list[float], ys: list[float]) -> float:
    """Two-sided permutation p-value of Kendall's tau-b, from seeded shuffles of one side."""
    observed = abs(stats.kendalltau(xs, ys).statistic)
    rng = random.Random(SEED)
    shuffled = list(xs)
    extreme = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(shuffled)
        extreme += abs(stats.kendalltau(shuffled, ys).statistic) >= observed
    return (extreme + 1) / (PERMUTATIONS + 1)


def correlation_line(label: str, xs: list[float], ys: list[float]) -> str:
    """Pearson, Spearman and Kendall's tau-b, with a permutation p on Kendall's tau-b."""
    p_value = permutation_p(xs, ys)
    return (f"    {label:26} n={len(xs):3}  "
            f"Pearson r {stats.pearsonr(xs, ys).statistic:>16.3f}   "
            f"Spearman rho {stats.spearmanr(xs, ys).statistic:>16.3f}   "
            f"Kendall tau-b {stats.kendalltau(xs, ys).statistic:>16.3f}"
            f"  (permutation p {'< 0.001' if p_value < 0.001 else f'{p_value:.3f}'})")


# =============================================================================
# Majority labels
# =============================================================================


def stance(flags: set[str]) -> str:
    # A review marked both ways counts as contradicting.
    if "contradicting" in flags:
        return "contradict"
    if "supporting" in flags:
        return "support"
    return "neutral"


def majority_label(marks: dict[str, set[str]], annotators: list[str]) -> tuple[bool, str]:
    """(relevant, stance) of one review for one clause, by majority vote."""
    flags = [marks.get(annotator, set()) for annotator in annotators]
    relevant = sum("relevant" in f for f in flags) * 2 > len(flags)
    counts = Counter(stance(f) for f in flags)
    label, top = counts.most_common(1)[0]
    if list(counts.values()).count(top) > 1 or top * 2 <= len(flags):
        label = "neutral"
    return relevant, label


def review_clause_pairs(restaurant: dict):
    """(clause number, marks) for every annotated review x clause of a restaurant."""
    for review in restaurant["reviews"]:
        for number in range(1, len(restaurant["clauses"]) + 1):
            yield number, review["marks"].get(number, {})


def annotator_mean(restaurant: dict, score: str) -> float | None:
    """The 1-5 rating averaged over the annotators who gave one."""
    given = [values[score] for values in restaurant["annotators"].values() if score in values]
    return sum(given) / len(given) if given else None


# =============================================================================
# Reports
# =============================================================================


def heading(title: str, rule: str = "=") -> None:
    print("\n" + rule * 72)
    print(title)
    print(rule * 72)


def report_flag_agreement(settings: dict, restaurants: list[dict], flag: str) -> None:
    """Agreement on one flag over every review x clause. Supporting and contradicting
    are compared among the annotators who marked the review relevant."""
    annotators = settings["annotators"]
    gated = flag != "relevant"
    title = f"AGREEMENT ON `{flag}` PER REVIEW x CLAUSE"
    heading(title + ("  (ratings of those who marked it relevant)" if gated else ""))

    pairs = [marks for restaurant in restaurants for _, marks in review_clause_pairs(restaurant)]
    for annotator in annotators:
        marked = sum(flag in marks.get(annotator, set()) for marks in pairs)
        print(f"  {annotator}: {marked} units marked {flag}")

    units = []
    for marks in pairs:
        raters = [a for a in annotators if not gated or "relevant" in marks.get(a, set())]
        if len(raters) >= 2:
            units.append([int(flag in marks.get(a, set())) for a in raters])
    unanimous = sum(len(set(unit)) == 1 for unit in units) / len(units)
    print(f"\n  units:                          {len(units)}")
    print(f"  all annotators agree:           {unanimous:.1%}")
    print(f"  Krippendorff's alpha (nominal): {krippendorff_alpha(units):.3f}")
    print(f"  Gwet's AC1:                     {gwet_ac1(units):.3f}")


def report_time(settings: dict, restaurants: list[dict]) -> None:
    """Minutes each annotator spent per restaurant: reading the reviews, labelling
    every clause and giving the two 1-5 ratings."""
    heading("TIME SPENT PER ANNOTATOR")
    rows = []
    for annotator in settings["annotators"]:
        timed = [r for r in restaurants if r["annotators"][annotator].get("annotation_seconds")]
        total = sum(r["annotators"][annotator]["annotation_seconds"] / 60 for r in timed)
        reviews = sum(len(r["reviews"]) for r in timed)
        rows.append({"label": annotator, "restaurants": len(timed), "total": total,
                     "per_restaurant": total / len(timed), "per_review": 60 * total / reviews})
    count = len(rows)
    rows.append({"label": f"mean of {count} annotators", "restaurants": "",
                 **{key: sum(r[key] for r in rows) / count
                    for key in ("total", "per_restaurant", "per_review")}})

    width = max(len(r["label"]) for r in rows)
    print(f"\n  {'annotator':{width}}  {'restaurants':>11}  {'total':>8}  "
          f"{'min / restaurant':>16}  {'s / review':>10}")
    for r in rows:
        hours, minutes = divmod(round(r["total"]), 60)
        print(f"  {r['label']:{width}}  {r['restaurants']:>11}  "
              f"{f'{hours}h {minutes:02d}m':>8}  {r['per_restaurant']:16.1f}  "
              f"{r['per_review']:10.1f}")


def report_rating_agreement(settings: dict, restaurants: list[dict]) -> None:
    """Agreement on each 1-5 rating among the annotators who gave it."""
    heading("SUMMARY SCORES (1-5), AGREEMENT OVER ALL ANNOTATORS")
    by_name = sorted(restaurants, key=lambda r: r["name"])
    for score in SCORES:
        units = [[v[score] for v in r["annotators"].values() if score in v] for r in by_name]
        print(f"\n  {score.capitalize()} Score")
        for annotator in settings["annotators"]:
            given = [r["annotators"][annotator][score] for r in by_name
                     if score in r["annotators"][annotator]]
            print(f"    {annotator}: {len(given)} restaurants scored"
                  + (f", mean {sum(given) / len(given):.2f}" if given else ""))
        exact = sum(len(set(unit)) == 1 for unit in units) / len(units)
        within_one = sum(max(unit) - min(unit) <= 1 for unit in units) / len(units)
        ratings = [value for unit in units for value in unit]
        print(f"    restaurants scored by 2+ annotators: {len(units)}")
        print(f"    mean over all ratings: {sum(ratings) / len(ratings):.2f}")
        print(f"    all scores identical: {exact:.1%}   all within 1 point: {within_one:.1%}")
        print(f"    Krippendorff's alpha (interval): {krippendorff_alpha(units, interval_distance):.3f}")
        # AC1 over the five rating values: alpha is depressed when nearly every
        # rating is the same high value, which AC1 is built to withstand.
        print(f"    Gwet's AC1 (1-5 categories):     {gwet_ac1(units, (1, 2, 3, 4, 5)):.3f}")


def report_rating_vs_geval(settings: dict, restaurants: list[dict], score: str) -> None:
    """The annotators' mean 1-5 rating per restaurant against G-Eval."""
    heading(f"ANNOTATOR MEAN {score.upper()} SCORE vs THE G-EVAL {score.upper()} METRIC")
    print(f"  metric: geval_{score} / {settings['criterion']}  |  judge: {settings['judge']}")

    rows = sorted(({"name": r["name"], "mean": annotator_mean(r, score), "metric": r["geval"][score]}
                   for r in sorted(restaurants, key=lambda r: r["name"])),
                  key=lambda row: (row["metric"], row["name"]))
    width = max(len(row["name"]) for row in rows)
    header = f"  {'restaurant':{width}}  {'mean':>6}  {'G-Eval':>7}  {'diff':>6}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        print(f"  {row['name']:{width}}  {row['mean']:6.2f}  "
              f"{row['metric']:7.3f}  {row['mean'] - row['metric']:+6.2f}")
    print("  (sorted by the metric; mean = annotators' mean; diff = mean - metric, both on 1-5)")

    diffs = [row["mean"] - row["metric"] for row in rows]
    print(f"\n  difference, annotator mean - metric: mean {sum(diffs) / len(diffs):+.3f}, "
          f"mean absolute {sum(abs(d) for d in diffs) / len(diffs):.3f}")
    print("\n  correlation with the metric")
    print(correlation_line("mean of annotators", [row["mean"] for row in rows],
                           [row["metric"] for row in rows]))
    print(f"\n  The annotator mean takes {len({row['mean'] for row in rows})} distinct values "
          f"over {len(rows)} restaurants, which is little to rank on.")


def report_contradiction_burden(settings: dict, restaurants: list[dict]) -> None:
    """How strongly each restaurant's reviews contradict its summary, against G-Eval.

    Per aspect, the share of its supporting or contradicting reviews that
    contradict it; per restaurant, the mean over its aspects. Aspects without a
    single supporting or contradicting review are left out.
    """
    heading("CONTRADICTION BURDEN vs G-EVAL CONSISTENCY")
    rows, skipped = [], 0
    for restaurant in restaurants:
        tallies = defaultdict(Counter)
        for number, marks in review_clause_pairs(restaurant):
            relevant, label = majority_label(marks, settings["annotators"])
            if relevant and label != "neutral":
                tallies[number][label] += 1
        rates = []
        for clause in restaurant["clauses"]:
            tally = tallies.get(clause["number"])
            if not tally:
                skipped += 1
                continue
            rates.append(tally["contradict"] / (tally["contradict"] + tally["support"]))
        rows.append({"burden": sum(rates) / len(rates), "aspects": len(rates),
                     "metric": restaurant["geval"]["consistency"],
                     "human": annotator_mean(restaurant, "consistency")})

    burdens = [row["burden"] for row in rows]
    print(f"  restaurants: {len(rows)}")
    print(f"  aspects rated: {sum(row['aspects'] for row in rows)}"
          f" ({skipped} with no supporting or contradicting review skipped)")
    print(f"  burden: mean {sum(burdens) / len(burdens):.3f}, "
          f"{sum(b > 0 for b in burdens)} restaurants above 0")
    print("\n  correlation with the contradiction burden")
    print(correlation_line("G-Eval Consistency", burdens, [row["metric"] for row in rows]))
    print(correlation_line("annotator mean Consistency", burdens, [row["human"] for row in rows]))
    print("\n  A negative correlation means summaries their reviews contradict more "
          "score lower.")


def report_contradicted_aspects(settings: dict, restaurants: list[dict]) -> None:
    """Which summaries carry an aspect more than half of its relevant reviews contradict.

    Restaurants are ranked by G-Eval Consistency, lowest first, the order an
    auditor would read them in.
    """
    heading("CONTRADICTED ASPECTS", rule="#")
    print()
    annotators = settings["annotators"]
    ranked = sorted(restaurants, key=lambda r: (r["geval"]["consistency"], r["name"]))

    rows, flagged, contradict_not_relevant = [], [], 0
    for rank, restaurant in enumerate(ranked, start=1):
        relevant, contradicting, supporting = Counter(), Counter(), Counter()
        for number, marks in review_clause_pairs(restaurant):
            is_relevant, label = majority_label(marks, annotators)
            if is_relevant:
                relevant[number] += 1
                contradicting[number] += label == "contradict"
                supporting[number] += label == "support"
            elif label != "neutral":
                contradict_not_relevant += 1

        hits = []
        for clause in restaurant["clauses"]:
            number = clause["number"]
            rate = contradicting[number] / relevant[number] if relevant[number] else 0.0
            if rate <= THRESHOLD:
                continue
            hits.append((clause, relevant[number]))
            flagged.append({"rank": rank, "name": restaurant["name"], "clause": clause["text"],
                            "relevant": relevant[number], "contradict": contradicting[number],
                            "support": supporting[number], "rate": rate})

        n_reviews = len(restaurant["reviews"])
        on_flagged = sum(n_rel for _, n_rel in hits)
        seconds = [v["annotation_seconds"] for v in restaurant["annotators"].values()]
        rows.append({
            "rank": rank,
            "name": restaurant["name"],
            "geval": restaurant["geval"]["consistency"],
            "human_con": annotator_mean(restaurant, "consistency"),
            "human_rel": annotator_mean(restaurant, "relevance"),
            "normal": int(bool(hits)),
            "on_flagged": on_flagged if hits else None,
            "share": on_flagged / n_reviews if hits else None,
            "comment": "; ".join(f"{clause['text']} ({contradicting[clause['number']]}/{n_rel})"
                                 for clause, n_rel in hits),
            "clauses": len(restaurant["clauses"]),
            "reviews": n_reviews,
            "minutes": sum(s / 60 for s in seconds) / len(seconds),
        })

    print(f"Labels: majority of {len(annotators)} annotators. "
          f"Flag: > {THRESHOLD:.0%} of relevant reviews contradict an aspect.\n")
    print("G-Eval: G-Eval Consistency. H-Con, H-Rel: the annotators' mean Consistency and "
          "Relevance ratings (1-5).\n")
    header = (f"{'Rank':>4}  {'Restaurant':<32} {'G-Eval':>6} {'H-Con':>5} {'H-Rel':>5}  "
              f"{'>50%':>4} "
              f"{'#Rel':>4} {'%Rev':>6}  {'#Cl':>3} {'#Rev':>4} {'Min':>5}  Flagged aspects (con/rel)")
    print(header)
    print("-" * len(header))
    for row in rows:
        on_flagged = row["on_flagged"] if row["on_flagged"] is not None else ""
        share = format(row["share"], ".1%") if row["share"] is not None else ""
        print(f"{row['rank']:>4}  {row['name'][:32]:<32} {row['geval']:>6.3f} "
              f"{row['human_con']:>5.2f} {row['human_rel']:>5.2f}  "
              f"{row['normal']:>4} {on_flagged:>4} {share:>6}  "
              f"{row['clauses']:>3} {row['reviews']:>4} {row['minutes']:>5.1f}  {row['comment']}")
    print("-" * len(header))

    n = len(rows)
    shares = [r["share"] for r in rows if r["share"] is not None]
    total_minutes = sum(r["minutes"] for r in rows)
    print(f"{'':>4}  {'Total':<32} {'':>6} {'':>5} {'':>5}  {sum(r['normal'] for r in rows):>4} "
          f"{sum(r['on_flagged'] or 0 for r in rows):>4} "
          f"{'':>6}  {sum(r['clauses'] for r in rows):>3} {sum(r['reviews'] for r in rows):>4} "
          f"{total_minutes:>5.0f}")
    print(f"{'':>4}  {'Average':<32} {sum(r['geval'] for r in rows) / n:>6.3f} "
          f"{sum(r['human_con'] for r in rows) / n:>5.2f} "
          f"{sum(r['human_rel'] for r in rows) / n:>5.2f}  "
          f"{sum(r['normal'] for r in rows) / n:>4.2f} "
          f"{'':>4} {format(sum(shares) / len(shares), '.1%'):>6}  "
          f"{sum(r['clauses'] for r in rows) / n:>3.1f} {sum(r['reviews'] for r in rows) / n:>4.1f} "
          f"{total_minutes / n:>5.1f}")

    screen = int(round(SCREEN_SHARE * n))
    flagged_ranks = sorted({a["rank"] for a in flagged})
    print(f"\nAspects: {sum(len(r['clauses']) for r in restaurants)}")
    print(f"Majority-contradicted aspects: {len(flagged)}, "
          f"in {len(flagged_ranks)} restaurants")
    for a in flagged:
        print(f"  rank {a['rank']:>3}  {a['name'][:32]:<32}  \"{a['clause']}\"  "
              f"{a['contradict']} contradict / {a['support']} support / {a['relevant']} relevant "
              f"({a['rate']:.0%})")
    print(f"Lowest-scoring {SCREEN_SHARE:.0%} ({screen} summaries) contain "
          f"{sum(rank <= screen for rank in flagged_ranks)} of the {len(flagged_ranks)} "
          f"restaurants with a majority-contradicted aspect "
          f"(ranks {', '.join(map(str, flagged_ranks))})")
    if contradict_not_relevant:
        print(f"Note: {contradict_not_relevant} unit(s) have a majority support/contradict label "
              "but no relevance majority; they are not counted.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    settings, restaurants = load()
    for flag in FLAGS:
        report_flag_agreement(settings, restaurants, flag)
    report_time(settings, restaurants)
    report_rating_agreement(settings, restaurants)
    for score in SCORES:
        report_rating_vs_geval(settings, restaurants, score)
    report_contradiction_burden(settings, restaurants)
    print()
    report_contradicted_aspects(settings, restaurants)


if __name__ == "__main__":
    main()
