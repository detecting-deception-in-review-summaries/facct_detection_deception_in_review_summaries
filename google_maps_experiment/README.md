# Google Maps case study

100 Google Maps restaurant review summaries with the reviews they summarize,
their G-Eval scores, and clause-level labels from three annotators (A1–A3).
The annotators followed `annotation_guidelines.pdf`.

| Analysis | Run |
|---|---|
| Platform verdict against the market-monitoring threshold | `google_maps_analysis.ipynb` |
| Annotator agreement, annotations against G-Eval, contradicted aspects | `uv run python google_maps_experiment/google_maps_annotation_analysis.py` |

## Review sets

Reviews were collected on `collection_date` (2026-04-26). G-Eval scored each
summary against three sets of reviews; each set keeps only reviews with text.

| Name | Reviews included |
|---|---|
| `all` | Every collected review. |
| `timestamp` | `published_at_date` on or after `timestamp_cutoff` (2025-04-26, twelve months before collection). |
| `time_label` | `timestamp`, plus reviews whose displayed label (`published_at`) reads "a year ago" or "Edited a year ago". The analyses use this set. |

## `google_maps_dataset.json`

Top level: `collection_date`, `timestamp_cutoff`, `judge` (the G-Eval model
and its pinned runtime), `annotation` (the annotators, and `metric_criterion`,
the review set whose G-Eval scores the analyses compare against), and
`places`. Each place has:

| Field | Meaning |
|---|---|
| `name`, `place_id` | The restaurant's Google Maps name and place ID. |
| `n_reviews` | Number of reviews in each review set. |
| `summary` | Google's review summary. |
| `geval` | G-Eval `consistency` and `relevance` score (1–5) under each review set. |
| `annotators` | Per annotator: `annotation_seconds` (time spent on the restaurant), and the 1–5 `consistency` and `relevance` ratings given by A1 and A2. |
| `clauses` | The summary split into numbered clauses. |
| `reviews` | The reviews G-Eval read or the annotators saw: `review_id`, `text`, `rating` (1–5 stars), `published_at` (the displayed relative label), `published_at_date`. |

Reviews in the `time_label` set were annotated and carry `clauses`: for each
clause number, the flags each annotator set, a subset of `relevant`,
`supporting` and `contradicting`, or `"not_relevant"` when the annotator found
the review unrelated to that clause. `["relevant"]` alone is the annotation
guide's "neutral": related to the clause, but neither supporting nor
contradicting it. Rating-only reviews without text were shown to
the annotators as well. `review_id` is the restaurant's position in the file
followed by the review's position, both counted from 001.

The analysis combines annotators by majority vote: a review is relevant to a
clause when most annotators marked it so; among relevant reviews, the stance
is the label most annotators gave (support, contradict or neutral), with a tie
going to neutral.
