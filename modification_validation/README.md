# Modification validation

Manual validation of the controlled modifications: 165 modified summaries,
stratified by summarization style, each labelled by three annotators (A1–A3)
as a correct or incorrect application of the requested modification. The
annotators followed `annotation_guidelines.pdf`.

Run the agreement analysis with:

```bash
uv run python modification_validation/modification_validation_analysis.py
```

## `modification_validation_dataset.json`

`annotators` lists the annotators; `items` holds one entry per modification:

| Field | Meaning |
|---|---|
| `source` | Review source of the product (`amazon`, `yelp`, `google_maps`, `hotel_rec`). |
| `style` | Summarization style of the parent summary, as in `data/summaries.jsonl`. |
| `family` | Requested modification: `factual_error`, `aspect_deletion` or `quantifier_swap`. |
| `direction` | `advantageous` or `disadvantageous` for one aspect; `mixed` for several. |
| `number_of_aspects` | How many aspects the modifier was asked to change (1–4). |
| `declined` | `true` when the modifier declined the request because the summary could not be modified as asked. Annotators then judged whether declining was correct. |
| `diff` | The modified summary against its parent, in the notation shown to the annotators: `[-text]` was removed and `[+text]` was added. For declined requests, the unchanged parent summary. |
| `labels` | Each annotator's label: `1` = correct, `0` = incorrect, `null` = unsure or left empty. |
