# Detecting Deception in Product Review Summaries

Code and data for the paper *Detecting Deception in Product Review Summaries* (under review at ACM FAccT).

## Abstract

Internet platforms increasingly deploy product review summaries to support consumer purchasing decisions. However, when a platform's own products compete with third-party offerings on its marketplace, there are incentives to deceive customers. At the same time, consumer protection enforcement agencies face difficulties detecting deception at scale due to the large volume of text requiring manual review. We evaluate whether reference-free summarization metrics can support two enforcement applications: market monitoring to identify platforms warranting further investigation, and ranking individual summaries to prioritize manual review.

We introduce controlled modifications into product review summaries across product categories, review sources, and summarization styles that are likely to be viewed as deceptive. These consist of factual errors, aspect deletions, changes to the stated prevalence of opinions, as well as ignoring the best and worst reviews during summary generation. After comparing alternative reference-free summarization metrics, we evaluate the best-performing metric, G-Eval, in two enforcement applications. In market-monitoring experiments, G-Eval Consistency achieves near-perfect detection when approximately 15% of summaries contain factual errors, with low false-positive rates across evaluation settings. For ranking of potentially deceptive summaries, Consistency scores effectively prioritize summaries containing factual errors by sorting them consistently lower. G-Eval is less reliable when detecting aspect deletions, changes to the stated prevalence of opinions, and review filtering across summarization styles, with G-Eval Relevance scores also producing false positives for certain styles, e.g., single-sentence summaries. A case study of 100 Google Maps restaurant summaries finds moderate correlations between G-Eval Consistency scores and human ratings and demonstrates that market-monitoring outcomes depend on the selection of underlying reviews.

Our findings show that regulators could benefit from using LLM-as-a-judge methods for both market monitoring and summary ranking, particularly for detecting factual errors. However, the metrics we evaluated were less effective at detecting other types of deception, especially omissions. Further research is required to address these limitations.

## Setup

```bash
uv sync
```

## Reproducing the results

To regenerate every figure, table and report:

```bash
uv run python run_all.py
```

This runs the steps below in order. Figures and tables are written to `results/figures/` and `results/tables/`, and the scripts' printed reports to `results/reports/`. Each step can also be run on its own:

| Result | Run |
|---|---|
| Score distributions of faithful and modified summaries; metric comparison on the hand-written edits | `src/plot_scores.ipynb` |
| Ranking modified summaries for manual review: rank correlation with the number of modifications, and yield at a fixed review budget | `src/manual_audit.ipynb` |
| Market monitoring: detection and false-positive rates over held-out styles, sources, categories and review counts | `src/market_monitoring.ipynb` |
| How much text the modifications change, by aspect count (appendix table) | `uv run python modification_validation/print_modifications_percentages.py` |
| Manual validation of the modifications: annotator agreement | `uv run python modification_validation/modification_validation_analysis.py` |
| Google Maps case study: platform verdict | `google_maps_experiment/google_maps_analysis.ipynb` |
| Google Maps case study: annotator agreement and annotations against G-Eval | `uv run python google_maps_experiment/google_maps_annotation_analysis.py` |

## Data

| File | Description |
|---|---|
| `data/reviews.json` | 240 products from Amazon, Yelp, Google Maps and HotelRec, with their 11,723 reviews (title, text, rating) in the order the summarizers saw them, and the settings used to select the products. |
| `data/summaries.jsonl` | Every scored text: 2,640 summaries and 36,890 modifications of them (36,590 generated, 300 written by hand). Summaries are the rows whose `record_id` equals `parent_summary_id`. |
| `data/scores.parquet` | Every metric score, keyed by `record_id`. `score_set` marks the generated panel (`main`), the hand-written edits (`manual`) and the LLM edits matched to them (`llm_matched`). G-Eval scores come from three judges, named in `result.judge`: `qwen36` (Qwen3.6-27B), used throughout the paper, and `gemma4` (Gemma 4 31B-IT) and `olmo31` (Olmo 3.1 32B-Instruct) for the judge comparison in the appendix; `result.model_revision` records the pinned weights and runtime. The other metrics of the metric comparison score only the hand-written edits, their parents and the matched LLM edits. |
| `modification_validation/modification_validation_dataset.json` | The manual validation of the modifications: 165 sampled modifications, each labelled by three anonymized annotators (A1–A3) as correct or incorrect. Fields are described in `modification_validation/README.md`. |
| `*/annotation_guidelines.pdf` | The guidelines given to the annotators of each study. |
| `prompts/` | The summarization, modification and G-Eval prompts. |
| `experiment.yaml` | The experiment configuration: models, conditions and metric settings. |

**Licensing.** The code is released under the MIT License (`LICENSE`). The review texts in `data/reviews.json` come from the Amazon Reviews, Yelp Open Dataset, Google Local (UCSD) and HotelRec datasets and remain subject to each source dataset's terms of use.

**Google Maps case study data.** `google_maps_experiment/google_maps_dataset.json` contains reviews and review summaries collected from Google Maps and is not part of the public release. It is available upon request for research use. The case-study notebook (`google_maps_analysis.ipynb`) and annotation analysis (`google_maps_annotation_analysis.py`) require this file and will not run without it.

## Repository structure

```
├── data/                     released data
├── src/                      analysis notebooks, the shared score loader (score_analysis.py) and pipeline/
├── modification_validation/  the modifications: validation data, guidelines and analyses
├── google_maps_experiment/   Google Maps case study: notebook, annotation analysis and guidelines
├── prompts/                  prompts used to generate, modify and score summaries
├── results/                  figures, tables and reports written by the analyses
├── run_all.py                regenerates everything in results/
└── experiment.yaml           models, conditions and metric settings
```

`src/pipeline/` documents how the data was produced and is not needed to reproduce the results. It reads and writes an earlier file layout, so it does not run against `data/`. Re-running it requires its own dependencies (`uv sync --extra pipeline`), API keys (see `.env.example`), and a GPU deployment on [Modal](https://modal.com). Its stages are the `generate_main`, `modify_main` and `score_main` functions in `src/pipeline/cli.py`, which read `experiment.yaml` from the working directory. The pipeline and `experiment.yaml` use the original names of the modifications: `factual_misrepresentation` (`misrepresent_*`), `omission` (`omit_*`) and `quantifier_change` (`quantifier_*`) for what the released data and the paper call factual errors, aspect deletions and quantifier swaps.

## Citation

Citation information will be added after review.
