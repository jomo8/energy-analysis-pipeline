# Accuracy–Energy Tradeoffs Pipeline

Modular version of the original `pipeline_revised_1__1_.ipynb`, split
into one notebook per pipeline stage plus a shared `common.py` library.

## Layout

```
common.py                       shared config, model registries, helpers
01_data_ingestion.ipynb         Stage 1: load text + generate numerical
02_preprocessing.ipynb          Stage 2: tokenization stats, split checks
03_sentiment_nlp.ipynb          Stage 3: fine-tune NLP models
04_regression.ipynb             Stage 4: train regression models
05_deployment_inference.ipynb   Stage 5: inference benchmarking
06_scoring_ranking.ipynb        Stage 6: aggregate + ANOVA/Tukey
07_visualization.ipynb          Stage 7: final plots + Pareto
artifacts/                      parquet files shared between stages (auto-created)
results/                        CSVs, plots, experiment_config.json (auto-created)
```

All notebooks must live in the same directory as `common.py` so that
`from common import *` works.

## How stages communicate

Notebooks don't import each other — they pass data through disk:

- Stage 1 writes `artifacts/text_df.parquet` and `artifacts/num_df.parquet`.
- Stages 3–5 read those artifacts and write `results/*.csv`.
- Stages 6–7 read the CSVs and produce plots + summaries.

Every notebook's first code cell is `from common import *`, which loads
the config, sets `DEVICE`, creates the `artifacts/` and `results/`
directories, and exposes the shared helpers and model registries.

## Running order

```
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07
```

Stages 3, 4, and 5 only depend on Stage 1's artifacts, so once Stage 1
has run they can be re-run in any order. Stages 6 and 7 only need the
CSVs produced by Stages 3–5. Stage 2 is diagnostic only — it validates
tokenization length assumptions and sanity-checks the split helpers but
does not produce artifacts the later stages depend on, so you can skip
it on re-runs.

## Where to plug in energy measurement

The timing code in `03_sentiment_nlp.ipynb`, `04_regression.ipynb`, and
`05_deployment_inference.ipynb` already brackets training and inference
with `t0 = time.time()`. To add CodeCarbon / NVML / RAPL readings, wrap
the same brackets with your energy tracker and add `energy_train_kwh` /
`energy_infer_kwh` keys to the `record` dicts. No downstream stage has
to change — and Stage 7's Pareto plot becomes a true energy–accuracy
Pareto just by swapping its `x_col` argument.

## Note on a fix carried over from the original

The original Stage 2 cell in `pipeline_revised_1__1_.ipynb` contained:

```python
text_df = ds["train"].to_pandas()
text_df = ds["test"].to_pandas()   # silently overwrites
```

which threw away the combined DataFrame built earlier in the notebook
and left only the test split for downstream use. In this modular
version `01_data_ingestion.ipynb` concatenates every split exactly once
and is the only place `text_df` is built, so every downstream stage
sees the full corpus.
