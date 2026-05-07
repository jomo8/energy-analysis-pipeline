# AMZN → KO Migration — Design Document

## Goal

Switch the energy-analysis pipeline's primary ticker from AMZN to KO (Coca-Cola Co., CIK `0000021344`), produce a complete fresh set of artifacts and results for KO, and update narrative documentation to reflect the change. AMZN compatibility is **not** preserved — the AMZN history lives in git and that is sufficient.

## Why

AMZN's FNSPID coverage is concentrated in 2020 (n=278) and 2023 (n=4,782) with no headlines in 2010–2019 or 2021–2022. KO has 10,521 FNSPID headlines spanning 2009–2023 continuously, with 500–1,300 per year in mature years. This unlocks a real 14-year evaluation window with informative NLP features throughout, rather than the ~3-month effective window AMZN produces.

## Non-goals

- Maintaining AMZN as a parallel evaluation target. AMZN artifacts and results stay in git history; we don't run them in parallel.
- Refactoring the pipeline to be ticker-pluggable. Single ticker per branch is fine.
- Rewriting methodology prose for the thesis-extension version. We make this functional and document the dataset accurately; deeper methodology revisions come later.
- Re-tuning hyperparameters. All NLP and regression hyperparameters stay at current values.

## Prerequisites confirmed

- KO CIK verified via SEC API (`COCA COLA CO`, CIK 21344). 4.8 MB facts JSON pulled cleanly.
- EDGAR concept availability confirmed via direct inspection of KO's facts JSON (see "Concept mapping" below).
- FNSPID KO coverage confirmed via per-ticker scan (10,521 rows, continuous 2009–2023).
- yfinance KO data confirmed available. `load_prices_from_yfinance` will work without changes.
- Pipeline architecture confirmed ticker-parameterized via single `CONFIG["company"]` block in `common.py`.

## Concept mapping (KO vs AMZN)

The two tickers file revenue under different XBRL tags. KO does not file `us-gaap:Liabilities` directly — same as AMZN — so the identity-reconstruction logic still applies.

| Concept | AMZN | KO | Action |
|---|---|---|---|
| `Revenues` | Not filed | 2016–2026, 100 obs | **ADD** to concept list |
| `SalesRevenueGoodsNet` | Not filed | 2007–2018, 117 obs (pre-2016 bridge) | **ADD** to concept list |
| `RevenueFromContractWithCustomerExcludingAssessedTax` | 2018+ | Not filed | **REMOVE** |
| `SalesRevenueNet` | Pre-2018 | Not filed | **REMOVE** |
| `Liabilities` | Not filed | Not filed | Keep identity reconstruction |
| `Assets` | OK | OK | No change |
| `StockholdersEquity` | OK | OK | No change |
| `NetIncomeLoss` | OK | OK | No change |
| `OperatingIncomeLoss` | OK | OK | No change |
| `LongTermDebt` | OK | OK | No change |
| `CashAndCashEquivalentsAtCarryingValue` | OK | OK | No change |
| `EarningsPerShareBasic` | OK | Sparse (4 obs) | Keep — not used in feature engineering |

## Functional changes

Three locations. All edits are surgical.

### 1. `common.py` CONFIG block

Inside `CONFIG["company"]`:
- `"ticker"`: `"AMZN"` → `"KO"`
- `"cik"`: `"0001018724"` → `"0000021344"`
- `"start_date"`: `"2010-01-01"` → `"2009-01-01"` (capture KO's earliest FNSPID coverage year)

Inside `CONFIG["edgar"]["concepts"]`:
- Remove: `RevenueFromContractWithCustomerExcludingAssessedTax`, `SalesRevenueNet`
- Add: `Revenues`, `SalesRevenueGoodsNet`

All other CONFIG values stay unchanged.

### 2. `01_data_ingestion.ipynb` — `build_num_df` Revenues coalescing block

Current logic uses `combine_first` to merge `RevenueFromContractWithCustomerExcludingAssessedTax` (post-ASC-606) onto `SalesRevenueNet` (pre-ASC-606).

Replace with a coalesce of `Revenues` (KO's modern tag, 2016+) onto `SalesRevenueGoodsNet` (KO's pre-2016 bridge). Same `combine_first` pattern, different inputs. The output column name stays `Revenues` so all downstream ratio code (`profit_margin = NetIncomeLoss / Revenues`) is unchanged.

Defensive logic: handle the case where one of the two tags is missing from `merged.columns` (e.g., `SalesRevenueGoodsNet` could be 404'd if SEC's response shape changes). Mirror the existing AMZN logic that gracefully picks whichever is present.

### 3. `01_data_ingestion.ipynb` — `build_num_df` Liabilities comment

The accounting-identity reconstruction `Liabilities = Assets - StockholdersEquity` stays as-is. The inline comment above it currently explains the workaround in AMZN-specific terms. Update the comment to reflect that KO (like AMZN) doesn't file `us-gaap:Liabilities` directly, so the identity is used. Functional code unchanged.

## Cosmetic changes (separate pass — do last, after results validate)

- Markdown headers across notebooks: "AMZN" → "KO" (~10 occurrences)
- Inline comments mentioning AMZN by name (~6 occurrences across notebooks)
- Stage 7 visualization titles: 6 chart titles with "AMZN" hardcoded
- Stage 1 markdown caveats section: rewrite the FNSPID coverage limitation paragraph to reflect KO's continuous 2009–2023 coverage instead of AMZN's gap

## Validation checkpoints

After each stage, before proceeding, confirm:

### After Stage 1
- `text_df.parquet`: ~10,000 rows, continuous date range starting 2009 through 2023, year-by-year histogram has no zero years
- `num_df.parquet`: rows present across full 2009–2023 window, `Revenues` column non-null on the vast majority of rows, `profit_margin` and `current_ratio` numerically reasonable (single-digit ratios, not exploding)
- Weak-label threshold sweep converges (no "factor not found" error), final class shares within reason on train slice
- Shared chronological cutoff lands somewhere around 2022 (i.e., ~80% through 2009–2023), not at a pathological corner

### After Stage 2
- Tokenization stats: nothing truncated at 128 tokens
- Train/test splits agree on text and numerical sides

### After Stage 3
- All 40 (model × seed) runs complete without OOM or NaN
- F1 scores reasonable (>0.4 weighted F1 is the floor; <0.3 means something's wrong)

### After Stages 4–7
- Standard pipeline outputs as before
- Stage 7 plots show recognizably KO output

## Sequencing

1. Edit `common.py` and `01_data_ingestion.ipynb` (functional changes only). Commit with message like *"Switch primary ticker to KO; update revenue concept tags."*
2. Run Stage 1. Validate against checkpoint criteria above. If fails, debug here before going further.
3. Run Stage 2. Validate.
4. Submit Stage 3 as batch job (multi-hour run). Use `sbatch` with appropriate time/mem/GPU; let it run unattended.
5. While Stage 3 runs: do the cosmetic markdown sweep. Independent of the data pipeline; doesn't risk breaking anything.
6. Run Stages 4–7 sequentially after Stage 3 completes. Each is fast.
7. Final validation: open Stage 7 notebook, look at the plots.
8. Commit and push.

## Known risks and mitigations

- **`Revenues` tag has 2009–2015 gap until `SalesRevenueGoodsNet` covers it.** Verified via JSON inspection that `SalesRevenueGoodsNet` covers 2007–2018, so the bridge is solid. If post-coalesce `Revenues` column has unexpected nulls, the Stage 1 validation checkpoint catches it.
- **Weak-label threshold may converge differently for KO than AMZN.** KO is a low-volatility consumer staple vs. AMZN's high-volatility tech profile. The volatility-scaled threshold sweep should adapt automatically. If the loop exhausts without finding a factor that hits the 10% per-class share constraint, the existing error message fires and we widen the sweep range. Low risk; the sweep currently spans 0.5–2.0.
- **Stage 3 wall-clock time roughly doubles** vs. AMZN due to ~2× headline count. Mitigated by running as batch job. If lab compute access is constrained, this is the bottleneck — start Stage 3 as early as possible.
- **EDGAR cache cross-contamination.** Cache files are keyed by CIK, so KO and AMZN files coexist cleanly in `artifacts/edgar_cache/`. No risk.

## Done criteria

- All seven stages run end-to-end with KO data, producing `nlp_results.csv`, `regression_results.csv`, `inference_results.csv`, `nlp_probs.parquet`, all four parquet artifacts, and Stage 7 plots.
- Stage 7 plots show recognizably KO output (titles updated, year coverage visible).
- Functional code edits committed; cosmetic sweep committed separately.

## Environment reference

- HPC venv: `/cluster/tufts/hrilab/jmonta04/.venvs/energy-pipeline`
- Project root: `/cluster/tufts/hrilab/jmonta04/modular_pipeline/`
- HF cache: `/cluster/tufts/hrilab/jmonta04/.hf_cache` (`HF_HOME` env var must be set)
- Artifacts dir: `./artifacts/` (relative to project root)
- Results dir: `./results/` (relative to project root)
