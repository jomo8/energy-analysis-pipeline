"""Shared configuration, registries, and helper utilities for all stages.

Every notebook imports this module first so that:
- Paths and config values stay consistent.
- Model registries are defined in one place.
- Data split and artifact helpers behave the same across stages.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import torch
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Keep notebook output focused on key messages.
warnings.filterwarnings("ignore")

# ----- paths ---------------------------------------------------------------
# Project root where notebooks are executed.
ROOT = Path(".").resolve()
# Shared directory for intermediate parquet/json artifacts.
ARTIFACTS_DIR = ROOT / "artifacts"
# Shared directory for CSV summaries and plots.
RESULTS_DIR = ROOT / "results"
# Cache directory for SEC API responses.
EDGAR_CACHE_DIR = ARTIFACTS_DIR / "edgar_cache"

ARTIFACTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
EDGAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ----- device --------------------------------------------------------------
# Prefer GPU when available; fall back to CPU otherwise.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- labels --------------------------------------------------------------
# Consistent mapping used across training, inference, and saved artifacts.
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

# ----- experiment configuration --------------------------------------------
CONFIG = {
    # Seeds used for repeated training runs.
    "random_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # Target company and analysis window.
    "company": {
        "ticker": "AMZN",
        "cik": "0001018724",
        "start_date": "2010-01-01",
        "end_date": "2023-12-31",
    },
    # EDGAR settings: user agent + concept list to pull.
    "edgar": {
        "user_agent": "Joseph Montalto joseph.montalto@tufts.edu",
        "concepts": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "NetIncomeLoss",
            "Assets",
            "StockholdersEquity",
            "OperatingIncomeLoss",
            "CashAndCashEquivalentsAtCarryingValue",
            "LongTermDebt",
            "EarningsPerShareBasic",
        ],
    },
    # Rules used to create weak labels from future returns.
    "labeling": {
        "method": "return_sign",
        "horizon_days": 1,
    },
    # Hyperparameters shared by all NLP fine-tuning runs.
    "nlp": {
        "max_length": 128,
        "train_batch_size": 16,
        "eval_batch_size": 32,
        "epochs": 3,
        "learning_rate": 2e-5,
        "num_labels": 3,
        "dataloader_num_workers": 4,
    },
    # Settings for regression stage and NLP feature lookback window.
    "regression": {
        "test_size": 0.2,
        "nlp_lookback_days": 5,
        "n_synthetic_samples": 5000
    },
    # Settings for deployment-style inference benchmark.
    "inference": {
        "n_samples": 1000,
        "batch_size": 32,
    },
    # Placeholder lambda scenarios for tradeoff scoring.
    "lambda_scenarios": {
        "research": 10,
        "moderate": 1_000,
        "production": 100_000,
    },
}

# Registry of transformer checkpoints to compare in Stage 3/5.
NLP_MODELS = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT-base": "bert-base-uncased",
    "RoBERTa-base": "roberta-base",
    "FinBERT": "ProsusAI/finbert",
    # "GPT-2 Small": "gpt2",
    # "GPT-2 Medium": "gpt2-medium",
}

# Registry of baseline and tree-based regressors for Stage 4.
REGRESSION_MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(),
    "Lasso": lambda: Lasso(),
    "Elastic Net": lambda: ElasticNet(),
    "XGBoost": lambda seed: xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        verbosity=0,
    ),
    "LightGBM": lambda seed: lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        verbose=-1,
    ),
    "CatBoost": lambda seed: CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=seed,
        verbose=0,
    ),
}


# ----- SEC EDGAR helpers ---------------------------------------------------
_SEC_LAST_REQUEST_TS = 0.0


def _sec_throttle() -> None:
    """Respect SEC's published <= 10 requests / second limit."""
    global _SEC_LAST_REQUEST_TS

    # Compute elapsed time since the previous SEC request.
    elapsed = time.time() - _SEC_LAST_REQUEST_TS
    # Sleeping ~0.11s keeps us below 10 requests/sec.
    minimum_interval = 0.11
    if elapsed < minimum_interval:
        time.sleep(minimum_interval - elapsed)
    # Record request time after throttling so next call can enforce spacing.
    _SEC_LAST_REQUEST_TS = time.time()


def build_sec_session() -> requests.Session:
    """Create a Session with the required SEC user-agent header."""
    session = requests.Session()
    session.headers.update({"User-Agent": CONFIG["edgar"]["user_agent"]})
    return session


def _sanitize_name(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _edgar_cache_path(cik: str, concept: str | None = None, kind: str = "concept") -> Path:
    safe_cik = _sanitize_name(cik)
    if concept is None:
        file_name = f"{safe_cik}_{kind}.json"
    else:
        safe_concept = _sanitize_name(concept)
        file_name = f"{safe_cik}_{safe_concept}_{kind}.json"
    return EDGAR_CACHE_DIR / file_name


def _sec_get_json(url: str, cache_path: Path) -> dict:
    """Read cached SEC JSON if present, otherwise fetch and cache it."""
    # Use local cache first to avoid repeated network calls and rate limits.
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # Build a fresh session with SEC-compliant headers.
    session = build_sec_session()
    # Enforce SEC request pacing.
    _sec_throttle()
    response = session.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    # Persist raw payload so reruns can reuse it.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return payload


def get_company_concept(cik: str, concept: str) -> dict:
    """Fetch one concept payload from SEC and cache it locally."""
    url = (
        f"https://data.sec.gov/api/xbrl/companyconcept/"
        f"CIK{cik}/us-gaap/{concept}.json"
    )
    cache_path = _edgar_cache_path(cik=cik, concept=concept, kind="concept")
    return _sec_get_json(url=url, cache_path=cache_path)


def get_company_facts(cik: str) -> dict:
    """Fetch full company facts payload from SEC and cache it locally."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    cache_path = _edgar_cache_path(cik=cik, concept=None, kind="facts")
    return _sec_get_json(url=url, cache_path=cache_path)


# ----- results collector ---------------------------------------------------
class ResultsCollector:
    """Accumulates per-run records for each stage and writes them to CSV."""

    def __init__(self):
        # Each list stores dictionaries collected in that stage.
        self.nlp_results: list[dict] = []
        self.regression_results: list[dict] = []
        self.inference_results: list[dict] = []

    def add_nlp(self, record: dict):
        self.nlp_results.append(record)

    def add_regression(self, record: dict):
        self.regression_results.append(record)

    def add_inference(self, record: dict):
        self.inference_results.append(record)

    def nlp_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.nlp_results)

    def regression_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.regression_results)

    def inference_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.inference_results)

    def save(self, path=RESULTS_DIR):
        # Ensure output directory exists before writing any CSV files.
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        # Write one CSV per stage if that stage produced any rows.
        for name in ("nlp", "regression", "inference"):
            frame = getattr(self, f"{name}_df")()
            if not frame.empty:
                frame.to_csv(output_path / f"{name}_results.csv", index=False)
        print(f"Results saved to {output_path}/")


# ----- dataset + split helpers --------------------------------------------
class SentimentDataset(Dataset):
    """PyTorch Dataset wrapping tokenized text + labels."""

    def __init__(self, texts, labels, tokenizer, max_length):
        # Tokenize all texts once so __getitem__ is lightweight.
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        # Trainer expects integer class IDs as torch tensors.
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def get_shared_chronological_cutoff(
    text_df: pd.DataFrame,
    num_df: pd.DataFrame,
    test_size: float = 0.2,
    text_date_col: str = "date",
    num_date_col: str = "date",
) -> pd.Timestamp:
    """Choose one cutoff date shared by text and numerical data."""
    # Normalize to date-only values so time-of-day differences do not matter.
    text_dates = pd.to_datetime(text_df[text_date_col]).dt.normalize()
    num_dates = pd.to_datetime(num_df[num_date_col]).dt.normalize()

    # We split only over overlapping dates so both modalities share test period.
    shared_dates = np.array(sorted(set(text_dates).intersection(set(num_dates))))
    if len(shared_dates) < 5:
        raise ValueError("Not enough overlapping dates to build a shared split cutoff.")

    # Convert test_size into the final training index.
    split_index = int(np.floor((1.0 - test_size) * len(shared_dates))) - 1
    split_index = max(split_index, 0)
    return pd.Timestamp(shared_dates[split_index])


def chronological_split_by_date(
    df: pd.DataFrame,
    date_col: str,
    cutoff_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test dataframes using <= cutoff for train and > cutoff for test."""
    # Work on a copy so caller data is not modified in-place.
    dated = df.copy()
    # Normalize all timestamps to pure dates.
    dated[date_col] = pd.to_datetime(dated[date_col]).dt.normalize()
    # Sort before splitting to guarantee chronological order.
    dated = dated.sort_values(date_col).reset_index(drop=True)

    # Training includes the cutoff date; testing starts strictly after it.
    train_df = dated[dated[date_col] <= cutoff_date].copy()
    test_df = dated[dated[date_col] > cutoff_date].copy()
    return train_df, test_df


def make_text_splits_chronological(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    cutoff_date: pd.Timestamp,
    date_col: str = "date",
) -> tuple[SentimentDataset, SentimentDataset, pd.DataFrame, pd.DataFrame]:
    """Build chronological train/test text datasets and return both raw splits too."""
    # Reuse the shared split rule to avoid leakage.
    train_df, test_df = chronological_split_by_date(df=df, date_col=date_col, cutoff_date=cutoff_date)
    # Convert each split into a Trainer-ready dataset object.
    train_ds = SentimentDataset(train_df["text"], train_df["label"], tokenizer, max_length)
    test_ds = SentimentDataset(test_df["text"], test_df["label"], tokenizer, max_length)
    return train_ds, test_ds, train_df, test_df


def make_num_splits_chronological(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    target_col: str = "risk_score",
    date_col: str = "date",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str], pd.DataFrame, pd.DataFrame]:
    """Scale and split numerical features chronologically."""
    # Split by date first so scaler is fit only on training rows.
    train_df, test_df = chronological_split_by_date(df=df, date_col=date_col, cutoff_date=cutoff_date)

    # All columns except date and target are input features.
    feature_cols = [
        col
        for col in train_df.columns
        if col not in {target_col, date_col}
    ]

    # Build feature matrices and target vectors.
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    # Fit scaler only on training data to avoid test leakage.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_cols,
        train_df,
        test_df,
    )


def attach_nlp_prob_features(
    num_df: pd.DataFrame,
    nlp_probs_df: pd.DataFrame,
    lookback_days: int = 5,
    date_col: str = "date",
) -> pd.DataFrame:
    """Attach rolling K-day average NLP class probabilities to numerical rows."""
    # If NLP probabilities are unavailable, fall back to uniform sentiment priors.
    if nlp_probs_df.empty:
        augmented = num_df.copy()
        augmented["nlp_prob_negative"] = 1.0 / 3.0
        augmented["nlp_prob_neutral"] = 1.0 / 3.0
        augmented["nlp_prob_positive"] = 1.0 / 3.0
        return augmented

    # Normalize dates so joins align exactly by day.
    probs = nlp_probs_df.copy()
    probs["date"] = pd.to_datetime(probs["date"]).dt.normalize()

    # Aggregate per-headline probabilities into per-day totals.
    daily = probs.groupby("date").agg(
        sum_negative=("prob_negative", "sum"),
        sum_neutral=("prob_neutral", "sum"),
        sum_positive=("prob_positive", "sum"),
        n_headlines=("prob_negative", "size"),
    )

    # Reindex to daily calendar so rolling windows behave consistently.
    full_dates = pd.date_range(
        start=min(pd.to_datetime(num_df[date_col]).min(), daily.index.min()),
        end=max(pd.to_datetime(num_df[date_col]).max(), daily.index.max()),
        freq="D",
    )
    daily = daily.reindex(full_dates).fillna(0.0)

    # Rolling sums over lookback window.
    rolling = daily.rolling(window=lookback_days, min_periods=1).sum()
    rolling = rolling.rename_axis("date").reset_index()

    # Divide by headline count to get mean probabilities; NaN means no headlines.
    safe_count = rolling["n_headlines"].replace(0, np.nan)
    rolling["nlp_prob_negative"] = rolling["sum_negative"] / safe_count
    rolling["nlp_prob_neutral"] = rolling["sum_neutral"] / safe_count
    rolling["nlp_prob_positive"] = rolling["sum_positive"] / safe_count

    # When no headlines are present, default to neutral prior 1/3 each.
    rolling["nlp_prob_negative"] = rolling["nlp_prob_negative"].fillna(1.0 / 3.0)
    rolling["nlp_prob_neutral"] = rolling["nlp_prob_neutral"].fillna(1.0 / 3.0)
    rolling["nlp_prob_positive"] = rolling["nlp_prob_positive"].fillna(1.0 / 3.0)

    # Merge rolling NLP features onto numerical rows by date.
    augmented = num_df.copy()
    augmented[date_col] = pd.to_datetime(augmented[date_col]).dt.normalize()

    rolling_features = rolling.rename(columns={"date": date_col})[
        [date_col, "nlp_prob_negative", "nlp_prob_neutral", "nlp_prob_positive"]
    ]
    augmented = augmented.merge(
        rolling_features,
        on=date_col,
        how="left",
    )
    return augmented


# ----- metrics -------------------------------------------------------------
def compute_clf_metrics(eval_pred):
    """Compute common classification metrics from Trainer eval output."""
    logits, labels = eval_pred
    # Pick the class with highest logit score for each example.
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# ----- artifact I/O helpers -----------------------------------------------
def load_text_df() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "text_df.parquet")


def load_num_df() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "num_df.parquet")


def load_price_df() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "price_df.parquet")


def load_edgar_df() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "edgar_df.parquet")


def load_nlp_probs_df() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "nlp_probs.parquet")


def save_text_df(df: pd.DataFrame):
    df.to_parquet(ARTIFACTS_DIR / "text_df.parquet", index=False)


def save_num_df(df: pd.DataFrame):
    df.to_parquet(ARTIFACTS_DIR / "num_df.parquet", index=False)


def save_price_df(df: pd.DataFrame):
    df.to_parquet(ARTIFACTS_DIR / "price_df.parquet", index=False)


def save_edgar_df(df: pd.DataFrame):
    df.to_parquet(ARTIFACTS_DIR / "edgar_df.parquet", index=False)


def save_nlp_probs_df(df: pd.DataFrame):
    df.to_parquet(ARTIFACTS_DIR / "nlp_probs.parquet", index=False)


print(
    f"[common] device={DEVICE}  artifacts={ARTIFACTS_DIR}  results={RESULTS_DIR}"
)
