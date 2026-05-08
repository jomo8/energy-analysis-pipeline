"""Join per-second CPU/GPU power samples to per-run results CSVs.

Reads the energy logger CSV (timestamp_iso, cpu_energy_j, gpu_power_w, gpu_util_pct)
and each stage's results CSV (which has wall-clock ISO start/end timestamps), then
computes per-run energy in kWh by integrating power over each run's window.

Outputs *_with_energy.csv files alongside the originals.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
ARTIFACTS = REPO_ROOT / "artifacts"
ENERGY_CSV = REPO_ROOT / "lab_machine_pkg" / "cpu_gpu_power_20260507_182332.csv"


def load_energy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp_iso"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    # CPU energy is a monotonic counter (joules). Convert to per-sample power
    # by differencing and dividing by elapsed seconds. The first row gets a
    # 0 power reading since there's no prior sample to diff against.
    dt_s = df["timestamp"].diff().dt.total_seconds()
    cpu_dj = df["cpu_energy_j"].diff()
    # RAPL counters wrap around their max value. If we see a negative diff,
    # treat as a wrap and use the next sample's power as a fallback.
    cpu_dj = cpu_dj.where(cpu_dj >= 0, np.nan)
    df["cpu_power_w"] = (cpu_dj / dt_s).fillna(0.0)
    return df


def integrate_power(energy_df: pd.DataFrame, start_iso: str, end_iso: str) -> dict:
    """Integrate cpu and gpu power over [start, end]. Returns kWh values."""
    start = pd.to_datetime(start_iso, utc=True)
    end = pd.to_datetime(end_iso, utc=True)
    if pd.isna(start) or pd.isna(end) or end <= start:
        return {"cpu_kwh": np.nan, "gpu_kwh": np.nan, "duration_s": np.nan, "n_samples": 0}

    mask = (energy_df["timestamp"] >= start) & (energy_df["timestamp"] <= end)
    window = energy_df.loc[mask].copy()
    if len(window) < 2:
        # Fallback: use the average power across samples bracketing the window.
        bracket = energy_df[
            (energy_df["timestamp"] >= start - pd.Timedelta("2s"))
            & (energy_df["timestamp"] <= end + pd.Timedelta("2s"))
        ]
        if len(bracket) == 0:
            return {"cpu_kwh": np.nan, "gpu_kwh": np.nan,
                    "duration_s": (end - start).total_seconds(), "n_samples": 0}
        duration_h = (end - start).total_seconds() / 3600.0
        return {
            "cpu_kwh": bracket["cpu_power_w"].mean() * duration_h / 1000.0,
            "gpu_kwh": bracket["gpu_power_w"].mean() * duration_h / 1000.0,
            "duration_s": (end - start).total_seconds(),
            "n_samples": len(bracket),
        }

    # Trapezoidal integration. timestamps are seconds since the window start.
    t = (window["timestamp"] - window["timestamp"].iloc[0]).dt.total_seconds().values
    cpu_w = window["cpu_power_w"].values
    gpu_w = window["gpu_power_w"].values
    cpu_j = np.trapz(cpu_w, t)
    gpu_j = np.trapz(gpu_w, t)
    return {
        "cpu_kwh": cpu_j / 3_600_000.0,
        "gpu_kwh": gpu_j / 3_600_000.0,
        "duration_s": (end - start).total_seconds(),
        "n_samples": int(len(window)),
    }


def join_csv(results_csv: Path, energy_df: pd.DataFrame, phase_pairs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """phase_pairs is a list of (label, start_col, end_col) tuples."""
    df = pd.read_csv(results_csv)
    for label, start_col, end_col in phase_pairs:
        if start_col not in df.columns or end_col not in df.columns:
            print(f"  skipping {label}: missing columns ({start_col}, {end_col})")
            continue
        cpu, gpu, dur, n = [], [], [], []
        for _, row in df.iterrows():
            r = integrate_power(energy_df, row[start_col], row[end_col])
            cpu.append(r["cpu_kwh"])
            gpu.append(r["gpu_kwh"])
            dur.append(r["duration_s"])
            n.append(r["n_samples"])
        df[f"energy_{label}_cpu_kwh"] = cpu
        df[f"energy_{label}_gpu_kwh"] = gpu
        df[f"energy_{label}_kwh"] = np.array(cpu) + np.array(gpu)
        df[f"energy_{label}_duration_s"] = dur
        df[f"energy_{label}_n_samples"] = n
    return df


def main():
    if not ENERGY_CSV.exists():
        sys.exit(f"Energy CSV not found at {ENERGY_CSV}")

    print(f"Loading energy samples from {ENERGY_CSV.name}...")
    energy_df = load_energy(ENERGY_CSV)
    print(f"  {len(energy_df):,} samples spanning {energy_df['timestamp'].iloc[0]} -> {energy_df['timestamp'].iloc[-1]}")
    print(f"  mean GPU power: {energy_df['gpu_power_w'].mean():.1f} W")
    print(f"  mean CPU power: {energy_df['cpu_power_w'].mean():.1f} W")
    print()

    # Stage 3 — NLP fine-tuning. Has both train and inference windows.
    print("Joining nlp_results.csv...")
    nlp = join_csv(
        RESULTS / "nlp_results.csv",
        energy_df,
        [("train", "wall_train_start_iso", "wall_train_end_iso"),
         ("infer", "wall_infer_start_iso", "wall_infer_end_iso")],
    )
    nlp.to_csv(RESULTS / "nlp_results_with_energy.csv", index=False)
    print(f"  wrote {RESULTS / 'nlp_results_with_energy.csv'}")
    print(nlp[["model", "seed", "energy_train_kwh", "energy_infer_kwh"]].head())
    print()

    # Stage 4 — regression. Each row has fit_start/fit_end and pred_start/pred_end
    # (column names may vary; we'll discover at runtime).
    print("Joining regression_results.csv...")
    reg = pd.read_csv(RESULTS / "regression_results.csv")
    print(f"  columns: {list(reg.columns)}")
    fit_cols = [c for c in reg.columns if "iso" in c.lower()]
    print(f"  ISO timestamp columns found: {fit_cols}")
    pairs = []
    if "wall_train_start_iso" in reg.columns and "wall_train_end_iso" in reg.columns:
        pairs.append(("train", "wall_train_start_iso", "wall_train_end_iso"))
    if "wall_infer_start_iso" in reg.columns and "wall_infer_end_iso" in reg.columns:
        pairs.append(("infer", "wall_infer_start_iso", "wall_infer_end_iso"))
    # Fallback: legacy column names from earlier notebook versions
    if not pairs and "wall_fit_start_iso" in reg.columns:
        pairs.append(("train", "wall_fit_start_iso", "wall_fit_end_iso"))
        if "wall_pred_start_iso" in reg.columns:
            pairs.append(("infer", "wall_pred_start_iso", "wall_pred_end_iso"))
    if pairs:
        reg = join_csv(RESULTS / "regression_results.csv", energy_df, pairs)
        reg.to_csv(RESULTS / "regression_results_with_energy.csv", index=False)
        print(f"  wrote {RESULTS / 'regression_results_with_energy.csv'}")
    else:
        print("  WARNING: no recognized ISO timestamp columns; skipping")
    print()

    # Stage 5 — inference benchmark. Single window per row.
    print("Joining inference_results.csv...")
    infer = pd.read_csv(RESULTS / "inference_results.csv")
    print(f"  columns: {list(infer.columns)}")
    iso_cols = [c for c in infer.columns if "iso" in c.lower()]
    print(f"  ISO timestamp columns found: {iso_cols}")
    pairs = []
    # Common naming patterns
    for prefix in ["wall_infer", "wall_inference", "wall"]:
        s, e = f"{prefix}_start_iso", f"{prefix}_end_iso"
        if s in infer.columns and e in infer.columns:
            pairs.append(("infer", s, e))
            break
    if pairs:
        infer = join_csv(RESULTS / "inference_results.csv", energy_df, pairs)
        infer.to_csv(RESULTS / "inference_results_with_energy.csv", index=False)
        print(f"  wrote {RESULTS / 'inference_results_with_energy.csv'}")
    else:
        print("  WARNING: no recognized ISO timestamp columns; skipping")

    # Best-run metadata for the Stage 3 -> Stage 4 handoff
    print()
    meta_path = ARTIFACTS / "nlp_probs_run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print("Best-run (Stage 3 retrain for nlp_probs.parquet):")
        for label, s, e in [
            ("best_train", "wall_best_train_start_iso", "wall_best_train_end_iso"),
            ("best_infer", "wall_best_infer_start_iso", "wall_best_infer_end_iso"),
        ]:
            if s in meta and e in meta:
                r = integrate_power(energy_df, meta[s], meta[e])
                print(f"  {label}: cpu={r['cpu_kwh']:.6f} kWh, gpu={r['gpu_kwh']:.6f} kWh, "
                      f"dur={r['duration_s']:.1f}s, samples={r['n_samples']}")
                meta[f"energy_{label}_cpu_kwh"] = r["cpu_kwh"]
                meta[f"energy_{label}_gpu_kwh"] = r["gpu_kwh"]
                meta[f"energy_{label}_kwh"] = r["cpu_kwh"] + r["gpu_kwh"]
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  updated {meta_path}")


if __name__ == "__main__":
    main()