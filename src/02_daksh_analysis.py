"""
02_daksh_analysis.py
====================
Analyses DAKSH writ petition CSV data directly.
No scraping needed — DAKSH already collected the data.

This script works on the case-level CSV from DAKSH.
Note: DAKSH case-level CSV does NOT have hearing-level records.
Hearing-level data is in a separate DAKSH file if available.

What we CAN compute from case-level CSV:
- Duration distribution (DATE_FILED to DECISION_DATE or today)
- Reserved judgment lag (DECISION_DATE minus last hearing — NOT available here)
- Government party share (from RESPONDENT column)
- Current stage distribution (CURRENT_STAGE column)
- Disposal pattern / competing risks inputs
- Pending days distribution
- Hearing count distribution

What we CANNOT compute from case-level CSV alone:
- State transition graph (needs hearing-level records)
- Edge durations between stages
- Reserved judgment lag precisely

Usage:
    python3 src/02_daksh_analysis.py --file data/raw/daksh/kerala_writ.csv --court Kerala
    python3 src/02_daksh_analysis.py --dir data/raw/daksh/ --all-courts
"""

import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Government party detection ────────────────────────────────────────────────
GOVT_PATTERNS = [
    "union of india", "government of india", "state of ", "govt of ",
    "state information commission", "district collector", "commissioner",
    "secretary to government", "ministry of", "department of",
    "municipal corporation", "income tax", "assessing officer",
    "central government", "president of india", "governor of",
    "high court of", "district judge", "tahasildar", "tahsildar",
    "kerala state", "cochin port", "industrial tribunal",
    "debt recovery", "housing board", "mahatma gandhi university",
    "state bank", "supdt. of police", "superintendent of police",
    "c.i. of police", "ci of police",
]

def is_govt(name: str) -> bool:
    if not name or str(name) == 'nan': return False
    n = str(name).lower()
    return any(p in n for p in GOVT_PATTERNS)

# ── Stage harmonisation for DAKSH current_stage field ────────────────────────
# DAKSH current_stage is much cleaner than raw eCourts purpose labels
# but still needs harmonisation

STAGE_MAP = {
    # ADMITTED / ADMISSION stage
    "admission":            "ADMITTED",
    "for admission":        "ADMITTED",
    "contempt of court cases ( for admission )": "ADMITTED",
    "for admission and stay": "ADMITTED",

    # NOTICED / waiting for response
    "for steps":            "NOTICED",
    "for service":          "NOTICED",
    "notice":               "NOTICED",
    "awaiting posting":     "NOTICED",
    "file retained":        "NOTICED",
    "case is not allocated": "NOTICED",

    # ARGUMENTS / hearing stage
    "for hearing":          "ARGUMENTS",
    "hearing":              "ARGUMENTS",
    "for hearing until disposal": "ARGUMENTS",
    "adjourned":            "ARGUMENTS",
    "for appearance":       "ARGUMENTS",
    "petitions":            "ARGUMENTS",
    "for disposal":         "ARGUMENTS",

    # RESERVED / judgment pending
    "for orders":           "RESERVED",
    "reserved":             "RESERVED",
    "judgment":             "RESERVED",

    # DECIDED
    "disposed":             "DECIDED",
    "disposal":             "DECIDED",

    # DEFECTIVE / filing issues
    "defect":               "FILED",
    "for filing":           "FILED",
    "objection":            "FILED",
}

def harmonise_stage(stage: str) -> str:
    if not stage or str(stage) == 'nan':
        return "OTHER"
    s = str(stage).lower().strip()
    # Exact match first
    if s in STAGE_MAP:
        return STAGE_MAP[s]
    # Substring match
    for pattern, canonical in STAGE_MAP.items():
        if pattern in s:
            return canonical
    return "OTHER"


# ── Load and clean ────────────────────────────────────────────────────────────

def load_daksh_csv(filepath: Path) -> pd.DataFrame:
    """Load DAKSH case CSV with proper types."""
    print(f"Loading: {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)")

    df = pd.read_csv(filepath, low_memory=False, na_values=["NA", "N/A", "", " "])

    # Standardise column names to uppercase
    df.columns = [c.strip().upper() for c in df.columns]

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Parse dates
    for col in ["DATE_FILED", "DECISION_DATE", "REGISTRATION_DATE", "LAST_SYNC_TIME"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric
    for col in ["PENDING_DAYS", "HEARING_COUNT", "DISPOSALTIME_ADJ", "YEAR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns."""
    today = pd.Timestamp.today()

    # Duration in days
    if "DATE_FILED" in df.columns:
        df["duration_days"] = (
            df["DECISION_DATE"].fillna(today) - df["DATE_FILED"]
        ).dt.days
        df.loc[df["duration_days"] < 0, "duration_days"] = np.nan

    # Disposed flag
    if "CURRENT_STATUS" in df.columns:
        df["is_disposed"] = df["CURRENT_STATUS"].str.upper().str.contains(
            "DISPOS", na=False
        )
    elif "DECISION_DATE" in df.columns:
        df["is_disposed"] = df["DECISION_DATE"].notna()

    # Government respondent
    if "RESPONDENT" in df.columns:
        df["resp_is_govt"] = df["RESPONDENT"].apply(is_govt)
    else:
        df["resp_is_govt"] = False

    # Harmonised current stage
    if "CURRENT_STAGE" in df.columns:
        df["canonical_stage"] = df["CURRENT_STAGE"].apply(harmonise_stage)

    # Filing year
    if "DATE_FILED" in df.columns:
        df["filing_year"] = df["DATE_FILED"].dt.year

    # Case age bucket
    if "duration_days" in df.columns:
        bins   = [0, 365, 365*3, 365*5, 365*10, 365*20, np.inf]
        labels = ["<1yr", "1-3yr", "3-5yr", "5-10yr", "10-20yr", ">20yr"]
        df["age_bucket"] = pd.cut(df["duration_days"], bins=bins, labels=labels)

    return df


# ── Analysis functions ────────────────────────────────────────────────────────

def describe_court(df: pd.DataFrame, court_name: str) -> dict:
    """Compute all key metrics for one court."""
    n = len(df)
    disposed = df["is_disposed"].sum() if "is_disposed" in df.columns else 0

    metrics = {
        "court":            court_name,
        "n_cases":          n,
        "n_disposed":       int(disposed),
        "disposal_rate_pct":round(disposed / n * 100, 1) if n > 0 else 0,
    }

    if "duration_days" in df.columns:
        dur = df["duration_days"].dropna()
        metrics.update({
            "median_duration_days": round(dur.median(), 0),
            "mean_duration_days":   round(dur.mean(), 0),
            "p90_duration_days":    round(dur.quantile(0.9), 0),
            "max_duration_days":    round(dur.max(), 0),
            "pct_over_10yr":        round((dur > 365*10).mean() * 100, 1),
            "pct_over_20yr":        round((dur > 365*20).mean() * 100, 1),
        })

    if "HEARING_COUNT" in df.columns:
        hc = df["HEARING_COUNT"].dropna()
        metrics.update({
            "median_hearings":  round(hc.median(), 1),
            "mean_hearings":    round(hc.mean(), 1),
            "pct_zero_hearings":round((hc == 0).mean() * 100, 1),
        })

    if "resp_is_govt" in df.columns:
        metrics["pct_govt_respondent"] = round(df["resp_is_govt"].mean() * 100, 1)

    if "PENDING_DAYS" in df.columns:
        pd_col = df["PENDING_DAYS"].dropna()
        metrics["median_pending_days"] = round(pd_col.median(), 0)

    return metrics


def analyse_current_stage(df: pd.DataFrame, court_name: str):
    """Stage distribution analysis — key for bottleneck classification."""
    if "CURRENT_STAGE" not in df.columns:
        return

    print(f"\n=== Stage Distribution: {court_name} ===")

    # Raw vocabulary
    raw_counts = df["CURRENT_STAGE"].value_counts()
    print(f"\nRaw CURRENT_STAGE vocabulary ({len(raw_counts)} unique values):")
    print(raw_counts.head(30).to_string())

    # Compute semantic disorder index
    total = len(df)
    n_unique = len(raw_counts)
    entropy = 0
    for count in raw_counts.values:
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    other_rate = (df["canonical_stage"] == "OTHER").mean()
    
    # Backward transition proxy: cases still at ADMISSION stage after many hearings
    if "HEARING_COUNT" in df.columns and "canonical_stage" in df.columns:
        stuck_early = (
            (df["canonical_stage"] == "ADMITTED") &
            (df["HEARING_COUNT"] > 10)
        ).mean()
    else:
        stuck_early = 0

    sdi = round(entropy * 0.4 + other_rate * 0.4 + stuck_early * 0.2, 4)

    print(f"\nSemantic Disorder Index (SDI): {sdi:.4f}")
    print(f"  Label entropy:    {entropy:.3f} bits")
    print(f"  OTHER rate:       {other_rate*100:.1f}%")
    print(f"  Stuck-at-admission (>10 hearings): {stuck_early*100:.1f}%")

    # Canonical stage distribution
    if "canonical_stage" in df.columns:
        print(f"\nCanonical stage distribution (pending cases):")
        pending = df[~df["is_disposed"]] if "is_disposed" in df.columns else df
        stage_dist = pending["canonical_stage"].value_counts(normalize=True).mul(100).round(1)
        print(stage_dist.to_string())

    return sdi


def analyse_bottleneck_signals(df: pd.DataFrame, court_name: str) -> dict:
    """
    Compute bottleneck signature metrics from case-level DAKSH data.
    
    Note: Without hearing-level data we cannot compute edge durations.
    But we CAN compute proxy signals:
    - Hearing churn: HEARING_COUNT / duration_days (hearings per day)
    - Progress efficiency proxy: disposed cases' duration distribution
    - Output bottleneck proxy: DISPOSAL_PATTERN or NATURE_OF_DISPOSAL
    - Justice attrition: withdrawal/dismissed rates
    """
    signals = {"court": court_name}

    # Hearing density (hearings per year of case life)
    if "HEARING_COUNT" in df.columns and "duration_days" in df.columns:
        mask = df["duration_days"].notna() & df["HEARING_COUNT"].notna() & (df["duration_days"] > 0)
        if mask.sum() > 0:
            hearing_density = (
                df.loc[mask, "HEARING_COUNT"] / (df.loc[mask, "duration_days"] / 365)
            )
            signals["median_hearings_per_year"] = round(hearing_density.median(), 2)
            signals["mean_hearings_per_year"]   = round(hearing_density.mean(), 2)

    # Justice attrition signal: withdrawal rate among disposed cases
    if "DISPOSAL_PATTERN" in df.columns and "is_disposed" in df.columns:
        disposed = df[df["is_disposed"]]
        if len(disposed) > 0:
            dp = disposed["DISPOSAL_PATTERN"].str.upper().fillna("")
            withdrawn = dp.str.contains("WITHDRAW|SETTLE|COMPROM|ABANDON", na=False)
            merits    = dp.str.contains("ALLOW|DISMISS|JUDG|ORDER|MERIT", na=False)
            signals["withdrawal_rate_pct"]   = round(withdrawn.mean() * 100, 1)
            signals["merit_disposal_rate_pct"] = round(merits.mean() * 100, 1)
            signals["attrition_signal"] = round(
                withdrawn.mean() / (merits.mean() + 0.001), 3
            )

    # Long pending cases — input bottleneck signal
    if "duration_days" in df.columns:
        pending_mask = ~df.get("is_disposed", pd.Series(False, index=df.index))
        pending = df[pending_mask]["duration_days"].dropna()
        if len(pending) > 0:
            signals["pct_pending_over_5yr"] = round((pending > 365*5).mean() * 100, 1)
            signals["pct_pending_over_10yr"] = round((pending > 365*10).mean() * 100, 1)
            signals["oldest_pending_years"]  = round(pending.max() / 365, 1)

    return signals


def plot_duration_distribution(df: pd.DataFrame, court_name: str, outpath: Path):
    """Duration histogram split by disposed/pending."""
    if "duration_days" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Case Duration Distribution — {court_name}", fontsize=13)

    for ax, (label, mask) in zip(axes, [
        ("Disposed Cases", df["is_disposed"] if "is_disposed" in df.columns else pd.Series(True, index=df.index)),
        ("Pending Cases",  ~df["is_disposed"] if "is_disposed" in df.columns else pd.Series(False, index=df.index)),
    ]):
        data = df.loc[mask, "duration_days"].dropna()
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(label)
            continue

        # Cap at 30 years for readability
        data = data.clip(upper=365*30)
        ax.hist(data, bins=50, color="steelblue" if "Disposed" in label else "darkorange",
                edgecolor="white", alpha=0.8)
        ax.axvline(data.median(), color="red", linestyle="--",
                   label=f"Median: {data.median()/365:.1f}yr")
        ax.set_xlabel("Days")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} (n={len(data):,})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Plot saved: {outpath.name}")


def plot_filing_trend(df: pd.DataFrame, court_name: str, outpath: Path):
    """Annual filing trend."""
    if "filing_year" not in df.columns:
        return
    
    yearly = df.groupby("filing_year").size()
    yearly = yearly[(yearly.index >= 2000) & (yearly.index <= 2026)]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(yearly.index, yearly.values, color="steelblue", alpha=0.8)
    ax.set_title(f"Annual Filings — {court_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cases Filed")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Plot saved: {outpath.name}")


def plot_stage_distribution(df: pd.DataFrame, court_name: str, outpath: Path):
    """Current stage distribution for pending cases."""
    if "canonical_stage" not in df.columns:
        return

    pending = df[~df.get("is_disposed", pd.Series(False, index=df.index))]
    counts = pending["canonical_stage"].value_counts()

    colors = {
        "ADMITTED": "#3498db", "NOTICED": "#2ecc71",
        "ARGUMENTS": "#f39c12", "RESERVED": "#9b59b6",
        "DECIDED": "#27ae60", "FILED": "#95a5a6", "OTHER": "#e74c3c"
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        counts.index, counts.values,
        color=[colors.get(c, "#bdc3c7") for c in counts.index]
    )
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:,}", ha="center", fontsize=9)
    ax.set_title(f"Current Stage Distribution (Pending) — {court_name}")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Cases")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def analyse_one_file(filepath: Path, court_name: str):
    print(f"\n{'='*60}")
    print(f"ANALYSING: {court_name}")
    print(f"{'='*60}")

    df = load_daksh_csv(filepath)
    df = enrich(df)

    # Core metrics
    metrics = describe_court(df, court_name)
    print(f"\n--- Core Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Stage analysis (returns SDI)
    sdi = analyse_current_stage(df, court_name)
    if sdi is not None:
        metrics["semantic_disorder_index"] = sdi

    # Bottleneck signals
    signals = analyse_bottleneck_signals(df, court_name)
    print(f"\n--- Bottleneck Signals ---")
    for k, v in signals.items():
        if k != "court":
            print(f"  {k}: {v}")
    metrics.update(signals)

    # Plots
    safe_name = court_name.replace(" ", "_").replace("/", "_")
    plot_duration_distribution(df, court_name, OUT / f"duration_{safe_name}.png")
    plot_filing_trend(df, court_name, OUT / f"filing_trend_{safe_name}.png")
    plot_stage_distribution(df, court_name, OUT / f"stage_dist_{safe_name}.png")

    # Save enriched CSV
    out_csv = OUT / f"enriched_{safe_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nEnriched data saved: {out_csv}")

    return metrics, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="Single DAKSH CSV file")
    ap.add_argument("--dir",  help="Directory of DAKSH CSV files")
    ap.add_argument("--court", default="Unknown", help="Court name label")
    ap.add_argument("--all-courts", action="store_true",
                    help="Analyse all CSV files in --dir")
    args = ap.parse_args()

    all_metrics = []

    if args.file:
        metrics, _ = analyse_one_file(Path(args.file), args.court)
        all_metrics.append(metrics)

    elif args.dir:
        dirpath = Path(args.dir)
        csv_files = sorted(dirpath.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {dirpath}")

        if not args.all_courts:
            # Default: just show what's available
            for f in csv_files:
                size_mb = f.stat().st_size / 1e6
                print(f"  {f.name} ({size_mb:.1f} MB)")
            print("\nAdd --all-courts to analyse all, or use --file for one.")
            return

        for csv_file in csv_files:
            court_name = csv_file.stem.replace("_", " ").title()
            try:
                metrics, _ = analyse_one_file(csv_file, court_name)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  ERROR on {csv_file.name}: {e}")

    else:
        raise SystemExit("Specify --file PATH or --dir PATH")

    # Cross-court summary
    if len(all_metrics) > 1:
        summary = pd.DataFrame(all_metrics)
        summary.to_csv(OUT / "court_comparison.csv", index=False)
        print(f"\n=== CROSS-COURT SUMMARY ===")
        cols = ["court", "n_cases", "disposal_rate_pct", "median_duration_days",
                "pct_over_10yr", "pct_govt_respondent", "semantic_disorder_index",
                "attrition_signal"]
        available = [c for c in cols if c in summary.columns]
        print(summary[available].to_string(index=False))
        print(f"\nSaved: outputs/court_comparison.csv")


if __name__ == "__main__":
    main()
