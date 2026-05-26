"""
03_all_courts.py
================
Runs analysis across all available DAKSH court CSV files.
Fixes stage harmonisation for court-specific vocabularies:
  - Calcutta: MOTION terminology = admission queue
  - Uttarakhand: numeric slot suffixes in stage labels
  - Andhra Pradesh: department-specific admission stages
  - Karnataka: PRELIMINARY HEARING = arguments queue

Also detects and flags the 20599-day sentinel value problem.

Usage:
    cd ~/judgments
    python3 src/03_all_courts.py 2>&1 | tee outputs/all_courts.txt
"""

import re
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = Path("data/csv")
OUT     = Path("outputs")
OUT.mkdir(exist_ok=True)

SENTINEL_DAYS = 20599  # DAKSH sentinel for missing filing date

# ── Court file registry ───────────────────────────────────────────────────────
COURTS = {
    "Allahabad_Writ_Case.csv":           "Allahabad HC",
    "Andhra Pradesh_Writ_Case.csv":      "Andhra Pradesh HC",
    "Calcutta_Writ_Case.csv":            "Calcutta HC",
    "Chhattisgarh_Writ_Case.csv":        "Chhattisgarh HC",
    "Jammu and Kashmir_Writ_Case.csv":   "Jammu Kashmir HC",
    "Karnataka_Writ_Case.csv":           "Karnataka HC",
    "Kerala_Writ_Case.csv":              "Kerala HC",
    "Manipur_Writ_Case.csv":             "Manipur HC",
    "Meghalaya_Writ_Case.csv":           "Meghalaya HC",
    "Uttarakhand_Writ_Case.csv":         "Uttarakhand HC",
    "Jammu and Kashmir_Writ_Case.csv":   "Jammu Kashmir HC",
}

# ── Government party patterns ─────────────────────────────────────────────────
GOVT_PATTERNS = [
    "union of india", "government of india", "state of ", "govt of ",
    "government of ", "district collector", "commissioner",
    "secretary to government", "ministry of", "department of",
    "municipal corporation", "income tax", "assessing officer",
    "central government", "president of india", "governor of",
    "high court of", "district judge", "tahasildar", "tahsildar",
    "state information", "kerala state", "cochin port",
    "industrial tribunal", "debt recovery", "housing board",
    "supdt. of police", "superintendent of police",
    "c.i. of police", "ci of police", "state bank",
    "public sector", "nationalised bank", "reserve bank",
    "panchayat", "gram sabha", "nagar palika", "nagar nigam",
]

def is_govt(name: str) -> bool:
    if not name or str(name) == 'nan': return False
    n = str(name).lower()
    return any(p in n for p in GOVT_PATTERNS)


# ── Improved stage harmonisation ──────────────────────────────────────────────
# Court-specific overrides applied before generic rules.
# Key insight: each court has its own procedural dialect.

COURT_SPECIFIC_PATTERNS = {
    # Calcutta uses MOTION for admission-queue hearings
    "Calcutta HC": {
        "motion":           "ADMITTED",
        "new motion":       "ADMITTED",
        "listed motion":    "ADMITTED",
        "urgent motion":    "ADMITTED",
        "civil new motion": "ADMITTED",
        "new motions":      "ADMITTED",
        "interview matter": "ARGUMENTS",
        "warning list":     "ARGUMENTS",
        "upgraded matters": "ARGUMENTS",
        "old matter":       "ARGUMENTS",
        "irrespective":     "ARGUMENTS",
        "for final disposal": "RESERVED",
        "court applications under art": "ADMITTED",
    },
    # Karnataka uses PRELIMINARY HEARING for admission/motions queue
    "Karnataka HC": {
        "preliminary hearing":          "ADMITTED",
        "preliminary hearing - b group":"ADMITTED",
        "preliminary hearing - 2:30":   "ADMITTED",
        "final hearing":                "ARGUMENTS",
        "further hearing":              "ARGUMENTS",
        "hearing - interlocutory":      "ARGUMENTS",
        "part heard":                   "ARGUMENTS",
        "further arguments":            "ARGUMENTS",
        "pronouncement of judgement":   "DECIDED",
        "pronouncement of judgment":    "DECIDED",
        "pronouncement of order":       "DECIDED",
        "dictating judgment":           "RESERVED",
        "dictating orders":             "RESERVED",
        "for being spoken to":          "OTHER",
        "being spoken to":              "OTHER",
        "other matters":                "OTHER",
        "non-compliance":               "OTHER",
        "final disposal":               "DECIDED",
        "orders reg":                   "OTHER",
    },
    # Andhra Pradesh has department-tagged admission stages
    "Andhra Pradesh HC": {
        "for admission":            "ADMITTED",
        "admission":                "ADMITTED",  # catches all ADMISSION (DEPT) variants
        "interlocutory":            "ARGUMENTS",
        "final hearing":            "ARGUMENTS",
        "for hearing":              "ARGUMENTS",
        "for orders":               "RESERVED",
        "for judgment":             "RESERVED",
        "for pronouncement":        "DECIDED",
        "for withdrawal":           "WITHDRAWN",
        "for dismissal":            "DECIDED",
        "for being mentioned":      "OTHER",
        "for extension":            "ARGUMENTS",
        "for admission & hearing":  "ARGUMENTS",
        "for admission & reply":    "ADMITTED",
        "adjourned":                "ARGUMENTS",
    },
    # J&K uses descriptive multi-word stages with notice/fresh/non-fresh variants
    "Jammu Kashmir HC": {
        "for admission (after notice)":      "ARGUMENTS",   # post-notice admission = substantive
        "for admission(non fresh)":          "ARGUMENTS",   # returning admission matters
        "for admission (before notice)":     "ADMITTED",    # fresh pre-notice
        "for admission(fresh)":              "ADMITTED",
        "for orders(non fresh)":             "RESERVED",
        "for orders (after notice)":         "RESERVED",
        "for orders(fresh)":                 "RESERVED",
        "for orders (before notice)":        "RESERVED",
        "for final hearing":                 "ARGUMENTS",
        "for final disposal":                "DECIDED",
        "for judgment":                      "RESERVED",
        "for judgement":                     "RESERVED",
        "reserve for judgement":             "RESERVED",
        "for dismissal":                     "DECIDED",
        "cases to be transferred":           "OTHER",       # admin transfer — not judicial
        "specially ordered":                 "ARGUMENTS",
        "in chambers":                       "ARGUMENTS",
        "for filing objections":             "NOTICED",
        "for filing counter":                "NOTICED",
        "for filing statement":              "NOTICED",
        "for service":                       "NOTICED",
        "for taking steps":                  "NOTICED",
        "defected case":                     "FILED",
        "infructuous":                       "DECIDED",
        "for withdrawal":                    "WITHDRAWN",
        "video conference":                  "ARGUMENTS",
        "for further proceedings":           "ARGUMENTS",
        "for framing of issues":             "ARGUMENTS",
    },
    # Uttarakhand embeds slot numbers — strip them first
    "Uttarakhand HC": {
        "admission matters":        "ADMITTED",
        "fresh cases for admission":"ADMITTED",
        "orders on applications":   "RESERVED",
        "order matters":            "RESERVED",
        "final hearing":            "ARGUMENTS",
        "further orders":           "RESERVED",
        "regularization matters":   "ARGUMENTS",
        "for admission":            "ADMITTED",
        "for final disposal":       "DECIDED",
        "personal appearance":      "ARGUMENTS",
        "for dictation of judgment":"RESERVED",
        "for order":                "RESERVED",
        "fresh cases for orders":   "RESERVED",
        "old matters":              "ARGUMENTS",
        "on receipt of report":     "OTHER",
        "lower court proceedings":  "OTHER",
        "pronouncement of judgment":"DECIDED",
        "defective":                "FILED",
    },
}

# Generic patterns applied to all courts
GENERIC_STAGE_MAP = [
    # FILED / registry stage
    ("defect",          "FILED"),
    ("filing defect",   "FILED"),
    ("for filing",      "FILED"),
    ("objection",       "FILED"),
    ("condonation",     "FILED"),
    # ADMITTED
    ("admission",       "ADMITTED"),
    ("for admission",   "ADMITTED"),
    ("habeas corpus",   "ADMITTED"),
    # NOTICED
    ("for steps",       "NOTICED"),
    ("for service",     "NOTICED"),
    ("notice",          "NOTICED"),
    ("file retained",   "NOTICED"),
    ("not allocated",   "NOTICED"),
    # ARGUMENTS
    ("for hearing",     "ARGUMENTS"),
    ("hearing",         "ARGUMENTS"),
    ("for disposal",    "ARGUMENTS"),
    ("adjourned",       "ARGUMENTS"),
    ("for appearance",  "ARGUMENTS"),
    ("petitions",       "ARGUMENTS"),
    ("interlocutory",   "ARGUMENTS"),
    ("final hearing",   "ARGUMENTS"),
    # RESERVED
    ("for orders",      "RESERVED"),
    ("for judgement",   "RESERVED"),
    ("for judgment",    "RESERVED"),
    ("reserved",        "RESERVED"),
    ("heard and",       "RESERVED"),
    ("orders",          "RESERVED"),
    # DECIDED
    ("judgment",        "DECIDED"),
    ("disposed",        "DECIDED"),
    ("disposal",        "DECIDED"),
    ("pronouncement",   "DECIDED"),
    ("dictating",       "DECIDED"),
    # WITHDRAWN
    ("withdrawal",      "WITHDRAWN"),
    ("withdrawn",       "WITHDRAWN"),
    ("settlement",      "WITHDRAWN"),
    ("for settlement",  "WITHDRAWN"),
]

def harmonise_stage(stage: str, court_name: str = "") -> str:
    if not stage or str(stage) == 'nan':
        return "OTHER"

    # Strip Uttarakhand-style slot suffixes: "ADMISSION MATTERS -25" -> "ADMISSION MATTERS"
    s = re.sub(r'\s*-\s*\d+\s*$', '', str(stage)).lower().strip()

    if not s:
        return "OTHER"

    # Court-specific overrides first
    court_patterns = COURT_SPECIFIC_PATTERNS.get(court_name, {})
    for pattern, canonical in court_patterns.items():
        if pattern in s:
            return canonical

    # Generic patterns
    for pattern, canonical in GENERIC_STAGE_MAP:
        if pattern in s:
            return canonical

    return "OTHER"


# ── Analysis functions ────────────────────────────────────────────────────────

def load_and_enrich(filepath: Path, court_name: str) -> pd.DataFrame:
    encoding = 'latin-1' if 'jammu' in filepath.name.lower() else 'utf-8'
    df = pd.read_csv(filepath, low_memory=False,
                     encoding=encoding,
                     na_values=["NA","N/A",""])
    df.columns = [c.strip().upper() for c in df.columns]

    for col in ["DATE_FILED","DECISION_DATE","REGISTRATION_DATE"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["PENDING_DAYS","HEARING_COUNT","DISPOSALTIME_ADJ","YEAR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    today = pd.Timestamp.today()

    # Duration — flag sentinel values
    if "DATE_FILED" in df.columns:
        df["duration_days"] = (
            df["DECISION_DATE"].fillna(today) - df["DATE_FILED"]
        ).dt.days
        # Detect sentinel
        sentinel_mask = df["duration_days"].round(0) == SENTINEL_DAYS
        sentinel_count = sentinel_mask.sum()
        if sentinel_count / len(df) > 0.5:
            df["duration_days"] = np.where(sentinel_mask, np.nan, df["duration_days"])
            df["_has_sentinel"] = True
        else:
            df["_has_sentinel"] = False
        df.loc[df["duration_days"] < 0, "duration_days"] = np.nan

    df["is_disposed"] = df.get("CURRENT_STATUS","").str.upper().str.contains(
        "DISPOS", na=False
    )
    df["resp_is_govt"] = df.get("RESPONDENT","").apply(is_govt)
    df["canonical_stage"] = df.get("CURRENT_STAGE","").apply(
        lambda x: harmonise_stage(x, court_name)
    )
    df["filing_year"] = df.get("DATE_FILED", pd.Series()).dt.year

    return df


def compute_sdi(df: pd.DataFrame, court_name: str) -> dict:
    """Semantic Disorder Index and vocabulary metrics."""
    stages = df.get("CURRENT_STAGE", pd.Series())
    total  = len(df)

    # Raw vocabulary
    counts   = stages.value_counts()
    n_unique = len(counts)

    # Shannon entropy
    entropy = 0.0
    for c in counts.values:
        p = c / total
        if p > 0:
            entropy -= p * np.log2(p)

    # OTHER rate after harmonisation
    other_rate = (df["canonical_stage"] == "OTHER").mean()

    # Stuck-at-admission: pending cases at ADMITTED with >10 hearings
    if "HEARING_COUNT" in df.columns:
        stuck = (
            (df["canonical_stage"] == "ADMITTED") &
            (df["HEARING_COUNT"] > 10) &
            (~df["is_disposed"])
        ).mean()
    else:
        stuck = 0.0

    # Backward proxy: admitted stage with very high hearing count (process stall)
    sdi = round(entropy * 0.4 + other_rate * 0.4 + stuck * 0.2, 4)

    return {
        "vocab_size":    n_unique,
        "label_entropy": round(entropy, 3),
        "other_rate_pct":round(other_rate * 100, 1),
        "stuck_pct":     round(stuck * 100, 1),
        "sdi":           sdi,
    }


def classify_bottleneck(df: pd.DataFrame) -> str:
    """
    Classify dominant bottleneck from stage distribution of pending cases.
    Uses pending cases only.
    """
    pending = df[~df["is_disposed"]]
    if len(pending) == 0:
        return "Unknown"

    stage_dist = pending["canonical_stage"].value_counts(normalize=True)

    admitted  = stage_dist.get("ADMITTED",  0)
    arguments = stage_dist.get("ARGUMENTS", 0)
    reserved  = stage_dist.get("RESERVED",  0)
    filed     = stage_dist.get("FILED",     0)

    # Zero-hearing rate as capacity signal
    zero_hearing_rate = 0
    if "HEARING_COUNT" in df.columns:
        zero_hearing_rate = (pending["HEARING_COUNT"] == 0).mean()

    # Classification logic
    if filed > 0.30:
        return "Input"          # cases stuck before admission
    if admitted > 0.45:
        return "Input"          # large admission queue
    if zero_hearing_rate > 0.80:
        return "Input"          # admitted but never scheduled
    if reserved > 0.15:
        return "Output"         # large reserved judgment backlog
    if arguments > 0.50:
        return "Capacity"       # stuck at hearing stage
    if admitted > 0.25:
        return "Capacity"       # mixed admission + hearing stall
    return "Capacity"           # default for congested courts


def compute_metrics(df: pd.DataFrame, court_name: str) -> dict:
    n = len(df)
    disposed = df["is_disposed"].sum()
    pending  = df[~df["is_disposed"]]

    m = {
        "court":            court_name,
        "n_cases":          n,
        "n_disposed":       int(disposed),
        "disposal_rate_pct":round(disposed / n * 100, 1),
    }

    # Duration
    dur = df["duration_days"].dropna()
    has_sentinel = df.get("_has_sentinel", pd.Series([False])).iloc[0]
    if len(dur) > 100 and not has_sentinel:
        m.update({
            "median_duration_days": round(dur.median(), 0),
            "mean_duration_days":   round(dur.mean(), 0),
            "p90_duration_days":    round(dur.quantile(0.9), 0),
            "pct_over_10yr":        round((dur > 365*10).mean() * 100, 1),
            "pct_over_20yr":        round((dur > 365*20).mean() * 100, 1),
        })
    else:
        m["duration_note"] = "unreliable (sentinel value detected)"

    # Hearings
    if "HEARING_COUNT" in df.columns:
        hc = df["HEARING_COUNT"].dropna()
        m.update({
            "median_hearings":      round(hc.median(), 1),
            "mean_hearings":        round(hc.mean(), 1),
            "pct_zero_hearings":    round((hc == 0).mean() * 100, 1),
        })

    # Pending analysis
    if len(pending) > 0:
        pend_dur = pending["duration_days"].dropna()
        if len(pend_dur) > 10:
            m["pct_pending_over_5yr"]  = round((pend_dur > 365*5).mean() * 100, 1)
            m["pct_pending_over_10yr"] = round((pend_dur > 365*10).mean() * 100, 1)
            m["oldest_pending_years"]  = round(pend_dur.max() / 365, 1)
        if "PENDING_DAYS" in df.columns:
            m["median_pending_days"] = round(df["PENDING_DAYS"].dropna().median(), 0)

    # Government respondent
    m["pct_govt_respondent"] = round(df["resp_is_govt"].mean() * 100, 1)

    # SDI
    sdi_metrics = compute_sdi(df, court_name)
    m.update(sdi_metrics)

    # Bottleneck
    m["bottleneck_class"] = classify_bottleneck(df)

    # Stage distribution for pending
    stage_dist = pending["canonical_stage"].value_counts(normalize=True).mul(100).round(1)
    m["stage_dist"] = stage_dist.to_dict()

    # Attrition signal
    if "DISPOSAL_PATTERN" in df.columns:
        disposed_df = df[df["is_disposed"]]
        if len(disposed_df) > 0:
            dp = disposed_df["DISPOSAL_PATTERN"].fillna("").str.upper()
            withdrawn = dp.str.contains(
                "WITHDRAW|SETTLE|COMPROM|ABANDON|NOT PRESSED|INFRUCTUOUS", na=False
            ).mean()
            merits = dp.str.contains(
                "ALLOW|DISMISS|JUDG|ORDER|MERIT|CLOSED", na=False
            ).mean()
            m["withdrawal_rate_pct"]    = round(withdrawn * 100, 1)
            m["merit_disposal_rate_pct"]= round(merits * 100, 1)
            m["attrition_ratio"]        = round(
                withdrawn / (merits + 0.001), 3
            )

    return m


def analyse_court(filepath: Path, court_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"{court_name}  ({filepath.stat().st_size/1e6:.1f} MB)")
    print(f"{'='*60}")

    df = load_and_enrich(filepath, court_name)
    print(f"Rows: {len(df):,}  |  Disposed: {df['is_disposed'].sum():,}")

    m = compute_metrics(df, court_name)

    # Print key metrics
    print(f"Disposal rate:      {m['disposal_rate_pct']}%")
    print(f"Govt respondent:    {m['pct_govt_respondent']}%")
    print(f"Vocab size:         {m['vocab_size']}")
    print(f"SDI:                {m['sdi']}")
    print(f"OTHER rate:         {m['other_rate_pct']}%")
    print(f"Zero hearings:      {m.get('pct_zero_hearings','N/A')}%")
    print(f"Bottleneck class:   {m['bottleneck_class']}")

    if "median_duration_days" in m:
        print(f"Median duration:    {m['median_duration_days']} days ({m['median_duration_days']/365:.1f} yr)")
        print(f"Pct >10yr:          {m.get('pct_over_10yr','N/A')}%")
    else:
        print(f"Duration:           {m.get('duration_note','N/A')}")

    print(f"\nStage distribution (pending):")
    for stage, pct in sorted(m["stage_dist"].items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"  {stage:12s} {pct:5.1f}%  {bar}")

    # Quick plot
    safe = court_name.replace(" ","_").replace("/","_")
    _plot_stage(df, court_name, OUT / f"stage_{safe}.png")
    if "median_duration_days" in m:
        _plot_duration(df, court_name, OUT / f"duration_{safe}.png")

    return m


def _plot_stage(df, court_name, outpath):
    pending = df[~df["is_disposed"]]
    counts  = pending["canonical_stage"].value_counts()
    colors  = {"ADMITTED":"#3498db","ARGUMENTS":"#f39c12","RESERVED":"#9b59b6",
               "FILED":"#95a5a6","NOTICED":"#2ecc71","DECIDED":"#27ae60",
               "WITHDRAWN":"#e67e22","OTHER":"#e74c3c"}
    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(c,"#bdc3c7") for c in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{val:,}", ha="center", fontsize=8)
    ax.set_title(f"Pending Case Stage Distribution — {court_name}", fontsize=11)
    ax.set_ylabel("Cases")
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def _plot_duration(df, court_name, outpath):
    fig, ax = plt.subplots(figsize=(10,4))
    dur = df["duration_days"].dropna().clip(upper=365*25) / 365
    ax.hist(dur, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(dur.median(), color="red", linestyle="--",
               label=f"Median {dur.median():.1f}yr")
    ax.set_title(f"Case Duration Distribution — {court_name}", fontsize=11)
    ax.set_xlabel("Years")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def main():
    all_metrics = []

    for filename, court_name in COURTS.items():
        filepath = CSV_DIR / filename
        if not filepath.exists():
            print(f"SKIPPING (not found): {filename}")
            continue
        try:
            m = analyse_court(filepath, court_name)
            all_metrics.append(m)
        except Exception as e:
            print(f"ERROR on {court_name}: {e}")
            import traceback; traceback.print_exc()

    # ── Cross-court summary table ──
    print("\n\n" + "="*80)
    print("CROSS-COURT SUMMARY")
    print("="*80)

    rows = []
    for m in all_metrics:
        rows.append({
            "Court":          m["court"],
            "N":              f"{m['n_cases']:,}",
            "Disposed%":      m["disposal_rate_pct"],
            "Med Days":       m.get("median_duration_days", "N/A"),
            ">10yr%":         m.get("pct_over_10yr", "N/A"),
            "ZeroHrg%":       m.get("pct_zero_hearings", "N/A"),
            "Vocab":          m["vocab_size"],
            "OTHER%":         m["other_rate_pct"],
            "SDI":            m["sdi"],
            "Govt%":          m["pct_govt_respondent"],
            "Attrition":      m.get("attrition_ratio", "N/A"),
            "Bottleneck":     m["bottleneck_class"],
        })

    summary = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 20)
    print(summary.to_string(index=False))

    # Save
    # Remove stage_dist dict before saving to CSV
    save_metrics = []
    for m in all_metrics:
        row = {k: v for k, v in m.items() if k != "stage_dist"}
        save_metrics.append(row)

    pd.DataFrame(save_metrics).to_csv(OUT / "cross_court_summary.csv", index=False)

    # Save stage distributions separately
    stage_rows = []
    for m in all_metrics:
        for stage, pct in m.get("stage_dist", {}).items():
            stage_rows.append({
                "court": m["court"],
                "stage": stage,
                "pct_pending": pct
            })
    pd.DataFrame(stage_rows).to_csv(OUT / "stage_distributions.csv", index=False)

    print(f"\nSaved: outputs/cross_court_summary.csv")
    print(f"Saved: outputs/stage_distributions.csv")
    print(f"Saved: outputs/stage_*.png  (one per court)")

    # ── Key findings ──
    df_sum = pd.DataFrame(save_metrics)
    print("\n\n=== KEY FINDINGS ===")

    bc = df_sum["bottleneck_class"].value_counts()
    print(f"\nBottleneck class distribution:")
    for cls, cnt in bc.items():
        courts_in_class = df_sum[df_sum["bottleneck_class"]==cls]["court"].tolist()
        print(f"  {cls:10s}: {cnt} courts — {', '.join(courts_in_class)}")

    sdi_sorted = df_sum.sort_values("sdi")
    print(f"\nSDI ranking (low=orderly, high=chaotic):")
    for _, row in sdi_sorted.iterrows():
        print(f"  {row['sdi']:.3f}  {row['court']}")

    vocab_sorted = df_sum.sort_values("vocab_size", ascending=False)
    print(f"\nVocabulary size ranking:")
    for _, row in vocab_sorted.iterrows():
        print(f"  {int(row['vocab_size']):4d}  {row['court']}")


if __name__ == "__main__":
    main()
