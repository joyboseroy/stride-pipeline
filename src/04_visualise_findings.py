"""
04_visualise_findings.py
========================
Generates publication-quality figures from cross_court_summary.csv.

Also fixes J&K encoding and adds it to the dataset.

Run after 03_all_courts.py:
    python3 src/04_visualise_findings.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ── Fix J&K encoding and run analysis ────────────────────────────────────────

def fix_jk():
    """Try multiple encodings for J&K CSV."""
    jk_path = Path("data/csv/Jammu and Kashmir_Writ_Case.csv")
    if not jk_path.exists():
        print("J&K file not found")
        return None

    for enc in ['latin-1', 'cp1252', 'iso-8859-1', 'utf-8-sig']:
        try:
            df = pd.read_csv(jk_path, low_memory=False,
                           encoding=enc,
                           na_values=["NA","N/A",""])
            print(f"J&K loaded with encoding: {enc}, rows: {len(df):,}")
            return df, enc
        except Exception as e:
            print(f"  {enc}: failed — {e}")
    return None, None


# ── Load summary data ─────────────────────────────────────────────────────────

def load_summary():
    path = OUT / "cross_court_summary.csv"
    if not path.exists():
        raise SystemExit("Run 03_all_courts.py first")
    df = pd.read_csv(path)
    # Remove sentinel-value courts from duration analysis
    df["duration_reliable"] = df["median_duration_days"] != 20599.0
    return df


# ── Figure 1: Bottleneck class distribution ───────────────────────────────────

def fig_bottleneck_distribution(df):
    colors = {"Input":"#e74c3c", "Capacity":"#3498db",
              "Output":"#9b59b6", "Unknown":"#95a5a6"}

    fig, ax = plt.subplots(figsize=(12, 6))

    courts = df.sort_values("bottleneck_class")["court"].str.replace(" HC","")
    bottlenecks = df.sort_values("bottleneck_class")["bottleneck_class"]
    disposal = df.sort_values("bottleneck_class")["disposal_rate_pct"]

    bar_colors = [colors.get(b, "#95a5a6") for b in bottlenecks]
    bars = ax.bar(courts, disposal, color=bar_colors, alpha=0.85, edgecolor="white")

    for bar, bc, val in zip(bars, bottlenecks, disposal):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bc}\n{val:.0f}%", ha="center", fontsize=8, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()
               if l != "Unknown"]
    ax.legend(handles=patches, title="Bottleneck Class", loc="lower right")
    ax.set_title("Bottleneck Classification by Court — Disposal Rate",
                 fontsize=13, pad=15)
    ax.set_ylabel("Disposal Rate (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_bottleneck_classes.png", dpi=150)
    plt.close()
    print("Saved: fig1_bottleneck_classes.png")


# ── Figure 2: SDI vs disposal rate ───────────────────────────────────────────

def fig_sdi_disposal(df):
    colors = {"Input":"#e74c3c", "Capacity":"#3498db", "Output":"#9b59b6"}

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        color = colors.get(row["bottleneck_class"], "#95a5a6")
        ax.scatter(row["sdi"], row["disposal_rate_pct"],
                   s=row["n_cases"]/5000,  # size proportional to caseload
                   color=color, alpha=0.7, edgecolor="white", linewidth=1.5)
        ax.annotate(
            row["court"].replace(" HC",""),
            (row["sdi"], row["disposal_rate_pct"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8
        )

    # Trend line (excluding sentinel-value outliers)
    valid = df[df["disposal_rate_pct"].notna() & df["sdi"].notna()]
    z = np.polyfit(valid["sdi"], valid["disposal_rate_pct"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid["sdi"].min()-0.05, valid["sdi"].max()+0.05, 100)
    ax.plot(x_line, p(x_line), "k--", alpha=0.3, linewidth=1)

    patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    size_note = mpatches.Patch(color="grey", alpha=0.3,
                               label="Circle size ∝ caseload")
    ax.legend(handles=patches + [size_note], title="Bottleneck Class")

    ax.set_xlabel("Semantic Disorder Index (SDI)", fontsize=11)
    ax.set_ylabel("Disposal Rate (%)", fontsize=11)
    ax.set_title("Semantic Disorder vs Court Performance\n"
                 "Bubble size proportional to total caseload",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_sdi_vs_disposal.png", dpi=150)
    plt.close()
    print("Saved: fig2_sdi_vs_disposal.png")


# ── Figure 3: Stage distribution heatmap ─────────────────────────────────────

def fig_stage_heatmap():
    path = OUT / "stage_distributions.csv"
    if not path.exists():
        print("stage_distributions.csv not found, skipping heatmap")
        return

    df = pd.read_csv(path)
    pivot = df.pivot(index="court", columns="stage", values="pct_pending")
    pivot = pivot.fillna(0)

    # Order stages logically
    stage_order = ["FILED","ADMITTED","NOTICED","ARGUMENTS",
                   "RESERVED","DECIDED","WITHDRAWN","OTHER"]
    cols = [c for c in stage_order if c in pivot.columns]
    pivot = pivot[cols]

    # Order courts by bottleneck class then SDI
    summary = pd.read_csv(OUT / "cross_court_summary.csv")
    court_order = summary.sort_values(
        ["bottleneck_class","sdi"]
    )["court"].tolist()
    pivot = pivot.reindex([c for c in court_order if c in pivot.index])

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=80)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [c.replace(" HC","") for c in pivot.index], fontsize=9
    )

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val > 1:
                ax.text(j, i, f"{val:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if val > 50 else "black")

    plt.colorbar(im, ax=ax, label="% of Pending Cases")
    ax.set_title("Stage Distribution of Pending Cases by Court\n"
                 "(Courts ordered by bottleneck class then SDI)",
                 fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_stage_heatmap.png", dpi=150)
    plt.close()
    print("Saved: fig3_stage_heatmap.png")


# ── Figure 4: Vocabulary size vs pendency ────────────────────────────────────

def fig_vocab_pendency(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: vocab size bar chart
    ax = axes[0]
    sorted_df = df.sort_values("vocab_size", ascending=True)
    colors_bc = {"Input":"#e74c3c","Capacity":"#3498db","Output":"#9b59b6"}
    bar_colors = [colors_bc.get(b,"#95a5a6")
                  for b in sorted_df["bottleneck_class"]]
    bars = ax.barh(
        sorted_df["court"].str.replace(" HC",""),
        sorted_df["vocab_size"],
        color=bar_colors, alpha=0.8
    )
    for bar, val in zip(bars, sorted_df["vocab_size"]):
        ax.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2,
                str(val), va="center", fontsize=9)
    ax.set_xlabel("Unique Stage Label Count (Vocabulary Size)")
    ax.set_title("Vocabulary Size by Court\n(Proxy for semantic fragmentation)",
                 fontsize=10)

    # Right: attrition signal
    ax2 = axes[1]
    attrition_df = df[df["attrition_ratio"].notna() &
                      (df["attrition_ratio"] > 0)].sort_values(
        "attrition_ratio", ascending=True
    )
    bar_colors2 = [colors_bc.get(b,"#95a5a6")
                   for b in attrition_df["bottleneck_class"]]
    bars2 = ax2.barh(
        attrition_df["court"].str.replace(" HC",""),
        attrition_df["attrition_ratio"],
        color=bar_colors2, alpha=0.8
    )
    for bar, val in zip(bars2, attrition_df["attrition_ratio"]):
        ax2.text(bar.get_width()+0.005,
                 bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax2.axvline(0.1, color="orange", linestyle="--", alpha=0.5,
                label="Threshold 0.1")
    ax2.set_xlabel("Attrition Ratio\n(Withdrawals / Merit Disposals)")
    ax2.set_title("Justice Attrition Signal by Court\n"
                  "(Higher = more litigants abandoning cases)",
                  fontsize=10)
    ax2.legend(fontsize=8)

    patches = [mpatches.Patch(color=c, label=l)
               for l, c in colors_bc.items()]
    fig.legend(handles=patches, title="Bottleneck",
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("Institutional Observability Metrics Across Indian High Courts",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_vocab_attrition.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fig4_vocab_attrition.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Try to fix J&K
    print("=== Attempting J&K fix ===")
    result = fix_jk()
    if result and result[0] is not None:
        jk_df, enc = result
        print(f"J&K rows: {len(jk_df):,}")
        print(f"Re-run 03_all_courts.py with encoding fix to include J&K")
        # Save encoding info for 03_all_courts.py
        Path("data/csv/.jk_encoding").write_text(enc)

    print("\n=== Generating figures ===")
    df = load_summary()

    fig_bottleneck_distribution(df)
    fig_sdi_disposal(df)
    fig_stage_heatmap()
    fig_vocab_pendency(df)

    print("\n=== Summary Statistics ===")
    print(f"Courts analysed: {len(df)}")
    print(f"\nBottleneck distribution:")
    print(df["bottleneck_class"].value_counts().to_string())
    print(f"\nSDI range: {df['sdi'].min():.3f} — {df['sdi'].max():.3f}")
    print(f"Vocab range: {df['vocab_size'].min()} — {df['vocab_size'].max()}")

    valid_dur = df[df["median_duration_days"] != 20599]
    if len(valid_dur) > 0:
        print(f"\nMedian duration range (valid courts only):")
        print(f"  Fastest: {valid_dur['median_duration_days'].min():.0f} days "
              f"({valid_dur.loc[valid_dur['median_duration_days'].idxmin(),'court']})")
        print(f"  Slowest: {valid_dur['median_duration_days'].max():.0f} days "
              f"({valid_dur.loc[valid_dur['median_duration_days'].idxmax(),'court']})")

    print("\nAll figures saved to outputs/")


if __name__ == "__main__":
    main()
