# Indian Judicial Pendency Analysis — STRIDE Pipeline
## Structural Diagnosis Framework for High Court Delay

This repository implements the analysis pipeline for the paper:

**"From Queue to Graph: Diagnosing Structural Bottlenecks in Indian High Courts"**
Joy Bose, Senior Data Scientist and Independent Researcher, Bengaluru

- Preprint: HAL (hal.science) — under review
- SSRN: under review
- Medium explainer: [Why Indian Courts Are Slow: A Graph Theory Explanation Nobody Asked For (But Everyone Needs)](https://medium.com/@joyboseroy/why-indian-courts-are-slow-a-graph-theory-explanation-nobody-asked-for-but-everyone-needs-56910f5442df)

---

### What This Does

Analyses 2.45 million writ petition records across ten Indian High Courts to classify courts by structural bottleneck type: Input, Capacity, or Output. Finds that courts fail at different stages of the case lifecycle and require different interventions — vacancy filling helps Input and Capacity courts but has no effect on Output bottleneck courts.

---

### Prerequisites

**Run extraction scripts on local WSL machine only.**
Government portals (ecourts.gov.in, njdg.ecourts.gov.in) block cloud IP ranges.

```bash
pip install ecourts pandas numpy scipy networkx matplotlib seaborn lifelines tqdm anthropic
```

---

### Pipeline (run in order)

```
01_data_extraction.py    Extract case + hearing data from eCourts
        ↓
02_data_quality.py       EDA, field completeness, anomaly detection
        ↓
        MANUAL STEP: Review outputs/purpose_vocabulary.csv
                     Update STAGE_RULES in 03_stage_harmonisation.py
        ↓
03_stage_harmonisation.py  Map raw labels → canonical stages
        ↓
04_graph_construction.py   Build transition graphs, find critical edges
        ↓
05_survival_analysis.py    Competing risks model, clearance curves
```

---

### Start Here (Kerala first)

```bash
# Step 1: Extract Kerala HC data (smallest, cleanest)
python src/01_data_extraction.py --court 13 --max-cases 200

# Step 2: Check what you got
python src/02_data_quality.py

# Step 3: Review outputs/purpose_vocabulary.csv
# Add any unrecognised labels to STAGE_RULES in 03_stage_harmonisation.py

# Step 4: Harmonise stages
python src/03_stage_harmonisation.py --no-llm

# Check outputs/unmatched_labels.csv
# If >20% unmatched, add more rules and re-run

# Step 5: Build graphs
python src/04_graph_construction.py
# Check outputs/graphs/graph_13.png for Kerala
# Check outputs/critical_edges.csv

# Once Kerala works, run all courts:
python src/01_data_extraction.py --all-courts --max-cases 500
```

---

### Key Outputs

| File | Contents |
|------|----------|
| outputs/purpose_vocabulary.csv | REVIEW MANUALLY — stage label vocabulary |
| outputs/unmatched_labels.csv | Labels needing more rules or LLM |
| outputs/court_summary.csv | Per-court descriptive statistics |
| outputs/critical_edges.csv | Critical edge per court |
| outputs/bottleneck_classification.csv | Input/Process/Capacity/Output per court |
| outputs/graphs/graph_*.png | Graph visualisations |
| outputs/h1_test_result.json | H1 chi-square test result |
| outputs/edge_duration_distributions.csv | Edge stats for all courts |
| outputs/cross_court_summary.csv | Cross-court comparison table (from DAKSH analysis) |
| outputs/stage_distributions.csv | Stage distribution of pending cases per court |

---

### Key Variables

**Variables not previously measured in the literature:**

- `reserved_lag_days`: decision_date minus last_hearing_date. First systematic measurement of reserved judgment delay across Indian High Courts. The Output bottleneck signal.

- `short_gap_count`: number of hearing intervals of 14 days or fewer. Proxy for listings without substantive hearing — adjournment signature.

- `bench_type`: single or division bench from coram field. Allows analysis of whether bench composition affects duration.

- `any_govt_party`: government entity in petitioner or respondent. Allows government litigation share analysis per court.

---

### Known Data Issues (documented in advance)

1. **CAPTCHA**: eCourts portal hits CAPTCHA intermittently. Script backs off 90 seconds automatically. Run overnight. Expect 5-15% failure rate.

2. **Vocabulary chaos**: Allahabad expected to have 50+ unique labels, many abbreviations, some numeric codes. Plan 2-3 hours of manual mapping. Calcutta has 350 unique stage labels in the DAKSH data.

3. **Missing dates**: Around 8-10% of cases will have no filing date. Flagged but retained. Cannot contribute to duration analysis.

4. **Sentinel values**: Karnataka and Andhra Pradesh in the DAKSH data show 20,599 days (56.4 years) as a placeholder for missing filing dates. Detect and exclude before duration analysis.

5. **Backward transitions**: Around 5-15% of hearing sequences show backward stage transitions. Mix of data entry errors and genuine remands. Reported as a finding, not silently dropped.

6. **J&K encoding**: The Jammu and Kashmir CSV requires latin-1 encoding due to regional language characters in party names.

---

### Data Sources

- **DAKSH High Court Data Portal**: database.dakshindia.org — CC BY-NC 4.0. Primary source for cross-court analysis. Download writ case CSV files after registration.
- **eCourts**: ecourts.gov.in via openjustice-in/ecourts library (GPL-3.0, DOI: 10.5281/zenodo.13324986). For hearing-level data.
- **IMLJD**: huggingface.co/datasets/joyboseroy/imljd — CC BY 4.0. Karnataka HC Section 482 petitions 2018-2024.

Raw DAKSH CSV files are not included in this repository due to licence restrictions. Download from DAKSH directly.

---

### License

GPL-3.0 (inherited from openjustice-in/ecourts dependency)
