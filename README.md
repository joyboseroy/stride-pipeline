# Indian Judicial Pendency Analysis — STRIDE Pipeline
## From Graph to Graph: A Structural Diagnosis Framework for High Court Delay

### Overview
This repository implements the STRIDE (Structured Transition and Rule-based 
Institutional Diagnosis Engine) pipeline for the paper:
"From Queue to Graph: A Structural Diagnosis Framework for Judicial Pendency 
in Indian High Courts"

### Prerequisites

**Run extraction scripts on local WSL machine only.**
Government portals (ecourts.gov.in, njdg.ecourts.gov.in) block cloud IP ranges.

```bash
pip install ecourts pandas numpy scipy networkx matplotlib seaborn lifelines tqdm anthropic
```

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
04_graph_construction.py   Build transition graphs, find critical edges (E1)
        ↓
05_survival_analysis.py    Competing risks model, clearance curves (E4)
```

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
# If <20%, proceed (LLM will handle residuals)

# Step 5: Build graphs
python src/04_graph_construction.py
# Check outputs/graphs/graph_13.png for Kerala
# Check outputs/critical_edges.csv

# Once Kerala works, run all courts:
python src/01_data_extraction.py --all-courts --max-cases 500
```

### Key Outputs

| File | Contents |
|------|----------|
| outputs/purpose_vocabulary.csv | REVIEW MANUALLY — stage label vocabulary |
| outputs/unmatched_labels.csv | Labels that need more rules or LLM |
| outputs/court_summary.csv | Per-court descriptive statistics |
| outputs/critical_edges.csv | CE(c) per court |
| outputs/bottleneck_classification.csv | Input/Process/Capacity/Output per court |
| outputs/graphs/graph_*.png | Graph visualisations |
| outputs/h1_test_result.json | H1 chi-square test result |
| outputs/edge_duration_distributions.csv | Edge stats for all courts |

### What Will Go Wrong (documented in advance)

1. **CAPTCHA**: ecourts portal hits CAPTCHA intermittently.
   Script backs off 90s automatically. Run overnight. Expect ~5-15% failure rate.

2. **Purpose vocabulary chaos**: Allahabad expected to have 50+ unique labels,
   many abbreviations, some numeric codes. Plan 2-3 hours of manual mapping.

3. **Missing dates**: ~8-10% of cases will have no filing date.
   Kept in dataset but flagged. Cannot contribute to duration analysis.

4. **Backward transitions**: ~5-15% of hearing sequences will show backward
   stage transitions. These are a mix of data entry errors and genuine remands.
   Reported as a finding, not silently dropped.

5. **High OTHER rate in Allahabad**: Expected ~25-30% unclassifiable hearings.
   Allahabad may need to be excluded from graph analysis or treated separately.
   This is itself a data quality finding.

6. **Reserved lag skew**: Expect heavy right tail. Some cases will show 
   1000+ day reserved judgment lags. These are real and important findings.

### Variables of Interest

**New variables not in existing literature:**
- `reserved_lag_days`: decision_date - last_hearing_date
  The previously unmeasured output bottleneck.
  First systematic measurement of reserved judgment delay across Indian HCs.
  
- `short_gap_count`: number of hearing intervals ≤ 14 days
  Proxy for listings without substantive hearing (adjournment signature).

- `bench_type`: single/division from coram field
  Allows analysis of whether bench composition affects duration.

- `any_govt_party`: government entity in petitioner or respondent
  Allows government litigation share analysis per court.

### Citing the Data Sources

If you use this code:
- eCourts data: cite openjustice-in/ecourts (DOI: 10.5281/zenodo.13324986)
- DAKSH data: cite dakshindia.org under CC BY-NC 4.0
- SC judgment data: cite Dattam Labs AWS Open Data under CC-BY-4.0

### License
GPL-3.0 (inherited from openjustice-in/ecourts dependency)
