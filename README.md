# Indian Judicial Pendency Analysis
## Diagnosing Structural Bottlenecks in Indian High Courts

This repository contains the analysis code for the paper:

**"From Queue to Graph: Diagnosing Structural Bottlenecks in Indian High Courts"**  
Joy Bose, Senior Data Scientist and Independent Researcher, Bengaluru

- Medium explainer: [Why Indian Courts Are Slow: A Graph Theory Explanation Nobody Asked For (But Everyone Needs)](https://medium.com/@joyboseroy/why-indian-courts-are-slow-a-graph-theory-explanation-nobody-asked-for-but-everyone-needs-56910f5442df)

---

### Repository Contents

```
src/02_daksh_analysis.py      Single-court analysis from DAKSH CSV
src/03_all_courts.py          Cross-court analysis — runs all ten courts
src/04_visualise_findings.py  Generates figures 1-5
outputs/cross_court_summary.csv   Per-court metrics and bottleneck classifications
outputs/stage_distributions.csv  Stage distribution of pending cases per court
LICENSE
```

---

### Data

Raw data is not included. Download writ case CSV files from the DAKSH High Court Data Portal at database.dakshindia.org (registration required, CC BY-NC 4.0 licence). Place files in `data/csv/`.

The Jammu and Kashmir file requires `encoding='latin-1'` when loading. Karnataka and Andhra Pradesh have a sentinel value of 20,599 days for missing filing dates — the scripts detect and exclude these automatically.

---

### Running the Analysis

```bash
pip install pandas numpy scipy matplotlib

# Analyse a single court first to check your data
python src/02_daksh_analysis.py --file data/csv/Kerala_Writ_Case.csv --court "Kerala HC"

# Then run all courts
python src/03_all_courts.py

# Generate figures
python src/04_visualise_findings.py
```

---

### License

CC BY-NC 4.0 (matching DAKSH data licence)
