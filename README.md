# The Style is the Signal (ICWSM)

This repo contains the code used to produce the paper’s reported numbers (logistic regression on CPU; LLM zero-shot via API; LLM tagger trained with 2–4×H200).

## What’s here
- `notebooks/01_mbfc_url_masked_logreg_v6.ipynb`: Table 2 (TF‑IDF, SBERT+LR, Tag2Cred, ensembles; domain-disjoint; URL-masked).
- `scripts/run_all_baselines.py`: LLM-zero-shot + PASTEL-style baseline (API; Snorkel LabelModel).
- `notebooks/02_tag_field_ablations.ipynb`: Table 3 tag-field ablations.
- `notebooks/05_regenerate_icwsm_figures.ipynb`: regenerates the leave-one-theme + tagger-noise plots from `results/`.
- `tagger/`: QLoRA SFT training/inference utilities for the codebook tagger.

## Setup
`pip install -r requirements.txt`

## Data
Message text is not included. Point the notebooks/scripts to your released CSV/JSON files via env vars (see headers in each file), e.g.:
- `MBFC_DATA_PATH=/path/to/messages_with_risk_label_urls_removed_nonempty_no_linkurl_evidence.csv`
- `OPENAI_API_KEY=...` (for `scripts/run_all_baselines.py`)

## Paper artifacts
Precomputed aggregates used in the paper are in `results/` (CSV only; no message text).
