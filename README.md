# AI Bias Audit (Adult) + AIF360 Reweighing (Phases 1–6)

This project audits fairness on the Adult income dataset and prepares bias-mitigation weights using AIF360 Reweighing.

## What was built
- Phase 1: Train/test split saved as NumPy indices.
- Phase 2: Baseline model + predictions (no mitigation).
- Phase 3: Baseline fairness metrics + plots.
- Phase 4: Bias mitigation preprocessing: AIF360 Reweighing weights computed on TRAIN ONLY and exported as scikit-learn `sample_weight`.
- Phase 5: Train a mitigated model using AIF360 Reweighing `sample_weight` (train only).
- Phase 6: Fairness audit + plots for the mitigated model (baseline vs mitigated comparison).


## Quick start (run from repo root)
```bash
conda activate fairness-adult
python -m pip install -r requirements.txt

python -m src.run_phase1
python -m src.run_phase2_baseline
python -m src.run_phase3_audit
python -m src.run_phase4_reweighing_weights
python -m src.run_phase5_reweighed
python -m src.run_phase6_audit_reweighed

## Results (sex as protected attribute)

Baseline fairness (Phase 3) vs reweighed fairness (Phase 6):

| Metric | Baseline | Reweighed |
|---|---:|---:|
| Demographic parity difference | 0.04724 | 0.04449 |
| Demographic parity ratio | 0.54920 | 0.56132 |
| Equalized odds difference | 0.02408 | 0.03010 |
| Equalized odds ratio | 0.84056 | 0.87491 |

Artifacts:
- Baseline: `reports/figures/baseline_fairness_metrics.png`
- Reweighed: `reports/figures/reweighed_fairness_metrics.png`
- Weight histogram: `reports/figures/reweighing_weight_hist.png`


