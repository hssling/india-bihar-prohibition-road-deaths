# Road traffic deaths after total alcohol prohibition in Bihar, India

Data and code for a natural-experiment evaluation of Bihar's total alcohol prohibition (country liquor banned on 1 April 2016, all alcohol from 5 April 2016) on police-recorded road traffic deaths. Bihar is compared with Indian states that did not change alcohol policy. The study uses three designs with a comparison group, placebo tests, a time-of-day mechanism test and an under-recording check.

[![reproduce-and-verify](https://github.com/hssling/india-bihar-prohibition-road-deaths/actions/workflows/verify.yml/badge.svg)](https://github.com/hssling/india-bihar-prohibition-road-deaths/actions/workflows/verify.yml)

**Status.** The accompanying manuscript has been submitted for publication. Please cite this archive (`CITATION.cff`) until the article is out.

## Main findings

| Design | April–December 2016 | Placebo (date moved to April 2015) |
|---|---|---|
| Bayesian counterfactual imputation | −14.2% (95% CrI −23.9 to −3.8); 3,548 observed vs 4,137 expected, ≈590 deaths averted | +9.5% (−3.6 to +24.2) |
| Synthetic difference-in-differences | −16.4% (most extreme of 17 states; rank p = 0.059) | +1.6% (rank p = 0.71) |
| Two-way fixed effects + Bihar trend | −16.0% (95% CI −20.3 to −11.5) | — |

**The effect was transient.** In 2017 the estimates were −3.0% (fixed effects, 95% CI −9.9 to +4.5) and +4.6% (Bayesian, April–December), while prohibition remained in force.

**It was not concentrated at night.** Night-time accidents did not fall more than daytime accidents: the night/day ratio rose by 4.8% relative to comparison states.

**Recording changes don't explain it.** Recorded accidents, injuries and deaths fell by similar proportions (−19.6%, −19.8% and −15.1%), the opposite of the injury-heavy pattern that pure under-recording would produce.

## Reproduce

```bash
pip install -r requirements.txt
python verify.py          # recomputes 49 reported numbers from data/ and outputs/ (seconds)
python run_pipeline.py    # re-runs synthetic DiD, trend fixed effects, recording check (~5 min), then verify
```

Optional steps:
- `pip install -r requirements-bayes.txt` then `python run_pipeline.py --bayes` refits the Bayesian models in NumPyro (about 10 minutes each).
- `python scripts/fetch_raw_reports.py` then `python run_pipeline.py --build` rebuilds the time-of-day table from the MoRTH reports.

GitHub Actions re-runs the frequentist analyses on Linux and runs `verify.py` on every push.

## Contents

| Path | Contents |
|---|---|
| `data/rai_state_month_2014_2024.csv` | **State × month road deaths, accidents and injuries, 2014–2024.** MoRTH *Road Accidents in India* annexures, reconciled to national totals. Built and archived with the MV Act fines study ([doi:10.5281/zenodo.24055110](https://doi.org/10.5281/zenodo.24055110)). |
| `data/time_of_day_state_year.csv` | State-wise annual accidents by day (06–18 h) and night (18–06 h), 2014–2017. Each row's eight 3-hour slots sum to the printed total. |
| `src/sdid_bihar.py` | Synthetic difference-in-differences with in-space placebo inference. |
| `src/trend_did.py` | Two-way fixed effects with and without a Bihar-specific trend. |
| `src/bayes_bihar.py` | NumPyro negative-binomial model with AR(1) state deviations. Bihar's post-ban months are masked, so the model predicts them as counterfactuals. Includes a placebo mode. |
| `src/recording_check.py`, `src/time_of_day_mechanism.py` | Under-recording check and night-time mechanism test. |
| `src/fig_bihar_tip.py` | Figures 1 and 2. |
| `src/sdid.py` | Estimator and panel loader, shared with the fines study. |
| `outputs/` | All results tables, posterior draws (`.npz`), logs and figures. |
| `results/expected_key_results.json` | Values reported in the manuscript, checked by `verify.py`. |

Five states with their own prohibition regimes or major alcohol-policy changes (Gujarat, Kerala, Manipur, Mizoram, Nagaland) are excluded from every comparison group. Estimates after March 2017 are long-range model predictions and are not interpreted causally. From April 2017 the Supreme Court's highway liquor restrictions also applied in the comparison states.

## Licences

- **Code:** MIT.
- **Derived data:** CC BY 4.0.
- **Raw reports:** the MoRTH reports are Government of India publications, redistributed unmodified through the fines study's archive.

## Author

Siddalingaiah H S, Department of Community Medicine, Shridevi Institute of Medical Sciences and Research Hospital, Tumakuru, Karnataka, India. ORCID [0000-0002-4771-8285](https://orcid.org/0000-0002-4771-8285).
