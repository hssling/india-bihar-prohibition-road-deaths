"""Recompute every number reported in the paper from data/ and outputs/ and compare with the manuscript values.
Exit code 1 on any failure. Tolerance 0.051 for one-decimal percentages, 0.0051 for p-values shown to 2-3 decimals."""
import json, math, pathlib, sys
import numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parent
E = json.loads((R / "results/expected_key_results.json").read_text())
fails, n = [], 0
def check(name, got, want, tol):
    global n; n += 1; ok = abs(got - want) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} got {got:+.4f}  expected {want:+.4f}")
    if not ok: fails.append(name)
panel = pd.read_csv(R / "data/rai_state_month_2014_2024.csv")
for y, v in E["panel_bihar_killed"].items():
    check(f"panel Bihar deaths {y}", panel[(panel.state == "Bihar") & (panel.year == int(y))].killed.sum(), v, 0)
sd = pd.read_csv(R / "outputs/sdid_bihar.csv").set_index("window")
for k, v in E["sdid_pct"].items(): check(f"SDID {k} %", sd.loc[k, "pct"], v, 0.051)
for k, v in E["sdid_p"].items(): check(f"SDID {k} rank p", sd.loc[k, "p_rank"], v, 0.0051)
td = pd.read_csv(R / "outputs/trend_did.csv"); td.index = td.model + "|" + td.term
for k, (m, lo, hi) in E["trend_did"].items():
    check(f"FE {k} %", td.loc[k, "pct"], m, 0.051)
    if lo is not None:
        check(f"FE {k} lo", td.loc[k, "lo"], lo, 0.051); check(f"FE {k} hi", td.loc[k, "hi"], hi, 0.051)
for tag, key in (("main", "bayes"), ("placebo2015", "bayes_placebo")):
    z = np.load(R / f"outputs/bayes_bihar_{tag}.npz", allow_pickle=True)   # our own output (string date array)
    d = pd.to_datetime(z["dates"]); mu = z["mu"].astype(float); y = z["y"]
    per = pd.read_csv(R / f"outputs/bayes_bihar_periods_{tag}.csv").set_index("period")
    for k, (m, lo, hi) in E[key].items():
        a_, b_ = {"Apr-Dec 2016 (months 0-8)": ("2016-04", "2016-12"), "Apr 2016-Mar 2017 (first year)": ("2016-04", "2017-03"),
                  "Jan-Mar 2017 (months 9-11)": ("2017-01", "2017-03"), "Apr-Dec 2017 (SC highway ban in donors)": ("2017-04", "2017-12"),
                  "placebo Apr 2015-Dec 2015": ("2015-04", "2015-12")}[k]
        msk = (d >= a_) & (d <= pd.Timestamp(b_) + pd.offsets.MonthEnd(0))
        r = y[msk].sum() / mu[:, msk].sum(1); q = 100 * (np.percentile(r, [50, 2.5, 97.5]) - 1)
        for lab, g, w in zip(("median", "lo", "hi"), q, (m, lo, hi)): check(f"Bayes {k} {lab} (recomputed from draws)", g, w, 0.051)
        check(f"Bayes {k} median (saved table)", per.loc[k, "pct"], m, 0.051)
z = np.load(R / "outputs/bayes_bihar_main.npz", allow_pickle=True); d = pd.to_datetime(z["dates"])
msk = (d >= "2016-04-01") & (d <= "2016-12-01"); mu = z["mu"].astype(float)
check("Bayes Apr-Dec 2016 observed", z["y"][msk].sum(), E["bayes_counts"]["observed"], 0)
check("Bayes Apr-Dec 2016 expected (median)", np.median(mu[:, msk].sum(1)), E["bayes_counts"]["expected"], 0.5)
check("Bayes Apr-Dec 2016 deaths averted (median)", np.median(mu[:, msk].sum(1) - z["y"][msk].sum()), E["bayes_counts"]["averted"], 0.5)
rc = pd.read_csv(R / "outputs/recording_check.csv").set_index("outcome")
for k, v in E["recording_relative_pct"].items(): check(f"recording {k} relative %", rc.loc[k, "relative_pct"], v, 0.051)
tod = pd.read_csv(R / "outputs/time_of_day_mechanism.csv").set_index("year")
check("night/day triple difference 2016 %", tod.loc[2016, "triple_diff_pct"], E["night_day_triple_diff_2016"], 0.051)
print(f"\n{n - len(fails)}/{n} checks passed"); sys.exit(1 if fails else 0)
