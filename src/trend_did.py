"""Two-way fixed-effects estimates for Bihar's prohibition, with and without a Bihar-specific linear trend.

log(deseasonalised deaths_st) = state FE + month FE + [Bihar x linear time] + b1 Bihar x Apr-Dec 2016 + b2 Bihar x 2017
Sample: Jan 2014 - Dec 2017; Bihar plus the 16 comparison states averaging >= 100 deaths/month (excluding states
with their own alcohol-policy regimes). Standard errors: Newey-West (HAC, 6 lags).
Output: outputs/trend_did.csv
"""
import pathlib, sys, contextlib, io
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

P = pathlib.Path(__file__).resolve().parents[1]
F = P
sys.path.insert(0, str(F / "src"))
from sdid import load                                              # noqa: E402  deseasonalised log deaths

with contextlib.redirect_stdout(io.StringIO()):
    Y, _ = load()
Y.columns = pd.to_datetime(Y.columns)
size = pd.read_csv(F / "data/rai_state_month_2014_2024.csv").groupby("state").killed.mean()
EXCL = ["Gujarat", "Nagaland", "Mizoram", "Manipur", "Kerala"]
units = [s for s in Y.index if size.get(s, 0) >= 100 and s not in EXCL]
assert "Bihar" in units
df = Y.loc[units].stack().rename("y").reset_index(); df.columns = ["state", "date", "y"]
df = df[(df.date >= "2014-01-01") & (df.date <= "2017-12-01")].dropna()
df["t"] = (df.date.dt.year - 2014) + (df.date.dt.month - 1) / 12
df["B"] = (df.state == "Bihar").astype(int)
df["p1"] = ((df.date >= "2016-04-01") & (df.date <= "2016-12-01")).astype(int)
df["p2"] = (df.date >= "2017-01-01").astype(int)
rows = []
for lab, f in (("no Bihar trend", "y ~ C(state) + C(date) + B:p1 + B:p2"),
               ("with Bihar trend", "y ~ C(state) + C(date) + B:t + B:p1 + B:p2")):
    m = smf.ols(f, df).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    for term, per in (("B:p1", "Apr-Dec 2016"), ("B:p2", "2017"), ("B:t", "pre-trend per year")):
        if term not in m.params: continue
        lo, hi = m.conf_int().loc[term]
        rows.append(dict(model=lab, term=per, pct=100 * np.expm1(m.params[term]), lo=100 * np.expm1(lo),
                         hi=100 * np.expm1(hi), p=m.pvalues[term], n_comparison=len(units) - 1))
out = pd.DataFrame(rows)
out.to_csv(P / "outputs/trend_did.csv", index=False)
print(out.round(3).to_string(index=False))
