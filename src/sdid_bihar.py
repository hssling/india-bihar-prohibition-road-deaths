"""Synthetic difference-in-differences for Bihar's prohibition (April 2016), with in-space placebo inference.

Outcome: log monthly deaths minus each state's 2014-2018 calendar-month means (sdid.load). Donors: states without
their own alcohol-policy regimes and with complete series (23). Windows: Apr-Dec 2016, Apr 2016-Mar 2017,
Apr 2016-Dec 2017, and a placebo date of April 2015 (pre-policy data only). Inference: each of the 16 donors averaging
>= 100 deaths/month is treated in turn; rank p = (# |placebo| >= |Bihar| + 1) / (16 + 1).
Output: outputs/sdid_bihar.csv
"""
import pathlib, sys, contextlib, io
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
F = P
sys.path.insert(0, str(F / "src"))
from sdid import sdid, load                                       # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    Y, _ = load()
Y.columns = pd.to_datetime(Y.columns)
size = pd.read_csv(F / "data/rai_state_month_2014_2024.csv").groupby("state").killed.mean()
EXCL = ["Gujarat", "Nagaland", "Mizoram", "Manipur", "Kerala"]
SPECS = [("Apr-Dec 2016", "2016-04-01", "2014-01-01", "2016-12-01"),
         ("Apr 2016-Mar 2017", "2016-04-01", "2014-01-01", "2017-03-01"),
         ("Apr 2016-Dec 2017", "2016-04-01", "2014-01-01", "2017-12-01"),
         ("placebo date Apr 2015", "2015-04-01", "2014-01-01", "2016-03-01")]
rows = []
for lab, t0, a, b in SPECS:
    t0, a, b = map(pd.Timestamp, (t0, a, b))
    cols = [d for d in Y.columns if a <= d <= b]
    donors = [s for s in Y.index if s != "Bihar" and s not in EXCL and Y.loc[s, cols].notna().all()]
    npre = sum(d < t0 for d in cols)
    tau = sdid(Y.loc[donors + ["Bihar"], cols].values, len(donors), npre)[0]
    pool = [s for s in donors if size[s] >= 100]
    plac = np.array([sdid(Y.loc[[d for d in donors if d != s] + [s], cols].values, len(donors) - 1, npre)[0] for s in pool])
    rows.append(dict(window=lab, pct=100 * np.expm1(tau), n_donors=len(donors), n_placebo=len(pool),
                     p_rank=(np.sum(np.abs(plac) >= abs(tau)) + 1) / (len(pool) + 1)))
out = pd.DataFrame(rows)
out.to_csv(P / "outputs/sdid_bihar.csv", index=False)
print(out.round(4).to_string(index=False))
