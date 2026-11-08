"""Recording check: did Bihar's recorded accidents and injuries fall much more than deaths after prohibition?

Pure under-recording (police capacity diverted to enforcement) should depress recorded accidents and injuries far
more than deaths. Compares April-December means (calendar held constant) of the post period (Apr-Dec 2016) with the
pre period (Apr-Dec 2014 and 2015), for Bihar against the 16 comparison states averaging >= 100 deaths/month.
Output: outputs/recording_check.csv
"""
import pathlib
import numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
F = P
p = pd.read_csv(F / "data/rai_state_month_2014_2024.csv", parse_dates=["date"])
p = p[(p.date >= "2014-01-01") & (p.date <= "2017-12-01") & (p.date.dt.month >= 4)]
p["period"] = np.select([p.date < "2016-04-01", p.date <= "2016-12-01"], ["pre", "post"], "y2017")
EXCL = ["Gujarat", "Kerala", "Mizoram", "Nagaland", "Manipur"]
size = pd.read_csv(F / "data/rai_state_month_2014_2024.csv").groupby("state").killed.mean()
units = [s for s in size.index if size[s] >= 100 and s not in EXCL]
g = p[p.state.isin(units)].groupby(["state", "period"])[["acc", "inj", "killed"]].mean()
rows = []
for v, lab in (("acc", "accidents"), ("inj", "injuries"), ("killed", "deaths")):
    x = np.log(g[v].unstack("period"))
    ch = (x["post"] - x["pre"]).dropna(); b = ch["Bihar"]; oth = ch.drop("Bihar")
    rows.append(dict(outcome=lab, bihar_pct=100 * np.expm1(b), comparison_median_pct=100 * np.expm1(oth.median()),
                     relative_pct=100 * np.expm1(b - oth.median()), rank=int((ch <= b).sum()), n=len(ch)))
out = pd.DataFrame(rows)
out.to_csv(P / "outputs/recording_check.csv", index=False)
print(out.round(2).to_string(index=False))
