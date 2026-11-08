"""Synthetic difference-in-differences (Arkhangelsky et al. 2021, AER) for staggered MV Act 2019 fine adoption.

Why SDID: the kill check found adopters ran ~5% above expected before adopting and that adoption timing
correlates with prior death growth (rho = -0.42). Plain two-way FE then credits mean reversion to the fines.
SDID re-weights control states to match each cohort's pre-adoption trajectory (unit weights) and re-weights
pre-periods to resemble the post-period (time weights), so the comparison is local to the pre-trend.

Staggered design ("stacked by cohort"): each adoption month A is a cohort. Its controls are Telangana (never
adopted) plus states adopting after A + POST months (not-yet-treated through the whole window). Cohort ATTs are
aggregated with weights n_treated x n_post_months. States with UNKNOWN adoption dates are excluded.

Outcome: log(killed + 1) minus the state x calendar-month mean over 2014-2018 (all states untreated then), so
state-specific seasonality is removed before SDID. COVID months (2020-03..06, 2021-04..06) are dropped from
both pre and post windows in the primary run.

Inference: leave-one-state-out jackknife over the whole stacked estimate (weights re-estimated each time,
which is more conservative than synthdid's fixed-weight jackknife). Falsification: placebo-in-time, i.e.
the same estimator with every adoption date moved 12 months earlier, using pre-adoption data only.
"""
import pathlib, sys
import numpy as np, pandas as pd

D = pathlib.Path(__file__).resolve().parents[1]
COVID = {f"{y}-{m:02d}" for y, ms in ((2020, range(3, 7)), (2021, range(4, 7))) for m in ms}


# ---------- SDID core ----------
def simplex_ls(A, b, reg, iters=5000, tol=1e-10):
    """min ||A w - b||^2 + reg * ||w||^2 over the simplex, Frank-Wolfe with exact line search."""
    n = A.shape[1]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = A @ w - b
        grad = 2 * (A.T @ r) + 2 * reg * w
        j = int(np.argmin(grad))
        d = -w.copy(); d[j] += 1.0                      # direction to vertex e_j
        Ad = A @ d
        den = Ad @ Ad + reg * (d @ d)
        if den <= 0:
            break
        step = np.clip(-(r @ Ad + reg * (w @ d)) / den, 0.0, 1.0)
        if step * np.abs(d).max() < tol:
            break
        w = w + step * d
    return w


def sdid(Y, n_co, t_pre):
    """Y: units x periods, controls first (n_co rows), pre-periods first (t_pre cols). Returns (tau, omega, lam)."""
    Yc_pre, Yc_post = Y[:n_co, :t_pre], Y[:n_co, t_pre:]
    Yt_pre, Yt_post = Y[n_co:, :t_pre], Y[n_co:, t_pre:]
    n_tr, t_post = Y.shape[0] - n_co, Y.shape[1] - t_pre
    sigma = np.std(np.diff(Yc_pre, axis=1), ddof=1)
    zeta = (n_tr * t_post) ** 0.25 * sigma
    # unit weights: controls' pre-paths (with intercept -> centred over time) match treated mean pre-path
    A = Yc_pre.T - Yc_pre.T.mean(0); b = Yt_pre.mean(0) - Yt_pre.mean()
    omega = simplex_ls(A, b, zeta ** 2 * t_pre)
    # time weights: pre-periods (with intercept -> centred over units) predict controls' post mean
    A = Yc_pre - Yc_pre.mean(0); b = Yc_post.mean(1) - Yc_post.mean()
    lam = simplex_ls(A, b, (1e-6 * sigma) ** 2 * n_co)
    tau = (Yt_post.mean() - Yt_pre.mean(0) @ lam) - (omega @ Yc_post.mean(1) - omega @ Yc_pre @ lam)
    return tau, omega, lam


# ---------- data ----------
def load(min_deaths=0):
    p = pd.read_csv(D / "data/rai_state_month_2014_2024.csv", parse_dates=["date"])
    f = pd.read_csv(D / "data/mva2019_state_fines.csv")
    f["state"] = f.state.replace({"Dadra & Nagar Haveli and Daman & Diu": "DNH & DD",
                                  "Jammu & Kashmir and Ladakh": "J&K & Ladakh"})
    f["adopt"] = pd.to_datetime(f.adoption_date).dt.to_period("M").dt.to_timestamp()
    f = f[(f.level == "NOT_ADOPTED") | f.adopt.notna()]
    p = p[p.state.isin(f.state)].copy()
    p["y"] = np.log1p(p.killed)
    base = p[p.date.dt.year <= 2018].groupby(["state", p.date.dt.month]).y.mean()
    p["y"] = p.y - base.reindex(list(zip(p.state, p.date.dt.month))).values
    big = p[p.date.dt.year == 2018].groupby("state").killed.sum()
    keep = big[big >= min_deaths].index
    Y = p[p.state.isin(keep)].pivot(index="state", columns="date", values="y")
    return Y, f.set_index("state").loc[keep]


def stacked(Y, f, post=6, shift=0, drop_covid=True, end=None):
    """Cohort-stacked SDID. shift = months to move every adoption date earlier (placebo-in-time)."""
    cols = [c for c in Y.columns if not (drop_covid and c.strftime("%Y-%m") in COVID)]
    if end is not None:
        cols = [c for c in cols if c < end]
    adopt = f.adopt.dropna() - pd.DateOffset(months=shift)
    real = f.adopt                                           # controls must be untreated in REAL time
    out = []
    for A, grp in adopt.groupby(adopt):
        tr = list(grp.index)
        post_cols = [c for c in cols if c >= A][:post]
        pre_cols = [c for c in cols if c < A]
        if len(post_cols) < post or len(pre_cols) < 24:
            continue
        last = post_cols[-1]
        co = [s for s in f.index if s not in tr and (pd.isna(real[s]) or real[s] > last)]
        if len(co) < 2:
            continue
        M = Y.loc[co + tr, pre_cols + post_cols].values
        tau, om, lam = sdid(M, len(co), len(pre_cols))
        out.append(dict(cohort=A.strftime("%Y-%m"), n_tr=len(tr), n_co=len(co), tau=tau, wt=len(tr) * post,
                        treated=";".join(tr), top_controls=";".join(f"{co[i]}:{om[i]:.2f}" for i in np.argsort(-om)[:3])))
    c = pd.DataFrame(out)
    att = np.average(c.tau, weights=c.wt) if len(c) else np.nan
    return att, c


def jackknife(Y, f, **kw):
    full, c = stacked(Y, f, **kw)
    reps = []
    for s in Y.index:
        keep = [x for x in Y.index if x != s]
        a, _ = stacked(Y.loc[keep], f.loc[keep], **kw)
        reps.append(a)
    reps = np.array(reps); n = len(reps)
    se = np.sqrt((n - 1) / n * np.sum((reps - reps.mean()) ** 2))
    return full, se, c


if __name__ == "__main__":
    rows = []
    for label, kw, md in [("primary: 6 mo, COVID dropped, all states", dict(post=6), 0),
                          ("large states (>=500 deaths 2018)", dict(post=6), 500),
                          ("12-month window", dict(post=12), 0),
                          ("COVID months kept", dict(post=6, drop_covid=False), 0),
                          ("PLACEBO: dates moved 12 mo earlier", dict(post=6, shift=12), 0)]:
        Y, f = load(md)
        if "PLACEBO" in label:   # placebo uses only data before each real adoption: truncate globally at the first one
            kw["end"] = f.adopt.min()
        att, se, c = jackknife(Y, f, **kw)
        rows.append(dict(spec=label, att_log=att, se=se, lo=att - 1.96 * se, hi=att + 1.96 * se,
                         pct=100 * np.expm1(att), mde80_reduction_pct=100 * -np.expm1(-2.8 * se),   # smallest % fall detectable, 80% power, 5% two-sided
                         cohorts=len(c), treated=int(c.n_tr.sum()) if len(c) else 0))
        if label.startswith("primary"):
            c.to_csv(D / "outputs/sdid_cohorts_primary.csv", index=False)
            print(c[["cohort", "n_tr", "n_co", "tau", "treated", "top_controls"]].round(3).to_string(index=False), "\n")
    r = pd.DataFrame(rows)
    r.to_csv(D / "outputs/sdid_results.csv", index=False)
    print(r.round(4).to_string(index=False))
