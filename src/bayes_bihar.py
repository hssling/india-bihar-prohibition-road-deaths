"""Bayesian counterfactual imputation of Bihar's monthly road deaths after total prohibition (1 April 2016).

Model (same family as the MV Act fines paper, src/bayes_imputation.py of the fines study, doi:10.5281/zenodo.24055110):
  y_st ~ NegBin(mu_st, phi),  log mu_st = a_s + seas_s[month] + delta_t + u_st,  u_st AR(1) (stationary)
All untreated state-months enter the likelihood; Bihar's months from the policy date onward are masked, so their
expected counts are counterfactual predictions. A pre-policy excess in u decays at rate rho instead of being credited
to the policy.

Span: Jan 2014 - Aug 2019 (Bihar adopted the MV Act 2019 fines in Sep 2019; no COVID months involved).
Donors: all units averaging >= 10 deaths/month EXCEPT states with their own liquor-policy changes in the span:
  Gujarat (long-standing prohibition), Kerala (bar closures 2014-17), Mizoram (prohibition 2015-19), Nagaland
  (prohibition), Manipur (partial prohibition).
Caveat handled in reporting: the Supreme Court highway liquor-sale ban (1 Apr 2017, diluted from Jul 2017) lowered
alcohol access in donor states only, which biases Bihar's post-Apr-2017 contrast towards zero.

Effects (per posterior draw): ratio observed/expected for Bihar by period and by month since the ban.
Usage: python src/bayes_bihar.py [PLACEBO=0]    PLACEBO=1 -> fake date Apr 2015 (data from Apr 2016 also masked)
"""
import os, sys, pathlib, time
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)

P = pathlib.Path(__file__).resolve().parents[1]
F = P
(P / "outputs").mkdir(exist_ok=True)
PLACEBO = int(sys.argv[1]) if len(sys.argv) > 1 else 0
REAL, FAKE = pd.Timestamp("2016-04-01"), pd.Timestamp("2015-04-01")
EXCL = ["Gujarat", "Kerala", "Mizoram", "Nagaland", "Manipur"]

p = pd.read_csv(F / "data/rai_state_month_2014_2024.csv", parse_dates=["date"])
p = p[(p.date >= "2014-01-01") & (p.date <= "2019-08-01")]
size = p.groupby("state").killed.mean()
keep = [s for s in size.index if size[s] >= 10 and s not in EXCL]
Ydf = p[p.state.isin(keep)].pivot(index="state", columns="date", values="killed")
Ydf = Ydf.loc[["Bihar"] + [s for s in Ydf.index if s != "Bihar"]]          # Bihar is row 0
assert Ydf.notna().all().all(), "missing cells"
dates = Ydf.columns
S, T = Ydf.shape
y = Ydf.values.astype(int)
t0 = FAKE if PLACEBO else REAL
obs = np.ones((S, T), bool)
obs[0] = dates < t0                                   # mask Bihar from the (fake) policy date onward
print(f"units {S} (Bihar + {S-1} donors), months {T}, masked Bihar months {(~obs[0]).sum()}{' PLACEBO' if PLACEBO else ''}")

mcal = jnp.array(dates.month.values - 1)
lag = np.arange(T)[:, None] - np.arange(T)[None, :]
lag_j, low = jnp.array(np.maximum(lag, 0)), jnp.array(lag >= 0)
ybar = np.array([y[i, obs[i]].mean() for i in range(S)])     # observed cells only (no leakage from masked months)


def model(y, obs):
    a = numpyro.sample("a", dist.Normal(jnp.log(jnp.maximum(jnp.array(ybar), 1.0)), 1.0))
    sig_seas = numpyro.sample("sig_seas", dist.HalfNormal(0.3))
    z = numpyro.sample("z_seas", dist.Normal(0, 1).expand([S, 12]))
    seas = sig_seas * (z - z.mean(1, keepdims=True))
    dr = numpyro.sample("delta_raw", dist.Normal(0, 0.5).expand([T]))
    delta = dr - dr.mean()
    rho = numpyro.sample("rho", dist.Uniform(0.0, 0.98))
    sig_u = numpyro.sample("sig_u", dist.HalfNormal(0.2))
    e = numpyro.sample("e", dist.Normal(0, 1).expand([S, T]))
    L = jnp.where(low, rho ** lag_j, 0.0)
    L = L.at[:, 0].multiply(1 / jnp.sqrt(1 - rho ** 2))
    u = sig_u * e @ L.T
    phi = numpyro.sample("phi", dist.Gamma(2.0, 0.01))
    log_mu = a[:, None] + seas[:, mcal] + delta[None, :] + u
    numpyro.deterministic("log_mu_bihar", log_mu[0])
    with numpyro.handlers.mask(mask=obs):
        numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(log_mu), phi), obs=y)


t_start = time.time()
NWARM, NDRAW = int(os.environ.get("NWARM", 1500)), int(os.environ.get("NDRAW", 1000))
mcmc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=8), num_warmup=NWARM, num_samples=NDRAW, num_chains=4,
            progress_bar=False)
mcmc.run(jax.random.PRNGKey(20261109), jnp.array(y), jnp.array(obs), extra_fields=("diverging", "num_steps"))
sm = jax.block_until_ready(mcmc.get_samples())
print(f"sampled in {time.time() - t_start:.0f}s")
tag = "placebo2015" if PLACEBO else "main"
mu = np.exp(np.asarray(sm["log_mu_bihar"], dtype=np.float64))           # draws x T
np.savez_compressed(P / f"outputs/bayes_bihar_{tag}.npz", mu=mu.astype(np.float32), y=y[0], dates=np.array(dates.astype(str)),
                    rho=np.asarray(sm["rho"]))

from numpyro.diagnostics import summary
g = mcmc.get_samples(group_by_chain=True)
summ = summary({k: g[k] for k in ("rho", "sig_u", "sig_seas", "phi")}, prob=0.95)
print(pd.DataFrame(summ).T[["mean", "2.5%", "97.5%", "n_eff", "r_hat"]].round(4).to_string())
print("divergences:", int(mcmc.get_extra_fields()["diverging"].sum()),
      f"| share at max tree depth: {(np.asarray(mcmc.get_extra_fields()['num_steps']) >= 2**8 - 1).mean():.3f}")

# ---- effects by period ----
yb = y[0]
q = lambda v: np.percentile(v, [50, 2.5, 97.5])
if PLACEBO:
    periods = {"placebo Apr 2015-Dec 2015": ("2015-04-01", "2015-12-01"), "placebo Jan-Mar 2016": ("2016-01-01", "2016-03-01")}
else:
    periods = {"Apr-Dec 2016 (months 0-8)": ("2016-04-01", "2016-12-01"),
               "Jan-Mar 2017 (months 9-11)": ("2017-01-01", "2017-03-01"),
               "Apr 2016-Mar 2017 (first year)": ("2016-04-01", "2017-03-01"),
               "Apr-Dec 2017 (SC highway ban in donors)": ("2017-04-01", "2017-12-01"),
               "2018": ("2018-01-01", "2018-12-01"), "Jan-Aug 2019": ("2019-01-01", "2019-08-01")}
rows = []
for lab, (a_, b_) in periods.items():
    m = (dates >= a_) & (dates <= b_)
    r = yb[m].sum() / mu[:, m].sum(1)
    md, lo, hi = q(r)
    rows.append(dict(period=lab, observed=int(yb[m].sum()), expected=float(np.median(mu[:, m].sum(1))),
                     pct=100 * (md - 1), lo=100 * (lo - 1), hi=100 * (hi - 1), p_fall=float((r < 1).mean()),
                     deaths_averted=float(np.median(mu[:, m].sum(1) - yb[m].sum()))))
res = pd.DataFrame(rows)
res.to_csv(P / f"outputs/bayes_bihar_periods_{tag}.csv", index=False)
print(res.round(2).to_string(index=False))
# monthly trajectory (for the figure) and an exponential-decay summary of the first 21 months
post = dates >= t0
traj = pd.DataFrame({"date": dates[post].astype(str), "observed": yb[post],
                     **{f"ratio_{k}": v for k, v in zip(("med", "lo", "hi"), np.percentile(yb[post] / mu[:, post], [50, 2.5, 97.5], 0))}})
traj.to_csv(P / f"outputs/bayes_bihar_monthly_{tag}.csv", index=False)
print("monthly ratio (median):", " ".join(f"{d[2:7]}:{r:.2f}" for d, r in zip(traj.date, traj.ratio_med)))
