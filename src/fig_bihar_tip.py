"""Traffic Injury Prevention figures: one column wide (7.6 cm / 3 in), Times New Roman, no grid lines, no titles on
the figure (captions carry them), legend at top or bottom, legible in greyscale. Writes 600-dpi greyscale TIFFs:
  outputs/tip_Figure1.tif  observed vs expected monthly deaths, Bihar, 2014-2019
  outputs/tip_Figure2.tif  monthly % difference in the first year, with the placebo year
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

P = pathlib.Path(__file__).resolve().parents[1]
K, DG, MG, LG, GR = "#000000", "#4d4d4d", "#8c8c8c", "#cfcfcf", "#ececec"
# allow_pickle: the npz holds a string array of dates and is written by our own src/bayes_bihar.py
z = np.load(P / "outputs/bayes_bihar_main.npz", allow_pickle=True)
zp = np.load(P / "outputs/bayes_bihar_placebo2015.npz", allow_pickle=True)
dates = pd.to_datetime(z["dates"]); mu = z["mu"].astype(float); y = z["y"]; mup = zp["mu"].astype(float)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})
t0 = pd.Timestamp("2016-04-01")


def save(fig, name):
    fig.savefig(P / f"outputs/{name}.png", dpi=600)
    Image.open(P / f"outputs/{name}.png").convert("L").save(P / f"outputs/{name}.tif", dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)


# ---- Figure 1 ----
fig, a = plt.subplots(figsize=(3.0, 2.7))
lo, md, hi = np.percentile(mu, [2.5, 50, 97.5], 0)
post = dates >= t0
a.axvspan(pd.Timestamp("2017-04-01"), dates.max() + pd.Timedelta(days=15), color=GR, lw=0, zorder=0)
a.fill_between(dates[post], lo[post], hi[post], color=LG, lw=0, label="Expected, 95% CrI")
a.plot(dates[post], md[post], color=DG, lw=0.9, ls=(0, (3, 1.5)), label="Expected, median")
a.plot(dates, y, color=K, lw=0.9, marker="o", ms=1.6, label="Observed")
a.axvline(t0, color=K, lw=0.6)
a.text(t0, 262, "Prohibition ", ha="right", va="bottom", fontsize=7)
a.text(pd.Timestamp("2017-05-01"), 262, "Not interpreted\ncausally", fontsize=6.5, color=DG, va="bottom")
a.set_ylim(250, 860); a.set_ylabel("Road deaths per month")
a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6.5, handlelength=1.6, columnspacing=0.8)
for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); save(fig, "tip_Figure1")

# ---- Figure 2 ----
win = (dates >= t0) & (dates < pd.Timestamp("2017-04-01"))
pl = (dates >= pd.Timestamp("2015-04-01")) & (dates < t0)
r = lambda m_, M: np.percentile(y[M] / m_[:, M], [2.5, 50, 97.5], 0)
rl, rm, rh = r(mu, win); _, pm, _ = r(mup, pl)
m = np.arange(12)
fig, b = plt.subplots(figsize=(3.0, 2.5))
b.fill_between(m, (rl - 1) * 100, (rh - 1) * 100, color=LG, lw=0)
b.plot(m, (rm - 1) * 100, color=K, lw=1.0, marker="o", ms=2.5, label="Apr 2016–Mar 2017 (95% CrI)")
b.plot(m, (pm - 1) * 100, color=MG, lw=0.9, ls=(0, (3, 1.5)), marker="s", ms=2.5, mfc="white", label="Placebo, Apr 2015–Mar 2016")
b.axhline(0, color=K, lw=0.6)
b.set_xticks(m); b.set_xticklabels(list("AMJJASONDJFM"))
b.set_xlabel("Month after start date"); b.set_ylabel("Observed vs expected (%)")
b.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=6.5, handlelength=1.8)
for s in ("top", "right"): b.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); save(fig, "tip_Figure2")
for n in ("tip_Figure1", "tip_Figure2"):
    im = Image.open(P / f"outputs/{n}.tif"); print(n, im.size, round(im.size[0] / 600 * 2.54, 1), "cm wide")
