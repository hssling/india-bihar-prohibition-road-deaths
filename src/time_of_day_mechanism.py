"""Mechanism check: did Bihar's NIGHT road accidents fall more than its DAY accidents after prohibition (Apr 2016)?

Alcohol-involved crashes concentrate at night, so a prohibition effect working through drinking should cut
night accidents (18:00-06:00) more than day accidents (06:00-18:00). MoRTH 'Road Accidents in India' reports
state-wise accidents by 3-hour slot (accidents only, not deaths). Years with a text layer: 2014, 2015, 2016, 2017, 2019.

Design (annual, triple difference): log(night) - log(day) for Bihar vs other large states, 2016-17 vs 2014-15.
2016 is only 9 months treated, so 2017 (12 months, but Supreme Court highway liquor ban in donor states from
Apr 2017) is shown separately. Each row is checked: 8 slots sum to the printed total.
Output: data/time_of_day_state_year.csv, outputs/time_of_day_mechanism.csv
"""
import pathlib, re
import fitz, numpy as np, pandas as pd

P = pathlib.Path(__file__).resolve().parents[1]
RAW = P / "data/raw/downloads_2026_11"
PAGES = {2014: [84], 2015: [83], 2016: [116], 2017: [104], 2019: [165]}
STATES = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
          "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
          "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
          "Telangana", "Tripura", "Uttarakhand", "Uttar Pradesh", "West Bengal", "Delhi", "Puducherry", "Chandigarh"]
recs, bad = [], []
for y, pages in PAGES.items():
    d = fitz.open(RAW / f"rai_{y}_opencity.pdf")
    for pno in pages:
        toks = [re.sub(r"^\d{1,2}\s+(?=[A-Z])", "", t.strip()) for t in d[pno - 1].get_text().split("\n") if t.strip()]
        for k, tok in enumerate(toks):
            if tok not in STATES: continue
            nums = []
            for t2 in toks[k + 1:]:
                if re.fullmatch(r"[\d,]+", t2): nums.append(int(t2.replace(",", "")))
                else: break
            if len(nums) < 9: bad.append((y, tok, nums)); continue
            slots, total = nums[:8], nums[8]
            if sum(slots) != total: bad.append((y, tok, "sum != total")); continue
            recs.append(dict(year=y, state=tok, day=sum(slots[:4]), night=sum(slots[4:]), total=total))
df = pd.DataFrame(recs).drop_duplicates(["year", "state"])
df.to_csv(P / "data/time_of_day_state_year.csv", index=False)
print("rows per year:", df.groupby("year").size().to_dict(), "| rejected rows:", len(bad), bad[:6])
print(df[df.state == "Bihar"].assign(night_share=lambda x: (x.night / x.total).round(3)).to_string(index=False))

EXCL = ["Gujarat", "Kerala", "Mizoram", "Nagaland", "Manipur"]
w = df.set_index(["state", "year"])
lr = (np.log(w.night) - np.log(w.day)).unstack("year")                 # log night/day ratio
big = w.xs(2014, level="year").total
big = big[(big >= 3000) & ~big.index.isin(EXCL)].index
pre = lr[[2014, 2015]].mean(1)
rows = []
for yr in (2016, 2017):                     # 2019 table layout differs (rows fail the sum check)
    ch = (lr[yr] - pre).reindex(big).dropna()
    b, oth = ch["Bihar"], ch.drop("Bihar")
    rows.append(dict(year=yr, bihar_change_pct=100 * np.expm1(b), others_median_pct=100 * np.expm1(oth.median()),
                     triple_diff_pct=100 * np.expm1(b - oth.median()), bihar_rank=f"{(ch <= b).sum()}/{len(ch)}",
                     note={2016: "9 of 12 months treated", 2017: "SC highway liquor ban in donors from Apr 2017",
                           2019: "late; fines and other changes"}[yr]))
res = pd.DataFrame(rows)
res.to_csv(P / "outputs/time_of_day_mechanism.csv", index=False)
print("\nChange in log(night/day accidents) vs 2014-15 mean (negative = night fell more than day):")
print(res.round(1).to_string(index=False))
