"""Re-run the analysis: python run_pipeline.py [--bayes] [--build]
  default : synthetic DiD, trend fixed effects, recording check (~5 min), then verify.py
  --bayes : refit the Bayesian models (NumPyro, ~10 min each; pip install -r requirements-bayes.txt)
  --build : rebuild the time-of-day table from the raw reports (python scripts/fetch_raw_reports.py first)"""
import pathlib, subprocess, sys
R = pathlib.Path(__file__).resolve().parent
run = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=R, check=True)
if "--build" in sys.argv: run("src/time_of_day_mechanism.py")
for s in ("sdid_bihar", "trend_did", "recording_check"): run(f"src/{s}.py")
if "--bayes" in sys.argv: run("src/bayes_bihar.py"); run("src/bayes_bihar.py", 1)
run("src/fig_bihar_tip.py"); run("verify.py")
