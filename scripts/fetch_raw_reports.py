"""Download the MoRTH 'Road Accidents in India' reports (archived with the MV Act fines study, doi:10.5281/zenodo.24055110)
into data/raw/, needed only to rebuild data/time_of_day_state_year.csv with src/time_of_day_mechanism.py."""
import io, pathlib, urllib.request, zipfile
R = pathlib.Path(__file__).resolve().parents[1]
URL = "https://github.com/hssling/india-mva2019-fines-road-deaths/releases/download/v1.0.0/raw_pdfs.zip"
with urllib.request.urlopen(URL) as r:
    zipfile.ZipFile(io.BytesIO(r.read())).extractall(R / "data/raw")
print("saved to data/raw/ (SHA-256 manifest: data/raw/MANIFEST.csv in the fines repository)")
