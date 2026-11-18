# US Used Car Price Estimator

Estimate what a used car is worth from its year, make, model, trim, mileage and history, or paste a VIN. The site is fully static: the trained model runs **in your browser**. There's no server to fall asleep, nothing to maintain, and it's hosted free on GitHub Pages.

**Live site:** `https://aaronh-uang.github.io/Predict_US_car_price/` (after the first deploy)

## What it does

- **Make → model → trim picker.** Picking a trim auto-fills engine, horsepower, drivetrain and mpg from real listings.
- **VIN lookup.** Decodes the VIN through NHTSA's free vPIC API, then matches it to the closest make, model and trim in the data.
- **Location, options and certified pre-owned.** All optional. A blank ZIP gives a nationwide price for buyers who'd shop anywhere or have the car shipped, and "Not sure" leaves options out. During training, location and options are hidden on 15% of cars, so the model has a proper answer for "unknown". The likely range is wider when they're left out.
- **Likely price range.** An 80% range calibrated on held-out cars, not just one number.
- **Depreciation charts.** Value by mileage, and projected value over the next 8 years at your current pace of driving.
- **Today's market vs 2020 market.** The model learns from 2020 listings. A per-model adjustment from current ForCar asking prices brings estimates up to date, and it refreshes weekly on its own.

## How it works

```
Kaggle CSV (9 GB, 3M listings)
   │  training/prep.py       keep ~25 columns, used cars only, dedupe VINs → data/listings.parquet
   ▼
training/train.py            LightGBM on log(price); make/model/trim as real categoricals;
   │                         common options as 0/1 flags, location as the 3-digit ZIP area centre;
   │                         price can only fall with mileage/age/owners/damage (monotone);
   │                         80% range from a held-out calibration split
   ▼
training/export.py           trees → web/data/model.json, specs → web/data/catalog/*.json
   ▼
training/calibrate.py        ForCar today's prices by make/model/model year → web/data/calibration.json
   ▼
web/                         static site: index.html + js/ (no build step, no dependencies)
```

The browser evaluator (`web/js/model.js`) follows LightGBM's own decision rules, and `tests/parity.mjs` checks it against LightGBM's predictions (they agree to within 1e-6 in log-price).

### Today's-market adjustment

For every ForCar row (for example "2019 Lexus RX: avg $X at Y miles, N listings"), the model predicts what that car would have listed for in 2020 at the same age and mileage. The log ratio becomes three parts:

`today = 2020 estimate × exp(age_factor[age] + make_offset[band] + model_offset[band])`

- `band` splits cars into under 8, 8–15 and 16+ years old, so a model's 20-year-old examples can't drag down the price of a 5-year-old one.
- When ForCar names a family ("RX", "Silverado", "F-450") and the 2020 data has the versions ("RX 350", "Silverado 1500", "F-450 Super Duty"), the price applies to the whole family, weighted by listings. A few naming differences (BMW 328i → 3 Series, Mercedes ML-Class → M-Class, Ford E-350 → E-Series) are mapped by hand.
- Offsets are shrunk toward zero when a make or model has few listings.

It corrects price *levels* per model, while trim, mileage and history effects still come from the 2020 data. `artifacts/calibration_rows.csv` holds the working for every ForCar row, if you want to check a model.

## Run it on your Mac

Requirements: Python 3.11+, Node 18+ (only for the test), and about 12 GB free disk.

```bash
# 1. get the data (9 GB) - either download the zip from
#    https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset
#    and put used_cars_data.csv in ./data/, or use the Kaggle CLI:
pip install kaggle   # needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create token)
kaggle datasets download -d ananaymital/us-used-cars-dataset -p data --unzip

# 2. set up and run everything
make setup          # .venv + numpy, pandas, pyarrow, lightgbm
make all            # prep → train → export → calibrate → parity test
make serve          # http://localhost:8000
```

`make prep` takes 10–20 minutes the first time. After that, `make train export calibrate` is a few minutes. For quick iterations, run `.venv/bin/python training/train.py --sample 300000`.

If `import lightgbm` fails with a `libomp` error on macOS, run `brew install libomp`.

Want to try the pipeline before downloading? `make demo` builds everything from a small fake dataset. It overwrites `web/data/`, so run `make all` with the real CSV afterwards.

## Deploy

1. Commit `web/data/` and `artifacts/` after training. Don't commit `data/`, which is git-ignored.
2. On GitHub: **Settings → Pages → Source: GitHub Actions**.
3. Push to `main`. `.github/workflows/deploy.yml` runs the parity test and publishes `web/`.

Every Monday the same workflow downloads ForCar's latest prices, re-runs `calibrate.py`, commits the new `calibration.json` and redeploys. You can also trigger it by hand from the Actions tab.

## Project layout

```
training/   common.py (features, shared helpers) · prep.py · train.py · export.py · calibrate.py
web/        index.html · styles.css · js/{app,model,chart,vin}.js · data/ (generated)
tests/      parity.mjs (JS vs LightGBM) · make_synthetic.py (fake data for `make demo`)
artifacts/  model.txt · meta.json · parity_fixture.json (generated, committed)
```

## Fixes from the first version

- The app sent model, trim and engine type as `"Unknown"`, so a Camry and a Corolla priced almost the same. They're now real inputs.
- The categories were integer codes treated as numbers (alphabetical order of 8,427 trims). They're now native categoricals.
- Manual transmissions were mapped to "automatic". The raw `A` / `M` / `CVT` / `Dual Clutch` values are now kept.
- "Truck" wasn't a training category ("Pickup Truck" is), and torque and owner count inputs were ignored. Every input on the page now reaches the model.
- Days on market was an input. A buyer doesn't know it, so it was dropped.
- Age is now measured from each listing's date, not a fixed year, and duplicate VINs are removed so the same car can't be in both train and test.

## Data and credits

- Training data: [US Used Cars dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset) (Kaggle), used-car dealer listings from about 2020.
- Current prices: [ForCar open data](https://forcar.org/open-data/), CC BY 4.0, average asking price by make/model/model year, rebuilt daily.
- VIN decoding: [NHTSA vPIC API](https://vpic.nhtsa.dot.gov/api/).

Estimates are statistical, not an appraisal. Condition, options, color and location also move real prices.

---
Created by Yonglang (Aaron) Huang · MIT License
