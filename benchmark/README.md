# KBB benchmark data

`autotrader_kbb.csv.gz` grows each week (via `training/benchmark.py`) with used-car listings from the free
[Rebrowser AutoTrader sample](https://github.com/rebrowser/autotrader-dataset): year, make, model, trim,
mileage, listing type, seller ZIP and Kelley Blue Book's fair-price range. Listing prices are not included.

The data comes from Rebrowser and is used for research and non-commercial purposes with attribution, per its
terms. Rebrowser is not affiliated with AutoTrader or Kelley Blue Book. This project only uses the file to
measure how far its today's-market estimates sit from KBB's fair price; see `web/data/benchmark.json`.
