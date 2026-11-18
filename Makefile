PY ?= .venv/bin/python
CSV ?= data/used_cars_data.csv

.PHONY: setup prep train export calibrate benchmark all serve test demo clean

setup:            ## create .venv and install training dependencies
	python3 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r training/requirements.txt

prep:             ## 9 GB CSV -> data/listings.parquet (10-20 min, once)
	$(PY) training/prep.py --csv $(CSV)

train:            ## train the model (a few minutes on an M3 Pro)
	$(PY) training/train.py

export:           ## write web/data/model.json + catalog
	$(PY) training/export.py

calibrate:        ## today's-market adjustment from ForCar
	$(PY) training/calibrate.py

benchmark:        ## score today's estimates against KBB (free AutoTrader sample)
	$(PY) training/benchmark.py

all: prep train export calibrate benchmark test

test:             ## check the browser model matches LightGBM
	node tests/parity.mjs

serve:            ## preview at http://localhost:8000
	cd web && python3 -m http.server 8000

demo:             ## try the whole pipeline on fake data (overwrites web/data!)
	$(PY) tests/make_synthetic.py --rows 80000 --out data/synthetic
	$(PY) training/prep.py --csv data/synthetic/used_cars_data.csv
	$(PY) training/train.py
	$(PY) training/export.py
	$(PY) training/calibrate.py --forcar data/synthetic/forcar.csv
	node tests/parity.mjs
	@echo "\nDemo built from FAKE data. Run 'make all' with the real CSV before deploying."

clean:
	rm -rf artifacts web/data data/listings.parquet
