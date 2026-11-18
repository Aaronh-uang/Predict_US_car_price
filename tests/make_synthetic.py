"""Generate a small fake used_cars_data.csv (same columns and quirks as the
Kaggle file) plus a fake ForCar CSV, to test the pipeline without the 9 GB
download. Prices follow a known formula, so we can see the model recover it.

    python tests/make_synthetic.py --rows 80000 --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CARS = [  # make, model, base 2020 price new, trims (name, premium), body, drive, fuel, engine, hp, disp, mpg
    ("Toyota", "Camry", 27000, [("LE", 1.0), ("SE", 1.05), ("XSE", 1.2), ("XLE", 1.15)], "Sedan", "FWD", "Gasoline", "I4", 203, 2500, (28, 39)),
    ("Toyota", "RAV4", 29000, [("LE AWD", 1.0), ("XLE AWD", 1.08), ("Limited AWD", 1.25)], "SUV / Crossover", "AWD", "Gasoline", "I4", 203, 2500, (27, 35)),
    ("Honda", "Civic", 23000, [("LX", 1.0), ("EX", 1.08), ("Sport", 1.06), ("Touring", 1.22)], "Sedan", "FWD", "Gasoline", "I4", 158, 2000, (30, 38)),
    ("Honda", "CR-V", 28000, [("LX AWD", 1.0), ("EX AWD", 1.08), ("Touring AWD", 1.25)], "SUV / Crossover", "AWD", "Gasoline", "I4", 190, 1500, (27, 32)),
    ("Ford", "F-150", 42000, [("XL SuperCrew 4WD", 1.0), ("XLT SuperCrew 4WD", 1.12), ("Lariat SuperCrew 4WD", 1.35), ("Raptor SuperCrew 4WD", 1.75)], "Pickup Truck", "4WD", "Gasoline", "V6", 375, 3500, (18, 23)),
    ("Ford", "Escape", 26000, [("S FWD", 1.0), ("SE FWD", 1.07), ("Titanium AWD", 1.25)], "SUV / Crossover", "FWD", "Gasoline", "I4", 181, 1500, (27, 33)),
    ("Chevrolet", "Silverado 1500", 41000, [("LT Crew Cab 4WD", 1.0), ("RST Crew Cab 4WD", 1.1), ("High Country Crew Cab 4WD", 1.4)], "Pickup Truck", "4WD", "Gasoline", "V8", 355, 5300, (16, 22)),
    ("BMW", "3 Series", 43000, [("330i", 1.0), ("330i xDrive", 1.05), ("M340i", 1.3)], "Sedan", "RWD", "Gasoline", "I4", 255, 2000, (26, 36)),
    ("Tesla", "Model 3", 45000, [("Standard Range Plus", 1.0), ("Long Range AWD", 1.15), ("Performance AWD", 1.3)], "Sedan", "AWD", "Electric", None, 283, None, (138, 124)),
    ("Jeep", "Wrangler", 35000, [("Sport 4WD", 1.0), ("Sahara 4WD", 1.15), ("Rubicon 4WD", 1.3)], "SUV / Crossover", "4WD", "Gasoline", "V6", 285, 3600, (17, 23)),
    ("Mazda", "MAZDA3", 23000, [("Select", 1.0), ("Preferred", 1.07), ("Premium AWD", 1.2)], "Hatchback", "FWD", "Gasoline", "I4", 186, 2500, (26, 35)),
    ("Porsche", "911", 105000, [("Carrera", 1.0), ("Carrera S", 1.12), ("Turbo S", 1.8)], "Coupe", "RWD", "Gasoline", "H6", 443, 3000, (18, 24)),
]
# How each model's price moved from 2020 to "today" (used by the fake ForCar file).
MARKET_MOVE = {"F-150": 1.45, "Silverado 1500": 1.40, "Model 3": 0.85, "Wrangler": 1.35, "911": 1.5}


def price_2020(base, prem, age, miles, acc, salvage, owners, manual, rng):
    dep = 0.86 ** np.clip(age, 0, None)
    mile_pen = np.exp(-miles / 160_000)
    p = base * prem * dep * mile_pen * np.where(acc == 1, 0.9, 1) * np.where(salvage == 1, 0.7, 1)
    p *= np.where(np.isnan(owners), 1, 1 - 0.03 * (np.nan_to_num(owners, nan=1) - 1))
    p *= np.where(manual, 0.95, 1)
    return np.maximum(p * np.exp(rng.normal(0, 0.08, len(p))), 600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=80_000)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = args.rows
    car_i = rng.integers(0, len(CARS), n)
    rows = []
    for i in range(len(CARS)):
        m = car_i == i
        k = int(m.sum())
        make, model, base, trims, body, drive, fuel, eng, hp, disp, mpg = CARS[i]
        t_i = rng.integers(0, len(trims), k)
        year = rng.integers(1997, 2022, k)
        listed = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 260, k), "D")
        age = (listed.year + (listed.dayofyear - 1) / 365.25).to_numpy() - year
        miles = np.clip(rng.normal(12_500, 4_000, k) * np.clip(age, 0.05, None), 5, 450_000)
        acc = rng.choice([1.0, 0.0, np.nan], k, p=[0.15, 0.6, 0.25])
        salv = rng.choice([1.0, 0.0, np.nan], k, p=[0.02, 0.7, 0.28])
        owners = rng.choice([1, 2, 3, 4, np.nan], k, p=[0.45, 0.25, 0.1, 0.05, 0.15])
        trans = rng.choice(["A", "M", "CVT", "Dual Clutch"], k, p=[0.75, 0.08, 0.15, 0.02])
        prem = np.array([trims[j][1] for j in t_i])
        price = price_2020(base, prem, age, miles, acc, salv, owners, trans == "M", rng)
        is_new = (age < 0.3) & (rng.random(k) < 0.7)
        def b(x):
            return np.where(np.isnan(x), "", np.where(x == 1, "True", "False"))
        rows.append(pd.DataFrame({
            "vin": [f"SYN{i:02d}{j:09d}" for j in range(k)],
            "back_legroom": "35.1 in",
            "body_type": body, "city": "Berkeley",
            "city_fuel_economy": mpg[0], "daysonmarket": rng.integers(0, 400, k),
            "description": np.where(rng.random(k) < 0.3, 'Clean car, "one owner"\nCall now, great deal!', "[!@@Additional Info@@!]Power"),
            "engine_cylinders": eng or "", "engine_displacement": disp if disp else "",
            "engine_type": eng or "", "fuel_type": fuel, "frame_damaged": b(np.where(rng.random(k) < 0.01, 1.0, np.where(np.isnan(salv), np.nan, 0.0))),
            "has_accidents": b(acc), "highway_fuel_economy": mpg[1], "horsepower": hp,
            "is_new": np.where(is_new, "True", "False"),
            "listed_date": listed.strftime("%Y-%m-%d"), "listing_id": rng.integers(1e8, 3e8, k),
            "major_options": "['Sunroof/Moonroof', 'Navigation System']",
            "make_name": make, "mileage": np.round(np.where(is_new, rng.integers(0, 30, k), miles)),
            "model_name": model, "owner_count": owners, "power": f"{hp} hp @ 5,500 RPM",
            "price": np.round(price), "salvage": b(salv), "torque": "",
            "transmission": trans, "trim_name": [trims[j][0] for j in t_i],
            "wheel_system": drive, "year": year,
        }))
    df = pd.concat(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    df["price"] = df["price"].astype(object)
    df["year"] = df["year"].astype(object)
    df.loc[df.sample(frac=0.01, random_state=1).index, "price"] = ""          # junk rows
    df.loc[df.sample(frac=0.005, random_state=2).index, "year"] = "not a year"
    df.to_csv(out / "used_cars_data.csv", index=False)

    # fake ForCar: today's average asking price by make/model/model year
    fc = []
    for make, model, base, trims, *_ in CARS:
        for my in range(2005, 2027):
            age = 2026.88 - my
            miles = 12_000 * max(age, 0.3)
            avg_prem = np.mean([t[1] for t in trims])
            p = price_2020(base, avg_prem, np.array([age]), np.array([miles]), np.array([0.0]),
                           np.array([0.0]), np.array([np.nan]), np.array([False]), np.random.default_rng(my))[0]
            p *= 1.3 * MARKET_MOVE.get(model, 1.0)
            name = model.replace("-", "") if model == "F-150" else model    # naming mismatch on purpose
            fc.append({"make": make, "model": name, "model_year": my, "listings": int(rng.integers(5, 900)),
                       "avg_price_usd": round(p), "min_price_usd": round(p * 0.6), "max_price_usd": round(p * 1.5),
                       "avg_mileage": round(miles), "forcar_url": ""})
    pd.DataFrame(fc).to_csv(out / "forcar.csv", index=False)
    print(f"wrote {len(df):,} listings and {len(fc)} ForCar rows to {out}/")


if __name__ == "__main__":
    main()
