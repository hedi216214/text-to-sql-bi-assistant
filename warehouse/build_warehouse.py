"""Build a small retail data warehouse (star schema) in SQLite with simulated data.

Tables:
    dim_date      one row per day of 2024-2025
    dim_product   products with category and unit cost
    dim_store     stores with city and region
    dim_customer  B2C and B2B customers
    fact_sales    one row per sales line (grain: product x customer x store x day)

Usage:
    python warehouse/build_warehouse.py --out warehouse/retail.db
"""
import argparse
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE dim_date (
    date_key     INTEGER PRIMARY KEY,   -- YYYYMMDD
    full_date    TEXT NOT NULL,         -- ISO date 'YYYY-MM-DD'
    year         INTEGER NOT NULL,
    quarter      INTEGER NOT NULL,      -- 1..4
    month        INTEGER NOT NULL,      -- 1..12
    month_name   TEXT NOT NULL,
    day_of_week  TEXT NOT NULL,         -- 'Monday'..'Sunday'
    is_weekend   INTEGER NOT NULL       -- 1 = Saturday or Sunday
);
CREATE TABLE dim_product (
    product_key  INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category     TEXT NOT NULL,
    brand        TEXT NOT NULL,
    unit_cost    REAL NOT NULL          -- purchase cost in TND
);
CREATE TABLE dim_store (
    store_key    INTEGER PRIMARY KEY,
    store_name   TEXT NOT NULL,
    city         TEXT NOT NULL,
    region       TEXT NOT NULL          -- 'North', 'Center', 'South'
);
CREATE TABLE dim_customer (
    customer_key INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    segment      TEXT NOT NULL,         -- 'B2C' or 'B2B'
    city         TEXT NOT NULL,
    signup_date  TEXT NOT NULL
);
CREATE TABLE fact_sales (
    sale_id      INTEGER PRIMARY KEY,
    date_key     INTEGER NOT NULL REFERENCES dim_date(date_key),
    product_key  INTEGER NOT NULL REFERENCES dim_product(product_key),
    store_key    INTEGER NOT NULL REFERENCES dim_store(store_key),
    customer_key INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    quantity     INTEGER NOT NULL,
    unit_price   REAL NOT NULL,         -- selling price in TND before discount
    discount_pct REAL NOT NULL,         -- 0.0 to 0.3
    revenue      REAL NOT NULL,         -- quantity * unit_price * (1 - discount_pct)
    cost         REAL NOT NULL          -- quantity * unit_cost
);
"""

CATEGORIES = {
    "Skincare": (["Hydrating Cream", "Sunscreen SPF50", "Cleansing Gel", "Night Serum",
                  "Repair Balm"], (12, 45)),
    "Hair care": (["Anti-dandruff Shampoo", "Hair Mask", "Strengthening Serum",
                   "Dry Shampoo"], (9, 30)),
    "Vitamins & supplements": (["Vitamin C 1000", "Magnesium B6", "Omega 3", "Iron Complex",
                                "Multivitamin Kids"], (10, 40)),
    "Baby care": (["Baby Wash", "Diaper Cream", "Baby Wipes", "Baby Lotion"], (6, 22)),
    "Oral care": (["Whitening Toothpaste", "Mouthwash", "Sensitive Toothpaste",
                   "Dental Floss"], (3, 14)),
    "Medical devices": (["Digital Thermometer", "Blood Pressure Monitor", "Glucose Strips",
                         "Nebulizer"], (15, 120)),
}
BRANDS = ["Avene", "Bioderma", "La Roche", "Vichy", "Nuxe", "Uriage", "Mustela", "Omron"]
STORES = [("Tunis Centre", "Tunis", "North"), ("Lac 2", "Tunis", "North"),
          ("Ariana Soukra", "Ariana", "North"), ("Sousse Corniche", "Sousse", "Center"),
          ("Monastir Marina", "Monastir", "Center"), ("Sfax Ville", "Sfax", "South"),
          ("Gabes", "Gabes", "South")]
CITIES = ["Tunis", "Ariana", "Ben Arous", "Sousse", "Monastir", "Sfax", "Nabeul", "Gabes"]


def build(out: Path, n_sales: int = 60000, seed: int = 7):
    rng = np.random.default_rng(seed)
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)

    days = [date(2024, 1, 1) + timedelta(d) for d in range(731)]
    con.executemany("INSERT INTO dim_date VALUES (?,?,?,?,?,?,?,?)", [
        (int(d.strftime("%Y%m%d")), d.isoformat(), d.year, (d.month - 1) // 3 + 1, d.month,
         d.strftime("%B"), d.strftime("%A"), int(d.weekday() >= 5)) for d in days])

    products, price_of = [], {}
    k = 1
    for cat, (names, (lo, hi)) in CATEGORIES.items():
        for name in names:
            cost = round(float(rng.uniform(lo, hi)), 2)
            products.append((k, name, cat, str(rng.choice(BRANDS)), cost))
            price_of[k] = round(cost * float(rng.uniform(1.35, 1.8)), 2)
            k += 1
    con.executemany("INSERT INTO dim_product VALUES (?,?,?,?,?)", products)
    con.executemany("INSERT INTO dim_store VALUES (?,?,?,?)",
                    [(i + 1, *s) for i, s in enumerate(STORES)])

    customers = []
    for i in range(1, 1201):
        b2b = rng.random() < 0.12
        name = f"Clinic {i:04d}" if b2b else f"Customer {i:04d}"
        signup = days[int(rng.integers(0, 500))].isoformat()
        customers.append((i, name, "B2B" if b2b else "B2C", str(rng.choice(CITIES)), signup))
    con.executemany("INSERT INTO dim_customer VALUES (?,?,?,?,?)", customers)

    # Seasonality: sunscreen peaks in summer, vitamins in winter; sales grow ~12% in 2025.
    day_w = np.array([(1.12 if d.year == 2025 else 1.0) * (1.25 if d.weekday() == 5 else 1.0)
                      for d in days])
    day_idx = rng.choice(len(days), n_sales, p=day_w / day_w.sum())
    rows = []
    for sid, di in enumerate(day_idx, start=1):
        d = days[di]
        pk = int(rng.integers(1, len(products) + 1))
        cat = products[pk - 1][2]
        if cat == "Skincare" and d.month in (6, 7, 8) and rng.random() < 0.35:
            pk = 2  # Sunscreen SPF50 sells more in summer
        if cat == "Vitamins & supplements" and d.month in (11, 12, 1, 2) and rng.random() < 0.3:
            pk = 11  # Vitamin C in winter
        ck = int(rng.integers(1, len(customers) + 1))
        b2b = customers[ck - 1][2] == "B2B"
        qty = int(rng.integers(5, 40)) if b2b else int(rng.integers(1, 4))
        disc = round(float(rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2, 0.3])), 2)
        if b2b:
            disc = max(disc, 0.1)
        price = price_of[pk]
        rows.append((sid, int(d.strftime("%Y%m%d")), pk, int(rng.integers(1, len(STORES) + 1)),
                     ck, qty, price, disc, round(qty * price * (1 - disc), 2),
                     round(qty * products[pk - 1][4], 2)))
    con.executemany("INSERT INTO fact_sales VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    print(f"Built {out}: {len(products)} products, {len(STORES)} stores, "
          f"{len(customers)} customers, {len(rows)} sales lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("retail.db")))
    build(Path(ap.parse_args().out))
