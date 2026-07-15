"""
Synthetic data generator for CardOperationScoring.

Builds an OLTP-shaped dataset:
    mcc_dictionary, customers, cards, devices, merchants,
    transactions, fraud_labels

Design choices that matter (and are defensible at interview):
  * Per-customer behavioural profiles are HIDDEN generator state
    (typical amount, active hours, home coords). They drive generation
    but are never stored -- the model must *derive* them later.
  * FRAUD is injected as bursty EPISODES on a compromised card, so it
    naturally carries the legend's top signals: velocity, new/shared
    device, geo anomaly + impossible travel, amount deviation,
    high-risk merchant, odd hours, decline-then-approve.
  * Signal is NOISY on purpose: legit customers also travel and spend
    big sometimes, so nothing is perfectly separable (no fake 0.99 AUC).
  * Labels are DELAYED: reported_at = authorized_at + delay. A small
    share of fraud is left UNLABELED (undetected) and a tiny share of
    legit is mislabeled -- realistic, imperfect ground truth.

Raw columns stay lean; the ~40 model features are derived downstream.

Run:
    pip install numpy pandas psycopg2-binary
    python generate.py                      # 600k tx into Postgres
    python generate.py --dry-run            # build only, print stats, no DB
"""

import argparse
import io
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
# (name, lat, lon) -- Russian cities where customers live / spend
CITIES = [
    ("Moscow", 55.7558, 37.6173), ("Saint Petersburg", 59.9343, 30.3351),
    ("Novosibirsk", 55.0084, 82.9357), ("Yekaterinburg", 56.8389, 60.6057),
    ("Kazan", 55.7963, 49.1088), ("Nizhny Novgorod", 56.2965, 43.9361),
    ("Chelyabinsk", 55.1644, 61.4368), ("Samara", 53.1959, 50.1002),
    ("Rostov-on-Don", 47.2357, 39.7015), ("Ufa", 54.7388, 55.9721),
    ("Krasnoyarsk", 56.0153, 92.8932), ("Perm", 58.0105, 56.2502),
    ("Voronezh", 51.6720, 39.1843), ("Volgograd", 48.7080, 44.5133),
    ("Krasnodar", 45.0355, 38.9753),
]

# Foreign countries used for legit travel and for fraud (name -> lat, lon)
FOREIGN = {
    "TR": (39.93, 32.86), "AE": (25.20, 55.27), "TH": (13.75, 100.50),
    "CN": (39.90, 116.40), "US": (40.71, -74.01), "GB": (51.51, -0.13),
    "DE": (52.52, 13.40), "GE": (41.72, 44.78),
}

# (mcc, description, category, base_risk 0..3)
MCCS = [
    ("5411", "Grocery Stores", "grocery", 0), ("5812", "Restaurants", "food", 0),
    ("5541", "Gas Stations", "fuel", 0), ("4111", "Local Transport", "transport", 0),
    ("5311", "Department Stores", "retail", 0), ("4814", "Telecom", "telecom", 1),
    ("5999", "Misc Retail", "retail", 1), ("5732", "Electronics", "electronics", 2),
    ("5944", "Jewelry", "luxury", 2), ("7011", "Hotels", "travel", 1),
    ("4511", "Airlines", "travel", 1), ("5967", "Direct Marketing", "cnp", 2),
    ("6011", "ATM Cash", "cash", 2), ("4829", "Money Transfer", "transfer", 3),
    ("7995", "Gambling", "gambling", 3),
]

# customer segments: (name, prob, lognormal mu, lognormal sigma) for amount (RUB)
SEGMENTS = [
    ("mass", 0.75, 7.4, 0.7),      # ~ e^7.4 ≈ 1600 RUB median
    ("affluent", 0.20, 8.4, 0.8),  # ~ 4400 RUB
    ("private", 0.05, 9.3, 0.9),   # ~ 11000 RUB
]

RISK_TIER = {0: "low", 1: "low", 2: "medium", 3: "high"}


def build_data(n_tx, n_customers, fraud_rate, seed):
    rng = np.random.default_rng(seed)
    date_start = np.datetime64("2024-01-01T00:00:00")
    n_days = 455  # ~15 months

    # ---- mcc dictionary -------------------------------------------------
    mcc_df = pd.DataFrame(MCCS, columns=["mcc", "description", "category", "base_risk"])

    # ---- customers (+ hidden behavioural state) -------------------------
    C = n_customers
    seg_idx = rng.choice(len(SEGMENTS), size=C, p=[s[1] for s in SEGMENTS])
    seg_mu = np.array([SEGMENTS[i][2] for i in seg_idx])
    seg_sigma = np.array([SEGMENTS[i][3] for i in seg_idx])
    home_city = rng.integers(0, len(CITIES), size=C)
    city_lat = np.array([c[1] for c in CITIES])
    city_lon = np.array([c[2] for c in CITIES])
    home_lat = city_lat[home_city] + rng.normal(0, 0.03, C)
    home_lon = city_lon[home_city] + rng.normal(0, 0.03, C)
    peak_hour = rng.integers(8, 22, size=C)          # hidden
    activity = rng.gamma(2.0, 1.0, size=C)           # hidden relative activity
    activity_p = activity / activity.sum()

    customers_df = pd.DataFrame({
        "customer_id": np.arange(1, C + 1),
        "home_country": "RU",
        "home_city": [CITIES[i][0] for i in home_city],
        "segment": [SEGMENTS[i][0] for i in seg_idx],
        "signup_at": date_start - rng.integers(30, 1500, C) * np.timedelta64(1, "D"),
        "home_lat": home_lat.round(6),
        "home_lon": home_lon.round(6),
    })

    # ---- cards (1..3 per customer) --------------------------------------
    n_cards = rng.integers(1, 4, size=C)
    first_card = np.concatenate([[0], np.cumsum(n_cards)])[:-1] + 1
    card_customer = np.repeat(np.arange(1, C + 1), n_cards)
    n_total_cards = int(n_cards.sum())
    cards_df = pd.DataFrame({
        "card_id": np.arange(1, n_total_cards + 1),
        "customer_id": card_customer,
        "issued_at": date_start - rng.integers(10, 1200, n_total_cards) * np.timedelta64(1, "D"),
    })

    # ---- devices (1..2 own per customer + a shared FRAUD pool) ----------
    n_dev = rng.integers(1, 3, size=C)
    first_dev = np.concatenate([[0], np.cumsum(n_dev)])[:-1] + 1
    n_own_dev = int(n_dev.sum())
    n_fraud_dev = 60  # small shared pool -> high "cards per device" signal
    dev_types = np.array(["mobile", "web", "pos", "atm"])
    devices_df = pd.DataFrame({
        "device_id": np.arange(1, n_own_dev + n_fraud_dev + 1),
        "device_type": rng.choice(dev_types, size=n_own_dev + n_fraud_dev, p=[.55, .3, .1, .05]),
        "first_seen_at": date_start + rng.integers(0, n_days, n_own_dev + n_fraud_dev) * np.timedelta64(1, "D"),
    })
    fraud_dev_ids = np.arange(n_own_dev + 1, n_own_dev + n_fraud_dev + 1)

    # ---- merchants ------------------------------------------------------
    M = 3000
    base_risk_arr = mcc_df["base_risk"].values
    # low-risk MCCs dominate the merchant population
    mcc_pop_w = np.where(base_risk_arr == 0, 6, np.where(base_risk_arr == 1, 3,
                 np.where(base_risk_arr == 2, 1.2, 0.4)))
    mcc_pop_w = mcc_pop_w / mcc_pop_w.sum()
    m_mcc_idx = rng.choice(len(MCCS), size=M, p=mcc_pop_w)
    merchants_df = pd.DataFrame({
        "merchant_id": np.arange(1, M + 1),
        "mcc": mcc_df["mcc"].values[m_mcc_idx],
        "merchant_country": rng.choice(["RU", "RU", "RU", "TR", "CN", "US"], size=M),
        "descriptor": [f"MERCHANT_{i:05d}" for i in range(1, M + 1)],
        "risk_segment": [RISK_TIER[base_risk_arr[i]] for i in m_mcc_idx],
    })
    m_base_risk = base_risk_arr[m_mcc_idx]

    # ==================================================================
    # BASELINE (legit) transactions
    # ==================================================================
    n_fraud_target = int(round(n_tx * fraud_rate))
    n_base = n_tx - n_fraud_target

    cust = rng.choice(C, size=n_base, p=activity_p)            # 0-based customer idx
    # timestamp: day uniform, hour ~ around the customer's peak hour
    day = rng.integers(0, n_days, size=n_base)
    hour = np.clip(rng.normal(peak_hour[cust], 3), 0, 23).astype(int)
    night = rng.random(n_base) < 0.08                        # ~8% legit night owls
    hour = np.where(night, rng.integers(0, 6, n_base), hour)
    minute = rng.integers(0, 60, size=n_base)
    ts = (date_start + day * np.timedelta64(1, "D")
          + hour * np.timedelta64(1, "h") + minute * np.timedelta64(1, "m"))

    amount = np.exp(rng.normal(seg_mu[cust], seg_sigma[cust])).round(2)
    card = first_card[cust] + rng.integers(0, n_cards[cust])
    device = first_dev[cust] + rng.integers(0, n_dev[cust])

    # merchants: baseline favours low risk
    base_merch_w = np.where(m_base_risk == 0, 6, np.where(m_base_risk == 1, 3,
                    np.where(m_base_risk == 2, 1.0, 0.2)))
    base_merch_w = base_merch_w / base_merch_w.sum()
    merch = rng.choice(M, size=n_base, p=base_merch_w) + 1

    # geography: mostly at home, ~3% legit domestic-travel, ~1.5% foreign trip
    lat = home_lat[cust] + rng.normal(0, 0.05, n_base)
    lon = home_lon[cust] + rng.normal(0, 0.05, n_base)
    country = np.array(["RU"] * n_base, dtype=object)
    trip = rng.random(n_base) < 0.015
    fkeys = list(FOREIGN.keys())
    tf = rng.choice(len(fkeys), size=trip.sum())
    country[trip] = np.array(fkeys)[tf]
    lat[trip] = np.array([FOREIGN[fkeys[i]][0] for i in tf]) + rng.normal(0, 0.1, trip.sum())
    lon[trip] = np.array([FOREIGN[fkeys[i]][1] for i in tf]) + rng.normal(0, 0.1, trip.sum())

    channel = rng.choice(["mobile", "web", "pos", "atm", "ecom"], size=n_base,
                         p=[.35, .15, .30, .05, .15])
    is_ecom = np.isin(channel, ["web", "ecom"])
    entry_mode = np.where(is_ecom, "cnp", "card_present")
    ip_country = np.where(is_ecom, country, None)             # CNP -> matches, mostly
    approved = rng.random(n_base) > 0.03
    resp = np.where(approved, None, "51")                     # 51 = insufficient funds

    base = pd.DataFrame({
        "customer_id": cust + 1, "card_id": card, "device_id": device,
        "merchant_id": merch, "amount": amount, "currency": "RUB",
        "country": country, "city": np.where(country == "RU",
                                             customers_df["home_city"].values[cust], None),
        "channel": channel, "entry_mode": entry_mode,
        "is_ecommerce": is_ecom, "is_recurring": rng.random(n_base) < 0.05,
        "is_tokenized": rng.random(n_base) < 0.4,
        "auth_result": np.where(approved, "approved", "declined"),
        "response_code": resp, "authorized_at": ts,
        "lat": lat.round(6), "lon": lon.round(6), "ip_country": ip_country,
        "_is_fraud": False,
    })

    # ==================================================================
    # FRAUD episodes
    # ==================================================================
    fraud_chunks = []
    produced = 0
    high_risk_merch = np.where(m_base_risk >= 2)[0] + 1
    cnp_merch = np.where(np.isin(merchants_df["mcc"].values, ["5967", "4829", "7995"]))[0] + 1
    cash_merch = np.where(merchants_df["mcc"].values == "6011")[0] + 1
    if len(cnp_merch) == 0:
        cnp_merch = high_risk_merch
    if len(cash_merch) == 0:
        cash_merch = high_risk_merch
    all_merch = np.arange(1, M + 1)
    ffkeys = ["TR", "CN", "US", "AE", "GE"]
    archs = ["foreign_cnp", "card_testing", "domestic_ato", "atm_cashout"]
    arch_p = [0.35, 0.20, 0.25, 0.20]

    def add(v, card, dev, mid, amt, country, ip, chan, entry, ecom, appr, resp, t, lat, lon):
        fraud_chunks.append({
            "customer_id": v + 1, "card_id": int(card), "device_id": int(dev),
            "merchant_id": int(mid), "amount": round(float(amt), 2), "currency": "RUB",
            "country": country, "city": None, "channel": chan, "entry_mode": entry,
            "is_ecommerce": ecom, "is_recurring": False, "is_tokenized": False,
            "auth_result": "approved" if appr else "declined",
            "response_code": None if appr else resp, "authorized_at": t,
            "lat": round(float(lat), 6), "lon": round(float(lon), 6),
            "ip_country": ip, "_is_fraud": True,
        })

    while produced < n_fraud_target:
        v = int(rng.integers(0, C))
        card = int(first_card[v] + rng.integers(0, n_cards[v]))
        dev = int(rng.choice(fraud_dev_ids))                 # shared pool -> cards-per-device signal
        typ = float(np.exp(seg_mu[v]))
        day0 = int(rng.integers(0, n_days))
        arch = archs[int(rng.choice(len(archs), p=arch_p))]

        if arch == "foreign_cnp":                            # abroad, night, big amounts
            burst = int(rng.integers(3, 10))
            fc = ffkeys[int(rng.integers(0, len(ffkeys)))]
            lat0, lon0 = FOREIGN[fc]
            h0 = int(rng.integers(0, 6))
            for i in range(burst):
                t = (date_start + day0 * np.timedelta64(1, "D") + h0 * np.timedelta64(1, "h")
                     + i * int(rng.integers(2, 25)) * np.timedelta64(1, "m"))
                appr = not (i < 2 and rng.random() < 0.5)
                add(v, card, dev, int(rng.choice(cnp_merch)), typ * rng.uniform(3, 15),
                    fc, fc, "ecom", "cnp", True, appr, "05", t,
                    lat0 + rng.normal(0, 0.1), lon0 + rng.normal(0, 0.1))
            produced += burst

        elif arch == "card_testing":                         # many tiny ops, many declines
            burst = int(rng.integers(6, 20))
            dom = rng.random() < 0.5
            fc = ffkeys[int(rng.integers(0, len(ffkeys)))]
            country = "RU" if dom else fc
            ip = "RU" if dom else fc
            if dom:
                ci = int(rng.integers(0, len(CITIES))); lat0, lon0 = city_lat[ci], city_lon[ci]
            else:
                lat0, lon0 = FOREIGN[fc]
            h0 = int(rng.integers(0, 24))
            for i in range(burst):
                t = (date_start + day0 * np.timedelta64(1, "D") + h0 * np.timedelta64(1, "h")
                     + i * int(rng.integers(1, 5)) * np.timedelta64(1, "m"))
                appr = rng.random() < 0.35                   # testing -> mostly declined
                add(v, card, dev, int(rng.choice(cnp_merch)), rng.uniform(30, 400),
                    country, ip, "ecom", "cnp", True, appr, "05", t,
                    lat0 + rng.normal(0, 0.1), lon0 + rng.normal(0, 0.1))
            produced += burst

        elif arch == "domestic_ato":                         # inside RU, daytime, IP matches
            burst = int(rng.integers(3, 8))
            ci = int(rng.integers(0, len(CITIES))); lat0, lon0 = city_lat[ci], city_lon[ci]
            h0 = int(np.clip(rng.normal(14, 4), 0, 23))
            for i in range(burst):
                t = (date_start + day0 * np.timedelta64(1, "D") + h0 * np.timedelta64(1, "h")
                     + i * int(rng.integers(3, 40)) * np.timedelta64(1, "m"))
                chan = "web" if rng.random() < 0.5 else "mobile"
                add(v, card, dev, int(rng.choice(all_merch)), typ * rng.uniform(2, 8),
                    "RU", "RU", chan, "cnp", True, True, "05", t,
                    lat0 + rng.normal(0, 0.05), lon0 + rng.normal(0, 0.05))
            produced += burst

        else:                                                # atm_cashout: card-present cash
            burst = int(rng.integers(2, 6))
            ci = int(rng.integers(0, len(CITIES))); lat0, lon0 = city_lat[ci], city_lon[ci]
            h0 = int(rng.integers(0, 24))
            for i in range(burst):
                t = (date_start + day0 * np.timedelta64(1, "D") + h0 * np.timedelta64(1, "h")
                     + i * int(rng.integers(5, 30)) * np.timedelta64(1, "m"))
                add(v, card, dev, int(rng.choice(cash_merch)), rng.uniform(5000, 60000),
                    "RU", None, "atm", "card_present", False, True, "05", t,
                    lat0 + rng.normal(0, 0.05), lon0 + rng.normal(0, 0.05))
            produced += burst

    fraud = pd.DataFrame(fraud_chunks)

    # ==================================================================
    # Combine, order by time, assign ids
    # ==================================================================
    tx = pd.concat([base, fraud], ignore_index=True)
    tx = tx.sort_values("authorized_at").reset_index(drop=True)
    tx.insert(0, "transaction_id", np.arange(1, len(tx) + 1))

    # ---- labels with DELAY + label noise --------------------------------
    fraud_mask = tx["_is_fraud"].values
    fraud_ids = tx.loc[fraud_mask, "transaction_id"].values
    fraud_ts = tx.loc[fraud_mask, "authorized_at"].values
    keep = rng.random(len(fraud_ids)) > 0.05                 # ~5% fraud stays undetected
    labeled_ids = fraud_ids[keep]
    labeled_ts = fraud_ts[keep]
    delay_days = rng.integers(3, 90, size=len(labeled_ids))
    reported = labeled_ts + delay_days * np.timedelta64(1, "D")

    # tiny false-positive labels on legit tx (chargeback abuse)
    legit_ids = tx.loc[~fraud_mask, "transaction_id"].values
    legit_ts = tx.loc[~fraud_mask, "authorized_at"].values
    fp_n = max(1, int(len(legit_ids) * 0.0002))
    fp_sel = rng.choice(len(legit_ids), size=fp_n, replace=False)
    fp_reported = legit_ts[fp_sel] + rng.integers(10, 120, fp_n) * np.timedelta64(1, "D")

    all_tx_ids = np.concatenate([labeled_ids, legit_ids[fp_sel]])
    all_reported = np.concatenate([reported, fp_reported])
    all_type = np.concatenate([
        rng.choice(["confirmed_fraud", "chargeback", "dispute"], len(labeled_ids), p=[.6, .3, .1]),
        np.array(["dispute"] * fp_n)])
    all_source = np.concatenate([
        rng.choice(["manual_investigation", "chargeback", "customer_report"], len(labeled_ids), p=[.4, .4, .2]),
        np.array(["customer_report"] * fp_n)])

    labels_df = pd.DataFrame({
        "label_id": np.arange(1, len(all_tx_ids) + 1),
        "transaction_id": all_tx_ids,
        "label_type": all_type,
        "source": all_source,
        "reported_at": all_reported,
    }).sort_values("transaction_id").reset_index(drop=True)
    labels_df["label_id"] = np.arange(1, len(labels_df) + 1)

    tx = tx.drop(columns=["_is_fraud"])
    return {
        "mcc_dictionary": mcc_df, "customers": customers_df, "cards": cards_df,
        "devices": devices_df, "merchants": merchants_df,
        "transactions": tx, "fraud_labels": labels_df,
    }, fraud_mask


# --------------------------------------------------------------------------
# Postgres load (COPY)
# --------------------------------------------------------------------------
LOAD_ORDER = ["mcc_dictionary", "customers", "cards", "devices",
              "merchants", "transactions", "fraud_labels"]

BOOL_COLS = {"is_ecommerce", "is_recurring", "is_tokenized"}


def _copy(cur, df, table):
    out = df.copy()
    for c in out.columns:
        if c in BOOL_COLS:
            out[c] = out[c].map({True: "true", False: "false"})
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cols = ",".join(df.columns)
    cur.copy_expert(
        f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buf)


def load_to_postgres(data, args):
    import psycopg2
    conn = psycopg2.connect(host=args.db_host, port=args.db_port, dbname=args.db_name,
                            user=args.db_user, password=args.db_password)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE fraud_labels, transactions, cards, devices, "
                        "merchants, customers, mcc_dictionary CASCADE;")
            for t in LOAD_ORDER:
                _copy(cur, data[t], t)
                print(f"  loaded {t:<15} {len(data[t]):>8,} rows")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transactions", type=int, default=600_000)
    p.add_argument("--customers", type=int, default=20_000)
    p.add_argument("--fraud-rate", type=float, default=0.004,
                   help="prod is ~0.0005; inflated for a workable local stand")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--db-host", default=os.environ.get("DB_HOST", "127.0.0.1"))
    p.add_argument("--db-port", default=os.environ.get("DB_PORT", "55432"))
    p.add_argument("--db-name", default=os.environ.get("DB_NAME", "cardops"))
    p.add_argument("--db-user", default=os.environ.get("DB_USER", "cardops"))
    p.add_argument("--db-password", default=os.environ.get("DB_PASSWORD", "cardops_pwd"))
    p.add_argument("--dry-run", action="store_true", help="build + print stats, no DB")
    args = p.parse_args()

    print(f"Building ~{args.transactions:,} tx / {args.customers:,} customers ...")
    data, fraud_mask = build_data(args.transactions, args.customers,
                                  args.fraud_rate, args.seed)

    tx = data["transactions"]
    n = len(tx)
    print(f"  transactions : {n:,}")
    print(f"  fraud (true) : {fraud_mask.sum():,}  ({fraud_mask.mean()*100:.3f}%)")
    print(f"  labels       : {len(data['fraud_labels']):,} "
          f"(delayed; ~5% fraud left unlabeled)")
    # quick signal sanity: fraud vs legit
    f, l = tx[fraud_mask], tx[~fraud_mask]
    print(f"  mean amount  : fraud {f['amount'].mean():>10,.0f} | legit {l['amount'].mean():>10,.0f}")
    print(f"  %% foreign    : fraud {(f['country']!='RU').mean()*100:>6.1f} | legit {(l['country']!='RU').mean()*100:>6.1f}")
    print(f"  %% night(0-5) : fraud {(pd.to_datetime(f['authorized_at']).dt.hour<6).mean()*100:>6.1f} | legit {(pd.to_datetime(l['authorized_at']).dt.hour<6).mean()*100:>6.1f}")

    if args.dry_run:
        print("dry-run: skipping DB load.")
        return
    print("Loading into Postgres ...")
    load_to_postgres(data, args)
    print("done.")


if __name__ == "__main__":
    main()