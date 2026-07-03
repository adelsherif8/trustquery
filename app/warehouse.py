"""Hotel analytics warehouse on DuckDB.

A small but realistic star schema. Seeded deterministically (fixed seed) so the
golden-eval numbers are stable and reconciliation is meaningful.

Grain notes (this is the whole game for trustworthy numbers):
  - `bookings` is at booking grain. Revenue / room-nights / ADR / booking counts
    all live here and are additive over booking rows.
  - `inventory` is at property-month grain and holds available room-nights.
    Occupancy is a NON-additive ratio (sold / available) that must be computed
    by joining sold room-nights to inventory on (property_id, month) BEFORE
    dividing -- never by fanning bookings out against inventory.
"""

import os
import duckdb

# Deterministic pseudo-data (no external randomness at query time).
_PROPERTIES = [
    # property_id, name, city, region, rooms
    (1, "Nile Grand Cairo",      "Cairo",     "Egypt",  120),
    (2, "Alexandria Marina",     "Alexandria","Egypt",  80),
    (3, "Dubai Marina Suites",   "Dubai",     "UAE",    200),
    (4, "Riyadh Business Tower", "Riyadh",    "KSA",    150),
    (5, "Red Sea Resort",        "Hurghada",  "Egypt",  260),
]

_ROOM_TYPES = [
    # room_type, base_rate
    ("Standard", 90),
    ("Deluxe",   150),
    ("Suite",    280),
]

_MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]


def _month_days(month: str) -> int:
    y, m = month.split("-")
    y, m = int(y), int(m)
    if m in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if m == 2:
        return 28
    return 30


def build(db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")

    # --- dimension: properties ---
    con.execute("""
        CREATE OR REPLACE TABLE analytics.properties (
            property_id   INTEGER,
            property_name VARCHAR,
            city          VARCHAR,
            region        VARCHAR,
            rooms         INTEGER
        );
    """)
    import pandas as pd  # local build only; Vercel loads a prebuilt DB file
    props_df = pd.DataFrame(
        _PROPERTIES,
        columns=["property_id", "property_name", "city", "region", "rooms"],
    )
    con.execute("INSERT INTO analytics.properties SELECT * FROM props_df")

    # --- dimension: room_types ---
    con.execute("""
        CREATE OR REPLACE TABLE analytics.room_types (
            room_type VARCHAR,
            base_rate INTEGER
        );
    """)
    con.executemany("INSERT INTO analytics.room_types VALUES (?,?)", _ROOM_TYPES)

    # --- fact: bookings (booking grain) ---
    con.execute("""
        CREATE OR REPLACE TABLE analytics.bookings (
            booking_id    INTEGER,
            property_id   INTEGER,
            room_type     VARCHAR,
            check_in_month VARCHAR,
            nights        INTEGER,
            room_revenue  DOUBLE,
            status        VARCHAR
        );
    """)

    import math

    bookings = []
    bid = 0
    _type_cycle = ["Standard", "Standard", "Standard", "Deluxe", "Deluxe", "Suite"]
    _rates = {rt: br for rt, br in _ROOM_TYPES}
    # per-property occupancy bias so cities/regions differ (demo tells a story)
    _prop_bias = {1: 0.02, 2: -0.10, 3: 0.12, 4: -0.05, 5: 0.07}
    # deterministic generation: fill each property-month toward a realistic
    # seasonal target occupancy (~45%-90%), then emit bookings until we reach it.
    for (pid, name, city, region, rooms) in _PROPERTIES:
        for mi, month in enumerate(_MONTHS):
            days = _month_days(month)
            avail = rooms * days
            # seasonal demand curve, phase-shifted per property
            season = 0.5 + 0.5 * math.sin((mi / 12.0) * 2 * math.pi + pid * 0.9)
            target_occ = min(0.95, max(0.38, 0.52 + 0.30 * season + _prop_bias[pid]))
            target_nights = int(target_occ * avail)

            acc = 0
            k = 0
            while acc < target_nights:
                bid += 1
                k += 1
                nights = 1 + ((bid * 7 + pid * 3 + k) % 5)      # 1..5
                rt = _type_cycle[(bid + pid) % len(_type_cycle)]
                base_rate = _rates[rt]
                rate = base_rate * (1.0 + 0.10 * (((bid + mi) % 5) - 2) / 2.0)
                revenue = round(rate * nights, 2)
                status = "cancelled" if (bid % 13) == 0 else "confirmed"
                bookings.append((bid, pid, rt, month, nights, revenue, status))
                if status == "confirmed":
                    acc += nights

    bookings_df = pd.DataFrame(
        bookings,
        columns=["booking_id", "property_id", "room_type", "check_in_month",
                 "nights", "room_revenue", "status"],
    )
    con.execute("INSERT INTO analytics.bookings SELECT * FROM bookings_df")

    # --- fact: inventory (property-month grain) : available room-nights ---
    con.execute("""
        CREATE OR REPLACE TABLE analytics.inventory (
            property_id INTEGER,
            month       VARCHAR,
            available_room_nights INTEGER
        );
    """)
    inv_rows = []
    for (pid, name, city, region, rooms) in _PROPERTIES:
        for month in _MONTHS:
            inv_rows.append((int(pid), month, rooms * _month_days(month)))
    inv_df = pd.DataFrame(
        inv_rows, columns=["property_id", "month", "available_room_nights"])
    con.execute("INSERT INTO analytics.inventory SELECT * FROM inv_df")

    return con


def materialize(path: str) -> None:
    """Build the warehouse once and persist it to a DuckDB file for fast,
    read-only loading in serverless (no 30s cold-start rebuild)."""
    import os
    if os.path.exists(path):
        os.remove(path)
    con = build(path)
    con.close()


if __name__ == "__main__":
    con = build()
    n = con.execute("SELECT COUNT(*) FROM analytics.bookings").fetchone()[0]
    rev = con.execute(
        "SELECT ROUND(SUM(room_revenue),2) FROM analytics.bookings WHERE status='confirmed'"
    ).fetchone()[0]
    print(f"bookings rows: {n}")
    print(f"total confirmed revenue: {rev}")
    print("sample:")
    for row in con.execute(
        "SELECT property_name, city, region, rooms FROM analytics.properties"
    ).fetchall():
        print("  ", row)
