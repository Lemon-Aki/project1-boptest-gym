import pandas as pd
import numpy as np



# ============================================================
# Settings
# ============================================================
CSV_PATH = "/mnt/d/dev-code/boptest-0.8.0/testcases/bestest_air/models/simulation.csv"

# The current Seoul PI simulation data uses elapsed seconds
# from 2021-01-01.
BASE_DATE = pd.Timestamp("2021-01-01")

TIME_COL = "interval_start_time_s"

# Preferred interval-level columns
COST_INTERVAL_COL = "cost_tot_interval_usd_per_m2"
TDIS_INTERVAL_COL = "tdis_interval_Kh"

# Fallback cumulative columns
COST_CUM_COL = "cost_tot_cumulative_usd_per_m2"
TDIS_CUM_COL = "tdis_tot_cumulative_Kh"

# Half-open intervals [start, end):
# Summer: 2021-05-22 00:00 <= t < 2021-09-30 00:00
#         -> May 22 through Sep 29
# Winter: Jan 1 through Apr 14, plus Dec 4 through Dec 31
PERIODS = {
    "All-year": [
        ("2021-01-01", "2022-01-01"),
    ],
    "Summer (05/22-09/29)": [
        ("2021-05-22", "2021-09-30"),
    ],
    "Winter (12/04-04/14)": [
        ("2021-01-01", "2021-04-15"),
        ("2021-12-04", "2022-01-01"),
    ],
}

REFERENCE_WEIGHT = 26.0


# ============================================================
# Helpers
# ============================================================
def get_interval_series(df, interval_col, cumulative_col):
    """
    Prefer an interval-level quantity.
    If only a cumulative quantity exists, reconstruct interval values
    by taking the first difference.
    """
    if interval_col in df.columns:
        return pd.to_numeric(df[interval_col], errors="coerce").fillna(0.0)

    if cumulative_col in df.columns:
        cum = pd.to_numeric(df[cumulative_col], errors="coerce")
        interval = cum.diff()
        interval.iloc[0] = cum.iloc[0]
        return interval.fillna(0.0)

    raise KeyError(
        f"Neither '{interval_col}' nor '{cumulative_col}' exists.\n"
        "If you are using the raw BOPTEST simulation.csv, first calculate "
        "the BOPTEST cost and tdis interval values, or change the column "
        "names in this script to match your processed file."
    )


def make_period_mask(timestamp_series, ranges):
    mask = pd.Series(False, index=timestamp_series.index)

    for start, end in ranges:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        mask |= (timestamp_series >= start) & (timestamp_series < end)

    return mask


def summarize_period(df, name, ranges):
    mask = make_period_mask(df["timestamp"], ranges)
    part = df.loc[mask]

    cost = part["_cost_interval"].sum()
    tdis = part["_tdis_interval"].sum()

    ratio = np.nan if cost == 0 else tdis / cost
    weighted_cost = REFERENCE_WEIGHT * cost
    weighted_cost_to_tdis = np.nan if tdis == 0 else weighted_cost / tdis

    # 15-minute data -> 96 intervals/day.
    # This is only for an intuitive daily average.
    days_equivalent = len(part) / 96.0

    return {
        "period": name,
        "interval_count": len(part),
        "days_equivalent": days_equivalent,
        "cost_total_USD_per_m2": cost,
        "tdis_total_Kh": tdis,
        "cost_to_tdis_ratio": f"1:{ratio:.3f}" if np.isfinite(ratio) else "undefined",
        "balance_weight_tdis_div_cost": ratio,
        f"{REFERENCE_WEIGHT:g}x_cost": weighted_cost,
        f"{REFERENCE_WEIGHT:g}x_cost_div_tdis": weighted_cost_to_tdis,
        "cost_per_day": cost / days_equivalent if days_equivalent else np.nan,
        "tdis_per_day": tdis / days_equivalent if days_equivalent else np.nan,
    }


# ============================================================
# Main
# ============================================================
def main():
    df = pd.read_csv(CSV_PATH)

    if TIME_COL not in df.columns:
        raise KeyError(
            f"'{TIME_COL}' column not found.\n"
            "Change TIME_COL to the elapsed-time column in your file."
        )

    # Convert BOPTEST elapsed seconds to calendar date/time.
    df["timestamp"] = (
        BASE_DATE
        + pd.to_timedelta(pd.to_numeric(df[TIME_COL], errors="coerce"), unit="s")
    )

    df["_cost_interval"] = get_interval_series(
        df, COST_INTERVAL_COL, COST_CUM_COL
    )
    df["_tdis_interval"] = get_interval_series(
        df, TDIS_INTERVAL_COL, TDIS_CUM_COL
    )

    rows = []
    for name, ranges in PERIODS.items():
        rows.append(summarize_period(df, name, ranges))

    result = pd.DataFrame(rows)

    # Pretty console output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\n=== Seasonal cost / tdis analysis ===\n")
    print(
        result[
            [
                "period",
                "cost_total_USD_per_m2",
                "tdis_total_Kh",
                "cost_to_tdis_ratio",
                "balance_weight_tdis_div_cost",
                f"{REFERENCE_WEIGHT:g}x_cost",
                f"{REFERENCE_WEIGHT:g}x_cost_div_tdis",
            ]
        ].to_string(index=False)
    )

    print("\nInterpretation:")
    print(
        "- balance_weight_tdis_div_cost is the weight w that makes "
        "w * cost approximately equal to tdis for that period."
    )
    print(
        f"- '{REFERENCE_WEIGHT:g}x_cost_div_tdis' near 1.0 means "
        f"the {REFERENCE_WEIGHT:g}:1 weighting balances the two terms."
    )
    print(
        "- Above 1.0: weighted cost term is larger than tdis. "
        "Below 1.0: tdis term is larger."
    )

    output_path = "seasonal_cost_tdis_summary.csv"
    result.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()