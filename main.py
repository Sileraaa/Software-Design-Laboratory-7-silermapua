"""
Strava Running Data — Interactive Menu Analysis
=================================================
Reads "Strava Running Data.xlsx" (must be in the same folder as this script)
and lets you choose which chart to view, one at a time, via a text menu.
No files are saved — each chart opens in its own matplotlib window.

Tools used
----------
- pandas    : loading, cleaning, resampling (weekly aggregation), datetime
              feature engineering
- matplotlib: all plotting (bar, line, scatter, heatmap, histogram)
- sklearn   : StandardScaler + LinearRegression to quantify the
              pace-vs-distance relationship (Chart 3)

Requirements: pandas, matplotlib, scikit-learn, openpyxl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

FILE_PATH = "Strava Running Data.xlsx"   # must sit next to this script
plt.rcParams["figure.dpi"] = 100


# ----------------------------------------------------------------------------
# DATA LOADING & PREP
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_excel(FILE_PATH)

    df["start_date_local"] = pd.to_datetime(df["start_date_local"])
    df["day_of_week"] = df["start_date_local"].dt.day_name()
    df["hour"] = df["start_date_local"].dt.hour
    df["week"] = df["start_date_local"].dt.tz_localize(None).dt.to_period("W")

    runs = df[df["sport_type"] == "Run"].copy()
    runs["distance_km"] = runs["distance"] / 1000
    runs["avg_pace_min_per_km"] = (1000 / runs["average_speed"]) / 60
    runs_valid_pace = runs[runs["average_speed"] > 0].copy()

    return df, runs, runs_valid_pace


# ----------------------------------------------------------------------------
# OPTION 1 — METADATA
# ----------------------------------------------------------------------------
def show_metadata(df, runs):
    print("\n" + "-" * 60)
    print("DATASET METADATA")
    print("-" * 60)
    print(f"Shape (rows, columns): {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTotal activities: {len(df)} | Runs only: {len(runs)}")
    print(f"Date range: {df['start_date_local'].min().date()} to "
          f"{df['start_date_local'].max().date()}")
    print("\nActivity type breakdown:")
    print(df["sport_type"].value_counts().to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("-" * 60)


# ----------------------------------------------------------------------------
# CHART 2 — Weekly training volume
# ----------------------------------------------------------------------------
def chart_weekly_volume(runs):
    weekly = runs.groupby("week")["distance_km"].sum()
    weekly.index = weekly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(weekly.index, weekly.values, width=5, color="#fc4c02", edgecolor="white")
    ax.set_title("Weekly Running Volume Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Distance (km)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# CHART 3 — Pace trend with rolling average
# ----------------------------------------------------------------------------
def chart_pace_trend(runs_valid_pace):
    rs = runs_valid_pace.sort_values("start_date_local").copy()
    rs["pace_roll5"] = rs["avg_pace_min_per_km"].rolling(5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(rs["start_date_local"], rs["avg_pace_min_per_km"],
               alpha=0.35, color="#fc4c02", label="Individual run pace")
    ax.plot(rs["start_date_local"], rs["pace_roll5"],
            color="#1f2937", linewidth=2.5, label="5-run rolling average")
    ax.set_title("Running Pace Trend Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Pace (min/km)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# CHART 4 — Distance vs Pace regression (sklearn)
# ----------------------------------------------------------------------------
def chart_distance_vs_pace(runs_valid_pace):
    X = runs_valid_pace[["distance_km"]].values
    y = runs_valid_pace["avg_pace_min_per_km"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reg = LinearRegression().fit(X_scaled, y)
    y_pred = reg.predict(X_scaled)
    slope_per_km = reg.coef_[0] / scaler.scale_[0]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(runs_valid_pace["distance_km"], y, alpha=0.6, color="#fc4c02",
               edgecolor="white", s=60, label="Runs")
    order = np.argsort(runs_valid_pace["distance_km"].values)
    ax.plot(runs_valid_pace["distance_km"].values[order], y_pred[order],
            color="#1f2937", linewidth=2.5,
            label=f"Linear fit ({slope_per_km:+.2f} min/km per extra km)")
    ax.set_title("Distance vs. Pace (with Linear Regression)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pace (min/km)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# CHART 5 — Day of week vs hour of day heatmap
# ----------------------------------------------------------------------------
def chart_day_hour_heatmap(runs):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = runs.pivot_table(index="day_of_week", columns="hour",
                              values="distance_km", aggfunc="count", fill_value=0)
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="Oranges", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Hour of Day (24h)")
    ax.set_ylabel("Day of Week")
    ax.set_title("When Do Runs Happen? (Day of Week vs. Hour)", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Runs")
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# CHART 6 — Elevation gain vs distance, colored by pace
# ----------------------------------------------------------------------------
def chart_elevation_vs_distance(runs_valid_pace):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(runs_valid_pace["distance_km"], runs_valid_pace["total_elevation_gain"],
                     c=runs_valid_pace["avg_pace_min_per_km"], cmap="RdYlGn_r",
                     s=70, edgecolor="white", alpha=0.85)
    ax.set_title("Elevation Gain vs. Distance (colored by Pace)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Total Elevation Gain (m)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Pace (min/km) — greener = faster")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# CHART 7 — Distance distribution histogram
# ----------------------------------------------------------------------------
def chart_distance_distribution(runs):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(runs["distance_km"], bins=15, color="#fc4c02", edgecolor="white")
    ax.axvline(runs["distance_km"].median(), color="#1f2937", linestyle="--",
               linewidth=2, label=f"Median = {runs['distance_km'].median():.1f} km")
    ax.set_title("Distribution of Run Distances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Number of Runs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# HOME UI SCREEN
# ----------------------------------------------------------------------------
def print_menu():
    width = 58
    top = "+" + "-" * (width - 2) + "+"

    def line(text=""):
        print("|" + text.center(width - 2) + "|")

    def left(text):
        print("| " + text.ljust(width - 4) + " |")

    print("\n" + top)
    line("STRAVA RUNNING DATA — ANALYSIS MENU")
    print(top)
    left("")
    left("[1] View dataset metadata")
    left("[2] Weekly training volume")
    left("[3] Pace trend over time")
    left("[4] Distance vs. Pace (regression)")
    left("[5] Day/Hour running heatmap")
    left("[6] Elevation vs. Distance vs. Pace")
    left("[7] Distance distribution")
    left("[8] Quit")
    left("")
    print(top)


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    try:
        df, runs, runs_valid_pace = load_data()
    except FileNotFoundError:
        print(f"\nError: could not find '{FILE_PATH}'.")
        print("Make sure the xlsx file is in the same folder as this script.")
        return

    actions = {
        "1": lambda: show_metadata(df, runs),
        "2": lambda: chart_weekly_volume(runs),
        "3": lambda: chart_pace_trend(runs_valid_pace),
        "4": lambda: chart_distance_vs_pace(runs_valid_pace),
        "5": lambda: chart_day_hour_heatmap(runs),
        "6": lambda: chart_elevation_vs_distance(runs_valid_pace),
        "7": lambda: chart_distance_distribution(runs),
    }

    while True:
        print_menu()
        choice = input("Enter your choice (1-8): ").strip()

        if choice == "8":
            print("\nGoodbye!")
            break
        elif choice in actions:
            actions[choice]()
        else:
            print("\nInvalid choice. Please enter a number from 1 to 8.")


if __name__ == "__main__":
    main()