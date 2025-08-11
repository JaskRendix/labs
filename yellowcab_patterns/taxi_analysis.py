from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(base_path: Path) -> pd.DataFrame:
    data_path = base_path / "yellow_tripdata_2019-01.parquet"
    df = pd.read_parquet(data_path)
    return df


def clean_data(df) -> Any:
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration"] > 0) & (df["fare_amount"] > 0)].copy()
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    return df


def plot_trip_counts_by_hour(df, output_path) -> None:
    hourly_counts = df["hour"].value_counts().sort_index()

    print("\nHourly Trip Counts (Top 5):")
    print(hourly_counts.head())

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=hourly_counts.index,
        y=hourly_counts.values,
        palette="viridis",
        hue=None,
        legend=False,
    )
    plt.title("Trip Counts by Hour of Day (Jan 2019)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(output_path / "trip_counts_by_hour.png")
    plt.close()


def plot_fare_distribution(df, output_path) -> None:
    print("\nFare Amount Statistics:")
    print(df["fare_amount"].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df["fare_amount"], bins=40, color="steelblue")
    plt.xlim(0, 50)
    plt.title("Fare Amount Distribution")
    plt.xlabel("Fare ($)")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(output_path / "fare_distribution.png")
    plt.close()


def detect_suspicious_trips(df: pd.DataFrame, output_path: Path) -> None:
    suspicious = df[
        ((df["trip_duration"] < 5) & (df["fare_amount"] > 50))
        | (df["trip_duration"] > 180)
        | ((df["trip_distance"] < 1) & (df["fare_amount"] > 100))
        | (df["fare_amount"] <= 0)
        | ((df["fare_amount"] / df["trip_distance"]) < 0.5)
    ].copy()

    conditions = [
        # Impossible trip: High distance but ultra short duration
        (suspicious["trip_distance"] > 20) & (suspicious["trip_duration"] < 10),
        # Extremely short & expensive
        (suspicious["trip_duration"] < 5) & (suspicious["fare_amount"] > 50),
        # Unrealistically long duration
        (suspicious["trip_duration"] > 180),
        # Short trip, but extreme fare
        (suspicious["trip_distance"] < 1) & (suspicious["fare_amount"] > 100),
        # Invalid fare
        (suspicious["fare_amount"] <= 0),
        # Long and extremely cheap (low fare-per-mile)
        (suspicious["fare_amount"] / suspicious["trip_distance"]) < 0.5,
    ]

    reasons = [
        "Impossible Distance-Duration Combo",
        "Extremely Short & Expensive",
        "Unrealistically Long (3+ hours)",
        "Short Trip, Extreme Fare",
        "Invalid Fare",
        "Long & Extremely Cheap",
    ]

    suspicious["reason"] = np.select(conditions, reasons, default="Other")

    print(f"Suspicious trips found: {suspicious.shape[0]}")
    print(
        suspicious[
            [
                "tpep_pickup_datetime",
                "trip_duration",
                "trip_distance",
                "fare_amount",
                "reason",
            ]
        ].head()
    )

    output_file = output_path / "suspicious_trips.csv"
    suspicious.to_csv(output_file, index=False)
    print(f"Saved suspicious trips to: {output_file}")

    reason_counts = suspicious["reason"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(y=reason_counts.index, x=reason_counts.values, palette="rocket")
    plt.title("Suspicious Trip Breakdown")
    plt.xlabel("Number of Trips")
    plt.ylabel("Anomaly Type")
    plt.tight_layout()
    plt.savefig("output/suspicious_trip_breakdown.png")
    plt.close()


def simulate_multiple_shifts(df, shift_starts: list[pd.Timestamp]) -> pd.DataFrame:
    df_sorted = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    results = []

    for shift_start in shift_starts:
        shift_end = shift_start + pd.Timedelta(hours=12)
        current_time = shift_start
        driver_trips = []

        start_idx = df_sorted["tpep_pickup_datetime"].searchsorted(
            shift_start, side="left"
        )

        for i in range(start_idx, len(df_sorted)):
            row = df_sorted.iloc[i]
            pickup = row["tpep_pickup_datetime"]
            duration = row["trip_duration"]

            if pickup < current_time:
                continue

            dropoff = pickup + pd.Timedelta(minutes=duration)
            if dropoff <= shift_end:
                driver_trips.append(row)
                current_time = dropoff + pd.Timedelta(minutes=np.random.randint(5, 11))
            else:
                break

        driver_df = pd.DataFrame(driver_trips)
        total_trips = driver_df.shape[0]
        total_earnings = driver_df["fare_amount"].sum()
        total_minutes = driver_df["trip_duration"].sum()
        idle_time = (shift_end - shift_start).total_seconds() / 60 - total_minutes

        results.append(
            {
                "Shift Start": shift_start,
                "Trips": total_trips,
                "Earnings ($)": round(total_earnings, 2),
                "Driving Time (min)": round(total_minutes, 1),
                "Idle Time (min)": round(idle_time, 1),
            }
        )
    return pd.DataFrame(results)


def export_cleaned_data(df, output_path) -> None:
    df_cleaned = df[
        [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "trip_duration",
            "fare_amount",
            "PULocationID",
            "DOLocationID",
        ]
    ]
    df_cleaned.to_csv(output_path / "yellow_taxi_jan2019_cleaned.csv", index=False)


def profile_data(df: pd.DataFrame, title: str) -> None:
    print(f"--- {title} Data Profile ---")
    print("Shape:", df.shape)
    print("\nColumn Information:")
    print(df.info())
    print("\nDescriptive Statistics (Fare Amount):")
    print(df["fare_amount"].describe())
    print("\nNull Values:")
    print(df.isnull().sum())
    print("-" * 30)


def main() -> None:
    parser = ArgumentParser(description="NYC Taxi Data Analysis")
    parser.add_argument(
        "--simulate", action="store_true", help="Run driver shift simulation"
    )
    parser.add_argument(
        "--detect-anomalies", action="store_true", help="Detect suspicious trips"
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    base_path = Path(__file__).parent
    output_path = base_path / "output"
    output_path.mkdir(exist_ok=True)

    df = load_data(base_path)
    profile_data(df, "Raw")

    df = clean_data(df)
    profile_data(df, "Cleaned")

    if args.plot:
        plot_trip_counts_by_hour(df, output_path)
        plot_fare_distribution(df, output_path)

    if args.detect_anomalies:
        detect_suspicious_trips(df, output_path)

    if args.simulate:
        shift_times = [
            pd.Timestamp("2019-01-15 08:00:00"),
            pd.Timestamp("2019-01-16 08:00:00"),
            pd.Timestamp("2019-01-17 08:00:00"),
        ]
        summary_df = simulate_multiple_shifts(df, shift_times)
        print(summary_df)
        summary_df.to_csv(output_path / "driver_shift_summary.csv", index=False)

    export_cleaned_data(df, output_path)


if __name__ == "__main__":
    main()
