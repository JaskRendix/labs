from pathlib import Path

import matplotlib as mpl
import pandas as pd
from anomaly_detection import (
    detect_suspicious_trips,
    run_isolation_forest,
    train_simple_model,
)
from visualization import (
    plot_average_duration_by_weekday,
    plot_hourly_trip_counts,
    plot_label_distribution,
    plot_suspicious_breakdown,
    plot_suspicious_fare_distribution,
    plot_suspicious_speed_distribution,
    plot_suspicious_time_heatmap,
    plot_top_suspicious_zones,
    plot_total_anomaly_trend,
    plot_trip_distance_histogram,
)

mpl.use("Agg")

DATA_FILES = {
    2019: "yellow_tripdata_2019-01.parquet",
    2025: "yellow_tripdata_2025-01.parquet",
}


def load_data(base_path: Path, year: int) -> pd.DataFrame:
    data_path = base_path / DATA_FILES[year]
    df = pd.read_parquet(data_path)
    df["year"] = year
    return df


def profile_data(df: pd.DataFrame, title: str) -> None:
    print(f"\n--- {title} Data Profile for {df['year'].iloc[0]} ---")
    print("Shape:", df.shape)
    print("\nColumn Information:")
    print(df.info())
    print("-" * 30)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration"] > 0) & (df["fare_amount"] > 0)].copy()
    df["hour"] = df["tpep_pickup_datetime"].dt.hour

    df["avg_speed_mph"] = (df["trip_distance"] / df["trip_duration"]) * 60

    df = df[(df["avg_speed_mph"] >= 0) & (df["avg_speed_mph"] < 100)]
    return df


def add_weighted_suspicion_score(df: pd.DataFrame) -> pd.DataFrame:
    weights = [2, 3, 2, 1, 2, 3]
    conditions = [
        (df["trip_duration"] < 5) & (df["fare_amount"] > 50),
        (df["trip_duration"] > 180),
        (df["trip_distance"] < 1) & (df["fare_amount"] > 100),
        (df["fare_amount"] <= 0),
        (df["avg_speed_mph"] < 1),
        (df["avg_speed_mph"] > 60),
    ]
    score = sum(w * cond.astype(int) for w, cond in zip(weights, conditions))
    df["severity_score"] = score
    return df


def print_anomaly_rate_benchmark(anomaly_rate_2019, anomaly_rate_2025):
    print(f"\nAnomaly Rate 2019: {anomaly_rate_2019:.2f}%")
    print(f"Anomaly Rate 2025: {anomaly_rate_2025:.2f}%")
    print(
        "Note: Typical anomaly rates in urban taxi datasets range between 1-3%, depending on data quality."
    )


def print_top_suspicious_zones(
    df: pd.DataFrame, zone_lookup: pd.DataFrame, year: int
) -> None:
    zone_counts = df["PULocationID"].value_counts().head(10)
    zone_names = zone_lookup.set_index("LocationID").loc[zone_counts.index]["Zone"]
    print(f"\n--- Top Pickup Zones for Suspicious Trips {year} ---")
    for zone_id, count in zone_counts.items():
        zone_name = zone_names.get(zone_id, "Unknown")
        print(f"{zone_name} (ID: {zone_id}): {count} suspicious trips")


def main() -> None:
    base_path = Path(__file__).parent
    output_path = base_path / "output_anomaly_comparison"
    output_path.mkdir(exist_ok=True)

    try:
        df_2019 = load_data(base_path, 2019)
        df_2019_cleaned = clean_data(df_2019)

        df_2025 = load_data(base_path, 2025)
        df_2025_cleaned = clean_data(df_2025)

        print("\n--- Generating New General Plots ---")
        plot_trip_distance_histogram(df_2019_cleaned, df_2025_cleaned, output_path)
        plot_hourly_trip_counts(df_2019_cleaned, df_2025_cleaned, output_path)
        plot_average_duration_by_weekday(df_2019_cleaned, df_2025_cleaned, output_path)

        suspicious_2019 = detect_suspicious_trips(df_2019_cleaned)
        suspicious_2025 = detect_suspicious_trips(df_2025_cleaned)

        suspicious_2019 = add_weighted_suspicion_score(suspicious_2019)
        suspicious_2025 = add_weighted_suspicion_score(suspicious_2025)

        df_2019_cleaned = add_weighted_suspicion_score(df_2019_cleaned)
        df_2025_cleaned = add_weighted_suspicion_score(df_2025_cleaned)

        print("\n--- Generating Existing Anomaly Plots ---")
        anomaly_rate_2019, anomaly_rate_2025 = plot_total_anomaly_trend(
            df_2019_cleaned,
            df_2025_cleaned,
            suspicious_2019,
            suspicious_2025,
            output_path,
        )
        print_anomaly_rate_benchmark(anomaly_rate_2019, anomaly_rate_2025)
        plot_suspicious_breakdown(suspicious_2019, suspicious_2025, output_path)
        plot_suspicious_fare_distribution(suspicious_2019, suspicious_2025, output_path)
        plot_suspicious_speed_distribution(
            suspicious_2019, suspicious_2025, output_path
        )
        plot_suspicious_time_heatmap(suspicious_2019, output_path, 2019)
        plot_suspicious_time_heatmap(suspicious_2025, output_path, 2025)

        zone_lookup_path = base_path / "taxi_zone_lookup.csv"
        if zone_lookup_path.exists():
            zone_lookup = pd.read_csv(zone_lookup_path)
            print_top_suspicious_zones(suspicious_2019, zone_lookup, 2019)
            print_top_suspicious_zones(suspicious_2025, zone_lookup, 2025)
            plot_top_suspicious_zones(suspicious_2019, zone_lookup, output_path, 2019)
            plot_top_suspicious_zones(suspicious_2025, zone_lookup, output_path, 2025)
        else:
            print("Warning: 'taxi_zone_lookup.csv' not found. Skipping zone analysis.")

        isolated_2019 = run_isolation_forest(df_2019_cleaned)
        print(f"\nIsolation Forest flagged {isolated_2019.shape[0]} trips in 2019.")

        df_2019_cleaned = add_weighted_suspicion_score(df_2019_cleaned)
        df_2025_cleaned = add_weighted_suspicion_score(df_2025_cleaned)

        train_simple_model(df_2019_cleaned, 2019)
        train_simple_model(df_2025_cleaned, 2025)
        plot_label_distribution(df_2019_cleaned, output_path, 2019)
        plot_label_distribution(df_2025_cleaned, output_path, 2025)

        print(
            "\nAnomaly comparison complete. All plots saved to 'output_anomaly_comparison' directory."
        )

    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure both '{DATA_FILES[2019]}' and '{DATA_FILES[2025]}' are in the directory."
        )


if __name__ == "__main__":
    main()
