from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

mpl.use("Agg")


def plot_total_anomaly_trend(
    df_2019: pd.DataFrame,
    df_2025: pd.DataFrame,
    suspicious_2019: pd.DataFrame,
    suspicious_2025: pd.DataFrame,
    output_path: Path,
) -> tuple[float, float]:
    total_trips_2019 = df_2019.shape[0]
    total_trips_2025 = df_2025.shape[0]

    anomaly_rate_2019 = (suspicious_2019.shape[0] / total_trips_2019) * 100
    anomaly_rate_2025 = (suspicious_2025.shape[0] / total_trips_2025) * 100

    print(f"\nTotal Anomaly Rate Summary:")
    print(
        f"2019 - {suspicious_2019.shape[0]} suspicious trips out of {total_trips_2019} total ({anomaly_rate_2019:.2f}%)"
    )
    print(
        f"2025 - {suspicious_2025.shape[0]} suspicious trips out of {total_trips_2025} total ({anomaly_rate_2025:.2f}%)"
    )

    data = {
        "Year": [2019, 2025],
        "Anomaly Rate (%)": [anomaly_rate_2019, anomaly_rate_2025],
    }
    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Year", y="Anomaly Rate (%)", data=df_plot, color="steelblue")
    plt.title("Overall Percentage of Suspicious Trips (2019 vs 2025)")
    plt.ylabel("Percentage of Trips")
    plt.tight_layout()
    plt.savefig(output_path / "total_anomaly_trend.png")
    plt.close()

    return anomaly_rate_2019, anomaly_rate_2025


def plot_trip_distance_histogram(
    df_2019: pd.DataFrame, df_2025: pd.DataFrame, output_path: Path
) -> None:
    print(f"\nTrip Distance Summary:")
    print(
        f"2019 - Mean: {df_2019['trip_distance'].mean():.2f}, Median: {df_2019['trip_distance'].median():.2f}, Max: {df_2019['trip_distance'].max():.2f}"
    )
    print(
        f"2025 - Mean: {df_2025['trip_distance'].mean():.2f}, Median: {df_2025['trip_distance'].median():.2f}, Max: {df_2025['trip_distance'].max():.2f}"
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(
        df_2019["trip_distance"],
        label="2019",
        color="salmon",
        kde=True,
        stat="density",
        common_norm=False,
    )
    sns.histplot(
        df_2025["trip_distance"],
        label="2025",
        color="steelblue",
        kde=True,
        stat="density",
        common_norm=False,
    )
    plt.title("Trip Distance Distribution (2019 vs 2025)")
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Density")
    plt.xlim(0, 20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "trip_distance_distribution.png")
    plt.close()


def plot_tip_vs_fare(
    df_2019: pd.DataFrame, df_2025: pd.DataFrame, output_path: Path
) -> None:
    print(f"\nTip vs Fare Summary:")
    print(
        f"2019 - Avg Tip: ${df_2019['tip_amount'].mean():.2f}, Avg Fare: ${df_2019['fare_amount'].mean():.2f}"
    )
    print(
        f"2025 - Avg Tip: ${df_2025['tip_amount'].mean():.2f}, Avg Fare: ${df_2025['fare_amount'].mean():.2f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    fig.suptitle("Tip Amount vs. Fare Amount (2019 vs 2025)", fontsize=16, y=1.02)

    sns.scatterplot(
        ax=axes[0],
        data=df_2019[df_2019["tip_amount"] < 50],
        x="fare_amount",
        y="tip_amount",
        alpha=0.5,
        color="salmon",
    )
    axes[0].set_title("2019")
    axes[0].set_xlabel("Fare Amount ($)")
    axes[0].set_ylabel("Tip Amount ($)")

    sns.scatterplot(
        ax=axes[1],
        data=df_2025[df_2025["tip_amount"] < 50],
        x="fare_amount",
        y="tip_amount",
        alpha=0.5,
        color="steelblue",
    )
    axes[1].set_title("2025")
    axes[1].set_xlabel("Fare Amount ($)")
    axes[1].set_ylabel("Tip Amount ($)")

    plt.tight_layout()

    plt.savefig(output_path / "tip_vs_fare_comparison.png")
    plt.close()


def plot_hourly_trip_counts(
    df_2019: pd.DataFrame, df_2025: pd.DataFrame, output_path: Path
) -> None:

    hourly_2019 = df_2019.groupby("hour").size()
    hourly_2025 = df_2025.groupby("hour").size()

    print(f"\nHourly Trip Count Summary:")
    print(f"2019 - Peak Hour: {hourly_2019.idxmax()} ({hourly_2019.max()} trips)")
    print(f"2025 - Peak Hour: {hourly_2025.idxmax()} ({hourly_2025.max()} trips)")

    plt.figure(figsize=(12, 6))
    hourly_2019.plot(
        kind="line",
        label="2019",
        marker="o",
        color="salmon",
        linestyle="--",
    )
    hourly_2025.plot(
        kind="line",
        label="2025",
        marker="o",
        color="steelblue",
        linestyle="-",
    )
    plt.title("Total Trips by Hour of Day (2019 vs 2025)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Trips")
    plt.xticks(range(24))
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path / "hourly_trip_counts.png")
    plt.close()


def plot_average_duration_by_weekday(
    df_2019: pd.DataFrame, df_2025: pd.DataFrame, output_path: Path
) -> None:
    df_2019["weekday"] = df_2019["tpep_pickup_datetime"].dt.day_name()
    df_2025["weekday"] = df_2025["tpep_pickup_datetime"].dt.day_name()

    ordered_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    avg_duration_2019 = (
        df_2019.groupby("weekday")["trip_duration"].mean().reindex(ordered_days)
    )
    avg_duration_2025 = (
        df_2025.groupby("weekday")["trip_duration"].mean().reindex(ordered_days)
    )

    print(f"\nAverage Duration by Weekday:")
    print("2019:")
    print(avg_duration_2019.round(2))
    print("2025:")
    print(avg_duration_2025.round(2))

    combined_df = pd.DataFrame({"2019": avg_duration_2019, "2025": avg_duration_2025})

    combined_df.plot(
        kind="bar", figsize=(12, 7), color=["salmon", "steelblue"], rot=45, width=0.8
    )
    plt.title("Average Trip Duration by Weekday (2019 vs 2025)")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Trip Duration (minutes)")
    plt.legend(title="Year")
    plt.tight_layout()
    plt.savefig(output_path / "avg_trip_duration_by_weekday.png")
    plt.close()


def plot_suspicious_breakdown(
    df_2019: pd.DataFrame, df_2025: pd.DataFrame, output_path: Path
) -> None:
    reasons_2019 = df_2019["reason"].value_counts(normalize=True)
    reasons_2025 = df_2025["reason"].value_counts(normalize=True)

    print(f"\nSuspicious Reason Breakdown:")
    print("2019:")
    print(reasons_2019.round(3))
    print("2025:")
    print(reasons_2025.round(3))

    combined_reasons = pd.DataFrame(
        {"2019": reasons_2019, "2025": reasons_2025}
    ).fillna(0)

    combined_reasons.sort_values(by="2025", ascending=False).plot(
        kind="barh", figsize=(12, 8), color=["salmon", "steelblue"]
    )
    plt.title("Proportion of Suspicious Trip Reasons (2019 vs 2025)")
    plt.xlabel("Proportion of Suspicious Trips")
    plt.ylabel("Anomaly Type")
    plt.tight_layout()
    plt.savefig(output_path / "suspicious_trip_breakdown_comparison.png")
    plt.close()


def plot_suspicious_fare_distribution(
    suspicious_2019: pd.DataFrame, suspicious_2025: pd.DataFrame, output_path: Path
) -> None:
    print(f"\nSuspicious Fare Summary:")
    print(f"2019 - Mean: ${suspicious_2019['fare_amount'].mean():.2f}")
    print(f"2025 - Mean: ${suspicious_2025['fare_amount'].mean():.2f}")

    plt.figure(figsize=(10, 6))
    sns.kdeplot(suspicious_2019["fare_amount"], label="2019", color="salmon", fill=True)
    sns.kdeplot(
        suspicious_2025["fare_amount"], label="2025", color="steelblue", fill=True
    )
    plt.title("Distribution of Fare Amounts for Suspicious Trips")
    plt.xlabel("Fare Amount ($)")
    plt.xlim(0, 200)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "suspicious_fare_distribution.png")
    plt.close()


def plot_suspicious_speed_distribution(
    suspicious_2019: pd.DataFrame, suspicious_2025: pd.DataFrame, output_path: Path
) -> None:
    suspicious_fast_2019 = suspicious_2019[suspicious_2019["avg_speed_mph"] > 60]
    suspicious_fast_2025 = suspicious_2025[suspicious_2025["avg_speed_mph"] > 60]

    print(f"\n'Impossibly Fast' Speed Summary:")
    print(f"2019 - Max Speed: {suspicious_fast_2019['avg_speed_mph'].max():.2f}")
    print(f"2025 - Max Speed: {suspicious_fast_2025['avg_speed_mph'].max():.2f}")

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        suspicious_fast_2019["avg_speed_mph"], label="2019", color="salmon", fill=True
    )
    sns.kdeplot(
        suspicious_fast_2025["avg_speed_mph"],
        label="2025",
        color="steelblue",
        fill=True,
    )
    plt.title("Distribution of Speeds for 'Impossibly Fast' Trips")
    plt.xlabel("Average Speed (mph)")
    plt.xlim(60, 200)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "suspicious_speed_distribution.png")
    plt.close()


def plot_suspicious_time_heatmap(
    df: pd.DataFrame, output_path: Path, year: int
) -> None:
    df["weekday"] = df["tpep_pickup_datetime"].dt.day_name()
    pivot_table = df.pivot_table(
        index="weekday", columns="hour", values="fare_amount", aggfunc="count"
    ).fillna(0)

    print(f"\n Suspicious Trip Time Heatmap Summary ({year}):")
    print(pivot_table.sum(axis=1).round(0))

    ordered_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    pivot_table = pivot_table.reindex(ordered_days)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, cmap="YlOrRd", linewidths=0.5)
    plt.title(f"Suspicious Trip Frequency by Hour and Weekday {year}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(output_path / f"suspicious_time_heatmap_{year}.png")
    plt.close()


def plot_top_suspicious_zones(
    df: pd.DataFrame, zone_lookup: pd.DataFrame, output_path: Path, year: int
) -> None:
    zone_counts = df["PULocationID"].value_counts().head(20)
    zone_names = zone_lookup.set_index("LocationID").loc[zone_counts.index]["Zone"]

    print(f"\nTop Suspicious Pickup Zones ({year}):")
    for zone, count in zip(zone_names.values, zone_counts.values):
        print(f"{zone}: {count} trips")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=zone_counts.values, y=zone_names.values, color="steelblue")
    plt.title(f"Top 20 Pickup Zones for Suspicious Trips {year}")
    plt.xlabel("Number of Suspicious Trips")
    plt.ylabel("Zone")
    plt.tight_layout()
    plt.savefig(output_path / f"top_suspicious_zones_{year}.png")
    plt.close()


def plot_label_distribution(df: pd.DataFrame, output_path: Path, year: int) -> None:
    df["label"] = df["severity_score"] > 0
    counts = df["label"].value_counts()

    print(f"\nLabel Distribution ({year}):")
    print(counts.rename({True: "Suspicious", False: "Normal"}))

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=counts.index.map({True: "Suspicious", False: "Normal"}),
        y=counts.values,
        color="steelblue",
    )
    plt.title(f"Label Distribution ({year})")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(output_path / f"label_distribution_bar_chart_{year}.png")
    plt.close()
