from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PAYMENT_TYPE_LABELS = {
    0: "Flex Fare",
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided Trip",
}


def load_data(file_path: Path, year: int) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df["year"] = year
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration"] > 0) & (df["fare_amount"] > 0)].copy()
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    return df


def load_zone_lookup(lookup_path: Path) -> pd.DataFrame:
    lookup = pd.read_csv(lookup_path)
    return lookup.set_index("LocationID")["Zone"]


def compute_payment_proportions(df: pd.DataFrame) -> pd.DataFrame:
    payment_counts = df.groupby(["year", "PULocationID", "payment_type"]).size()
    payment_counts = payment_counts.unstack(fill_value=0)
    proportions = payment_counts.div(payment_counts.sum(axis=1), axis=0)
    return proportions


def plot_change_in_proportions(
    proportions: pd.DataFrame, zone_names: pd.Series, output_path: Path
) -> None:
    output_path.mkdir(exist_ok=True)

    for payment_type in proportions.columns:
        pivot = proportions[payment_type].unstack("year")

        if 2019 in pivot.columns and 2025 in pivot.columns:
            change = (pivot[2025] - pivot[2019]).dropna()

            top_changes = change.abs().sort_values(ascending=False).head(10).index
            change_to_plot = change.loc[top_changes].sort_values()

            if change_to_plot.empty:
                print(
                    f"Skipping payment type {payment_type}: no significant changes to plot."
                )
                continue

            change_to_plot.index = change_to_plot.index.map(
                lambda loc: zone_names.get(loc, f"Zone {loc}")
            )

            label = PAYMENT_TYPE_LABELS.get(payment_type, f"Type {payment_type}")

            plt.figure(figsize=(10, 6))
            change_to_plot.plot(
                kind="barh",
                color=change_to_plot.apply(lambda x: "green" if x > 0 else "red"),
            )
            plt.title(
                f"Top 10 Zones with Largest Change in Proportion for {label} (2019 to 2025)"
            )
            plt.xlabel("Change in Proportion (2025 - 2019)")
            plt.ylabel("Pickup Zone")
            plt.tight_layout()
            plt.savefig(
                output_path
                / f"change_payment_type_{payment_type}_{label.replace(' ', '_')}.png"
            )
            plt.close()


def print_payment_distribution(df: pd.DataFrame) -> None:
    print("\n--- Payment Type Distribution by Year ---")
    payment_counts = df.groupby(["year", "payment_type"]).size().unstack(fill_value=0)
    payment_proportions = payment_counts.div(payment_counts.sum(axis=1), axis=0)

    for year in payment_proportions.index:
        print(f"\nYear: {year}")
        counts = payment_counts.loc[year]
        props = payment_proportions.loc[year]
        for pt, count in counts.items():
            label = PAYMENT_TYPE_LABELS.get(pt, f"Type {pt}")
            proportion = props.get(pt, 0)
            print(f"  {label}: {count:,} trips ({proportion:.2%})")


def print_average_tip_by_payment_type(df: pd.DataFrame) -> None:
    print("\n--- Average Tip Amount by Payment Type and Year ---")
    tip_by_payment = df.groupby(["year", "payment_type"])["tip_amount"].mean()

    for (year, payment_type), avg_tip in tip_by_payment.items():
        label = PAYMENT_TYPE_LABELS.get(payment_type, f"Type {payment_type}")
        print(f"  Year {year}, {label}: ${avg_tip:.2f}")


def print_top_changing_locations_by_payment(
    proportions: pd.DataFrame, zone_names: pd.Series
) -> None:
    print("\n--- Top Locations with Largest Change in Payment Type Proportion ---")
    for payment_type in proportions.columns:
        pivot = proportions[payment_type].unstack("year")
        if 2019 in pivot.columns and 2025 in pivot.columns:
            change = (pivot[2025] - pivot[2019]).dropna()

            # Find locations with the biggest increase and decrease
            largest_increase = change.nlargest(3)
            largest_decrease = change.nsmallest(3)

            label = PAYMENT_TYPE_LABELS.get(payment_type, f"Type {payment_type}")
            print(f"\n{label}:")

            print("  Largest Increase:")
            for loc_id, val in largest_increase.items():
                zone = zone_names.get(loc_id, f"Zone {loc_id}")
                print(f"    {zone}: +{val:.2%}")

            print("  Largest Decrease:")
            for loc_id, val in largest_decrease.items():
                zone = zone_names.get(loc_id, f"Zone {loc_id}")
                print(f"    {zone}: {val:.2%}")


def main() -> None:
    base_path = Path(__file__).parent
    output_path = base_path / "output_change_in_proportions"
    output_path.mkdir(exist_ok=True)

    try:
        zone_names = load_zone_lookup(base_path / "taxi_zone_lookup.csv")

        df_2019 = clean_data(
            load_data(base_path / "yellow_tripdata_2019-01.parquet", 2019)
        )
        df_2025 = clean_data(
            load_data(base_path / "yellow_tripdata_2025-01.parquet", 2025)
        )

        df_combined = pd.concat([df_2019, df_2025], ignore_index=True)
        proportions = compute_payment_proportions(df_combined)

        # Print useful data
        print_payment_distribution(df_combined)
        print_average_tip_by_payment_type(df_combined)
        print_top_changing_locations_by_payment(proportions, zone_names)

        # Generate and save plots
        plot_change_in_proportions(proportions, zone_names, output_path)

        print(
            "\nComparative payment analysis complete. Plots saved to 'output_change_in_proportions' directory."
        )

    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure all required data files are in the directory."
        )


if __name__ == "__main__":
    main()
