import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
import pandas as pd
import yaml


class RealEstateSimulator:
    def __init__(
        self,
        initial_investment=10_000_000,
        holding_period=5,
        discount_rate=0.10,
        annual_debt_service=800_000,
    ):
        self.initial_investment = initial_investment
        self.holding_period = holding_period
        self.discount_rate = discount_rate
        self.annual_debt_service = annual_debt_service
        self.output_dir = Path(__file__).parent / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.scenarios = {}
        self.df = pd.DataFrame()

    def load_scenarios(self, path: str):
        ext = Path(path).suffix
        if ext == ".json":
            with open(path, "r") as f:
                self.scenarios = json.load(f)
        elif ext in [".yaml", ".yml"]:
            with open(path, "r") as f:
                self.scenarios = yaml.safe_load(f)
        elif ext == ".csv":
            df = pd.read_csv(path)
            self.scenarios = {
                row["scenario"]: {
                    "cf_mean": row["cf_mean"],
                    "cf_std": row["cf_std"],
                    "exit_mean": row["exit_mean"],
                    "exit_std": row["exit_std"],
                }
                for _, row in df.iterrows()
            }
        else:
            raise ValueError("Unsupported file format. Use .json, .yaml, or .csv")

    def set_scenario(self, scenario):
        if scenario not in self.scenarios:
            raise ValueError(
                f"Scenario '{scenario}' not found. Available: {list(self.scenarios.keys())}"
            )
        self.scenario = self.scenarios[scenario]

    def run_simulation(self, n_simulations=10000):
        results = []
        for _ in range(n_simulations):
            cash_flows = np.random.normal(
                self.scenario["cf_mean"], self.scenario["cf_std"], self.holding_period
            )
            exit_value = np.random.normal(
                self.scenario["exit_mean"], self.scenario["exit_std"]
            )
            cash_flows[-1] += exit_value

            irr = npf.irr([-self.initial_investment] + list(cash_flows)) * 100
            npv = npf.npv(
                self.discount_rate, [-self.initial_investment] + list(cash_flows)
            )
            equity_multiple = sum(cash_flows) / self.initial_investment
            dscr = np.mean(cash_flows) / self.annual_debt_service

            results.append(
                {
                    "IRR (%)": irr,
                    "NPV ($)": npv,
                    "Equity Multiple": equity_multiple,
                    "DSCR": dscr,
                }
            )

        self.df = pd.DataFrame(results)
        return self.df

    def load_investment_data(self, filepath: str):
        self.df = pd.read_csv(filepath)
        print(f"Loaded {len(self.df)} investment records from {filepath}")

    def plot_results(self):
        if self.df.empty:
            print("No data to plot. Run simulation or load data first.")
            return
        plt.figure(figsize=(10, 6))
        plt.hist(self.df["IRR (%)"], bins=50, color="skyblue", edgecolor="black")
        plt.title(f"Monte Carlo Simulation of IRR ({len(self.df)} runs)")
        plt.xlabel("IRR (%)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plot_path = self.output_dir / "irr_distribution_histogram.png"
        plt.savefig(plot_path)
        plt.show()

    def plot_anomalies(self):
        if self.df.empty:
            print("No data to analyze. Run simulation or load data first.")
            return
        df = self.df.copy()
        df["Anomaly"] = (
            (df["IRR (%)"] > 20) & (df["DSCR"] < 1.0) & (df["Equity Multiple"] < 1.2)
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(
            df["IRR (%)"],
            df["Equity Multiple"],
            c=df["Anomaly"].map({True: "red", False: "blue"}),
            alpha=0.6,
        )
        plt.title("Anomaly Detection: IRR vs Equity Multiple")
        plt.xlabel("IRR (%)")
        plt.ylabel("Equity Multiple")
        plt.grid(True)
        plt.legend(["Normal", "Anomaly"], loc="upper left")
        anomaly_path = self.output_dir / "irr_equity_anomalies.png"
        plt.savefig(anomaly_path)
        plt.show()

        print(f"Anomalies detected: {df['Anomaly'].sum()} out of {len(df)} records")

    def save_outputs(self):
        if self.df.empty:
            print("No data to save. Run simulation or load data first.")
            return
        summary = self.df.describe(percentiles=[0.05, 0.5, 0.95])
        self.df.to_csv(self.output_dir / "simulation_raw_results.csv", index=False)
        summary.to_csv(self.output_dir / "simulation_summary_statistics.csv")
        print("Saved outputs to 'outputs/' folder.")


if __name__ == "__main__":
    sim = RealEstateSimulator()
    scenario_path = Path(__file__).parent / "scenarios.json"
    sim.load_scenarios(scenario_path.as_posix())
    sim.set_scenario("base")
    sim.run_simulation()
    sim.plot_results()
    sim.plot_anomalies()
    sim.save_outputs()

    investment_data = Path(__file__).parent / "investment_data.csv"
    sim.load_investment_data(investment_data.as_posix())
    sim.plot_anomalies()
