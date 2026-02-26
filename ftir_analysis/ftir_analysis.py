from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def inspect_data(file_path: Path) -> None:
    import pandas as pd

    try:
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            print("CSV loaded successfully.\n")
            print("Head of the dataset:")
            print(df.head(), "\n")
            print("Shape of dataset:", df.shape)
            print("Column names:", df.columns.tolist(), "\n")
            print("Data types:")
            print(df.dtypes, "\n")
            print("Unique values per column:")
            for col in df.columns:
                unique_vals = df[col].unique()
                print(f"  {col}: {len(unique_vals)} unique values")
                if len(unique_vals) <= 10:
                    print(f"    Sample values: {unique_vals[:5]}")
            print()
            missing = df.isnull().sum()
            print("Missing values:")
            print(
                missing[missing > 0] if missing.sum() > 0 else "  No missing values.\n"
            )
            print("Descriptive stats for numerical columns:")
            print(df.describe(), "\n")
            print("Random samples:")
            print(df.sample(3), "\n")
        elif file_path.suffix == ".npy":
            arr = np.load(file_path)
            print("NPY loaded successfully. Shape:", arr.shape, "\n")
            print("Sample values:", arr[:5])
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        print("Error loading file:", e)
        return


# Options: "pca_dots", "loadings", "spectra", "mean_spectra", "diff_spectrum", "heatmap"
ACTION: str = "pca_dots"  # Default action, can be changed by user
INSPECT: bool = True  # Flag to control data inspection

data_dir = Path(__file__).parent / "data"
wine_ftir = data_dir / "Wine_FTIR_Triplicate_Spectra.csv"


if not wine_ftir.exists():
    raise FileNotFoundError(f"Missing file: {wine_ftir}")


# Load and prepare data
df = pd.read_csv(wine_ftir)
if INSPECT:
    inspect_data(wine_ftir)


wavenumbers = df["Wavenumbers"].values
spectra = df.drop(columns="Wavenumbers").T
labels = spectra.index.to_series().apply(
    lambda name: "Cabernet" if "Cab" in name else "Shiraz"
)

# This PCA model is now used for all PCA-related actions
pca = PCA(n_components=2)
components = pca.fit_transform(spectra.values)

if ACTION == "pca_dots":
    plt.figure(figsize=(8, 6))
    for wine_type in ["Cabernet", "Shiraz"]:
        idx = labels == wine_type
        plt.scatter(components[idx, 0], components[idx, 1], label=wine_type, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter: Cabernet vs Shiraz")
    plt.legend()
    plt.tight_layout()
    plt.show()

elif ACTION == "loadings":
    pc1_loadings = pca.components_[0]
    pc2_loadings = pca.components_[1]
    plt.figure(figsize=(10, 5))
    plt.plot(wavenumbers, pc1_loadings, label="PC1 Loadings (from data)")
    plt.plot(wavenumbers, pc2_loadings, label="PC2 Loadings (from data)")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Loading Weight")
    plt.title("PCA Loadings Across FTIR Spectrum")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

elif ACTION == "spectra":
    chosen_samples = ["Wine_01_Cab_Rep1", "Wine_36_Syr_Rep1"]
    plt.figure(figsize=(10, 5))
    for name in chosen_samples:
        plt.plot(wavenumbers, df[name].values, label=name)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorbance")
    plt.title("Selected FTIR Spectra")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

elif ACTION == "mean_spectra":
    cab_cols = [c for c in df.columns if "Cab" in c]
    syr_cols = [c for c in df.columns if "Syr" in c]
    cab_mean = df[cab_cols].mean(axis=1)
    syr_mean = df[syr_cols].mean(axis=1)
    plt.plot(wavenumbers, cab_mean, label="Cabernet Mean")
    plt.plot(wavenumbers, syr_mean, label="Shiraz Mean")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Mean Absorbance")
    plt.title("Mean FTIR Spectra: Cabernet vs Shiraz")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

elif ACTION == "diff_spectrum":
    cab_cols = [c for c in df.columns if "Cab" in c]
    syr_cols = [c for c in df.columns if "Syr" in c]
    cab_mean = df[cab_cols].mean(axis=1)
    syr_mean = df[syr_cols].mean(axis=1)
    diff = cab_mean - syr_mean
    plt.plot(wavenumbers, diff)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorbance Difference")
    plt.title("Spectral Difference: Cabernet - Shiraz")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

elif ACTION == "heatmap":
    sns.heatmap(spectra.values, cmap="magma", xticklabels=50)
    plt.title("FTIR Spectra Heatmap")
    plt.xlabel("Wavenumbers")
    plt.ylabel("Wine Samples")
    plt.tight_layout()
    plt.show()
