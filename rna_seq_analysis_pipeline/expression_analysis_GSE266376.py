from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure the output directory exists
OUTPUT_DIR = Path(__file__).parent / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data(base_path):
    """Loads raw and TPM normalized count data."""
    print("Loading data...")
    raw_counts = pd.read_csv(
        f"{base_path}/GSE266376_Raw_counts.txt.gz", sep="\t", index_col=0
    )
    tpm_counts = pd.read_csv(
        f"{base_path}/GSE266376_TPM_Normalized_counts.txt.gz", sep="\t", index_col=0
    )
    print("Data loaded successfully.")
    return raw_counts, tpm_counts


def preprocess_data(raw_counts, tpm_counts, min_total_counts=10):
    """Filters out low-count genes from both datasets."""
    print("Preprocessing data...")
    filtered_raw = raw_counts[raw_counts.sum(axis=1) > min_total_counts]
    filtered_tpm = tpm_counts[tpm_counts.sum(axis=1) > min_total_counts]
    print(
        f"Filtered out genes with total counts <= {min_total_counts}. "
        f"Original genes: {raw_counts.shape[0]}, Filtered genes: {filtered_raw.shape[0]}"
    )
    return filtered_raw, filtered_tpm


def perform_exploratory_data_analysis(filtered_tpm):
    """Performs and saves EDA plots like hierarchical clustering and PCA."""
    print("Performing EDA...")

    # Hierarchical Clustering
    print("Generating hierarchical clustering heatmap...")
    top_genes = filtered_tpm.var(axis=1).sort_values(ascending=False).head(20).index
    sns.clustermap(filtered_tpm.loc[top_genes], cmap="viridis", figsize=(12, 8))
    plt.title("Top 20 Most Variable Genes (TPM)")
    plt.savefig(FIGURES_DIR / "top_20_variable_genes_heatmap.png")
    plt.close()
    print(f"Hierarchical clustering heatmap saved to {FIGURES_DIR}.")

    # PCA Plot
    print("Generating PCA plot...")
    X = StandardScaler().fit_transform(filtered_tpm.T)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1])
    for i, sample in enumerate(filtered_tpm.columns):
        plt.text(components[i, 0], components[i, 1], sample, fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of TPM-normalized Expression")
    plt.savefig(FIGURES_DIR / "pca_plot.png")
    plt.close()
    print(f"PCA plot saved to {FIGURES_DIR}.")


def annotate_genes(pval_df):
    """Annotates the top significant genes using MyGene.info."""
    print("Annotating top 10 significant genes...")
    try:
        import mygene

        mg = mygene.MyGeneInfo()

        top_genes = pval_df.head(10).index.tolist()
        annotations = mg.querymany(
            top_genes, scopes="symbol", fields="name,summary", species=9031
        )

        print("\n--- Top 10 Significant Genes and Annotations ---")
        for gene in annotations:
            name = gene.get("name", "N/A")
            summary = gene.get("summary", "N/A")
            print(f"Gene: {gene['query']} - Name: {name}\nSummary: {summary}\n")

    except ImportError:
        print("mygene library not found. Skipping gene annotation.")
        print("Please install it with: pip install mygene")


def plot_volcano(pval_df, log2_fc):
    """Creates and saves a volcano plot."""
    pval_df["log2FC"] = log2_fc
    pval_df["significant"] = (pval_df["p_value"] < 0.05) & (pval_df["log2FC"].abs() > 1)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=pval_df,
        x="log2FC",
        y="-log10(p)",
        hue="significant",
        palette={True: "red", False: "gray"},
        edgecolor=None,
        alpha=0.7,
    )
    plt.title("Volcano Plot: hZ_A vs Ctrl_A")
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-log10(p-value)")
    plt.axhline(-np.log10(0.05), linestyle="--", color="blue", linewidth=1)
    plt.axvline(1, linestyle="--", color="blue", linewidth=1)
    plt.axvline(-1, linestyle="--", color="blue", linewidth=1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "volcano_plot.png")
    plt.close()
    print("Volcano plot saved.")


def analyze_differential_expression(filtered_tpm):
    """
    Calculates log2 fold changes, performs a t-test, and saves the results.

    Returns:
        tuple: A tuple containing:
            - pval_df (pd.DataFrame): DataFrame with p-values and -log10(p).
            - log2_fc (pd.Series): Series containing the log2 fold change for each gene.
    """
    print("Analyzing differential expression...")

    # Define sample groups
    hz_a_cols = [col for col in filtered_tpm.columns if "hZ_A" in col]
    ctrl_a_cols = [col for col in filtered_tpm.columns if "Ctrl_A" in col]

    # Calculate log2 fold change (FC)
    hz_a = filtered_tpm[hz_a_cols].mean(axis=1)
    ctrl_a = filtered_tpm[ctrl_a_cols].mean(axis=1)
    log2_fc = np.log2((hz_a + 1) / (ctrl_a + 1))

    # Plot top FCs
    top_fc = log2_fc.abs().sort_values(ascending=False).head(20)
    top_fc.plot(
        kind="barh", figsize=(10, 6), title="Top 20 Log2 Fold Changes (hZ_A vs Ctrl_A)"
    )
    plt.xlabel("Log2 Fold Change")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_20_log2_fc.png")
    plt.close()
    print(f"Log2 fold change plot saved to {FIGURES_DIR}.")

    # Perform t-test
    print("Performing t-test...")
    p_values = {}
    for gene in filtered_tpm.index:
        hz_vals = filtered_tpm.loc[gene, hz_a_cols]
        ctrl_vals = filtered_tpm.loc[gene, ctrl_a_cols]
        stat, p = ttest_ind(hz_vals, ctrl_vals, equal_var=False)
        p_values[gene] = p

    pval_df = pd.DataFrame.from_dict(p_values, orient="index", columns=["p_value"])
    pval_df["-log10(p)"] = -np.log10(pval_df["p_value"])
    pval_df.sort_values("p_value", inplace=True)

    pval_df.to_csv(OUTPUT_DIR / "t-test_results.csv")
    print(f"T-test results saved to {OUTPUT_DIR / 't-test_results.csv'}.")

    return pval_df, log2_fc


def main():
    """Main function to run the entire analysis pipeline."""
    base_path = Path(__file__).parent

    # Load and preprocess data
    raw_counts, tpm_counts = load_data(base_path)
    filtered_raw, filtered_tpm = preprocess_data(raw_counts, tpm_counts)

    # Exploratory Data Analysis
    perform_exploratory_data_analysis(filtered_tpm)

    # Differential Expression
    pval_df, log2_fc = analyze_differential_expression(filtered_tpm)

    # Volcano Plot
    plot_volcano(pval_df, log2_fc)

    # Gene Annotation
    annotate_genes(pval_df)

    print("Full analysis complete. All results saved to the 'results' directory.")


if __name__ == "__main__":
    main()
