import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
import umap.umap_ as umap
from hdbscan import HDBSCAN
from collections import Counter
import plotly.express as px
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import euclidean_distances

# --- Tuxemon Image Analysis Actions ---
# This script processes Tuxemon battle sprites to extract color features
# and apply various clustering and dimensionality reduction techniques.
# Each 'ACTION' below provides a different insight into the dataset.

# 1. tuxemon_pca_dots:
#    Technical: Plots a 2D scatter of the principal components, capturing the most variance.
#    Down-to-earth: Shows where each Tuxemon "lives" on the two most important scales of visual variation.
#    Use: To see the overall spread and if there are obvious linear groupings in your dataset.

# 2. tuxemon_clustering_viz:
#    Technical: Visualizes KMeans clustering results on the 2D PCA-reduced feature space and displays sample images from each cluster.
#    Down-to-earth: After we sorted Tuxemon into 'K' (e.g., 5) piles based on color similarity, this map shows where those piles ended up on our main visual axes. It then shows example Tuxemon from each pile.
#    Use: To see if your KMeans clusters make sense visually and to inspect example members of each cluster.

# 3. tuxemon_heatmap:
#    Technical: Displays a heatmap of the higher-dimensional PCA components, showing feature intensity per image.
#    Down-to-earth: Think of it like a spreadsheet of key visual traits for each Tuxemon; dark spots mean less of that trait, bright spots mean more.
#    Use: To observe patterns in your raw features, like if many Tuxemon share certain strong or weak visual characteristics across components.

# 4. tuxemon_medoid_images:
#    Technical: Identifies and displays the medoid image for each cluster, which is the data point closest to the cluster's centroid.
#    Down-to-earth: Finds the "most typical" Tuxemon for each group, the one that best represents its cluster.
#    Use: To quickly understand the visual characteristics that define each cluster by looking at its most representative member.

# 5. tuxemon_tsne_viz:
#    Technical: Applies t-SNE, a non-linear dimensionality reduction technique, to embed high-dimensional data into 2D, primarily preserving local similarities.
#    Down-to-earth: Takes all the complex visual data and squishes it into a 2D map, trying to keep similar Tuxemon close together. It's like drawing a social network where friends (similar Tuxemon) stay near each other.
#    Use: Excellent for visualizing natural groupings and sub-structures in complex data that linear PCA might miss. Helps you see if visually similar Tuxemon truly form tight groups.

# 6. tuxemon_umap_viz:
#    Technical: Applies UMAP, another non-linear dimensionality reduction technique, aiming to preserve both local and global data structure in a lower-dimensional embedding.
#    Down-to-earth: Similar to t-SNE, it creates a 2D map. But UMAP tries to keep not just friends close, but also make sure the bigger "neighborhoods" (overall groups) are positioned relatively correctly too.
#    Use: Offers a good balance between speed and quality for visualizing high-dimensional data, often providing more coherent global structures than t-SNE, useful for both visualization and as input for further clustering.

# 7. tuxemon_hdbscan_clustering:
#    Technical: Performs HDBSCAN clustering, a density-based algorithm that discovers clusters of varying densities and identifies noise, without requiring a pre-defined number of clusters or an 'eps' parameter.
#    Down-to-earth: It finds groups of Tuxemon that are "densely packed" together, automatically figuring out how many groups there are. It also identifies loners or outliers that don't fit neatly into any group.
#    Use: When you don't know how many clusters to expect, or if you suspect clusters have different shapes or densities (e.g., some Tuxemon types are very visually consistent, others more varied). Great for finding natural, non-spherical clusters and identifying unique items.

# 8. tuxemon_hierarchical_clustering_viz:
#     Technical: Performs Agglomerative Hierarchical Clustering, building a tree of clusters that can be cut at any level to define groups. The dendrogram visualizes this hierarchy.
#     Down-to-earth: Imagine sorting your Tuxemon by always combining the two most similar ones until they're all in one big family tree. The dendrogram is that family tree, showing how groups merge.
#     Use: When you want to explore relationships between images at different levels of granularity, or you're unsure how many clusters exist and want a visual guide to decide.

# --- Configuration ---
ACTION: str = "tuxemon_pca_dots"
IMAGE_TYPE: str = "-fron"  # or back
IMAGE_EXTENSION: str = "png"
IMAGE_SIZE: tuple[int, int] = (64, 64)
COLOR_MODE: str = "RGB"  # Ensure final conversion is to RGB

FOLDER_NAME: str = "battle"

NUMBER_CLUSTERS: int = 5  # Option related to KMeans and Hierarchical Clustering
LINKAGE_OPTION: str = (
    "ward"  # You can choose linkage methods: 'ward', 'complete', 'average', 'single' ('ward' minimizes variance within clusters.)
)


def load_tuxemon_images(image_dir: Path, target_size: tuple[int, int] = IMAGE_SIZE):
    """
    Loads images from a specified directory, resizes them, and converts to RGB.

    Parameters:
        image_dir (Path): The directory containing the image files.
        target_size (tuple): Desired (width, height) for resizing images.

    Returns:
        tuple: A tuple containing:
            - np.array: A 4D numpy array of image data (num_images, height, width, channels).
            - list: A list of image filenames (stems).
    """
    images = []
    image_names = []
    for img_path in image_dir.glob(f"*{IMAGE_TYPE}.{IMAGE_EXTENSION}"):
        try:
            img = Image.open(img_path)

            # Convert to RGB (safe even if already RGB, handles RGBA by compositing on black)
            if img.mode != COLOR_MODE:
                # Log if image is not already in the target COLOR_MODE
                print(
                    f"Converting image mode from {img.mode} to {COLOR_MODE}: {img_path.name}"
                )
                img = img.convert(COLOR_MODE)  # Convert directly to RGB

            # Resize if needed
            if img.size != target_size:
                img = img.resize(target_size)

            # Append image data and name
            images.append(np.array(img))
            image_names.append(img_path.stem)

        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")

    return np.array(images), image_names


def extract_color_histograms(images, bins: int = 16):
    """
    Extracts concatenated color histograms (R, G, B channels) from a batch of images.

    Parameters:
        images (np.array): A 4D numpy array of image data.
        bins (int): The number of bins to use for each color channel histogram.

    Returns:
        np.array: A 2D numpy array where each row is the concatenated histogram
                  feature vector for an image.
    """
    histograms = []
    for img_array in images:
        # Reshape to (pixels, channels)
        pixels = img_array.reshape(-1, img_array.shape[-1])
        hists_per_image = []
        for c in range(pixels.shape[1]):
            # Extract histogram for each color channel
            hist, _ = np.histogram(pixels[:, c], bins=bins, range=(0, 256))
            hists_per_image.append(hist)
        histograms.append(np.concatenate(hists_per_image))
    return np.array(histograms)


def plot_embedding(data, labels, title, filename_suffix, tuxemon_names):
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=data[:, 0],
        y=data[:, 1],
        hue=labels,
        palette="viridis",
        legend="full",
        s=100,
        alpha=0.8,
    )
    plt.title(title)
    plt.xlabel(f"Component 1 ({filename_suffix})")
    plt.ylabel(f"Component 2 ({filename_suffix})")
    plt.grid(True)

    display_names = [name.replace(f"{IMAGE_TYPE}", "") for name in tuxemon_names]
    num_tuxemon = len(tuxemon_names)
    annotate_density = "medium"  # options: low, medium, high
    density_map = {
        "low": num_tuxemon // 5,
        "medium": num_tuxemon // 10,
        "high": num_tuxemon // 20,
    }
    annotate_stride = max(1, density_map[annotate_density])

    for i in range(num_tuxemon):
        if i % annotate_stride == 0:
            plt.annotate(
                display_names[i],
                (data[i, 0], data[i, 1]),
                textcoords="offset points",
                xytext=(3, 3) if i % 2 == 0 else (-10, -10),
                ha="left",
                fontsize=7,
                alpha=0.7,
            )
    plt.tight_layout()
    plt.show()


def main():
    tuxemon_data_dir = Path(__file__).parent / FOLDER_NAME

    if not tuxemon_data_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {tuxemon_data_dir}.")

    images_array, tuxemon_names = load_tuxemon_images(tuxemon_data_dir)

    if len(tuxemon_names) == 0:
        print(f"No Tuxemon images found in {tuxemon_data_dir}.")
        return  # Exit if no images found
    else:
        print(
            f"Loaded {len(tuxemon_names)} Tuxemon images with shape {images_array.shape[1:]}"
        )

        features = extract_color_histograms(images_array, bins=16)
        print(f"Color histogram features shape: {features.shape}")

        # PCA for 2D visualization (used by KMeans and basic viz)
        pca_tuxemon = PCA(n_components=2)
        components_tuxemon = pca_tuxemon.fit_transform(features)
        print(f"PCA-reduced Tuxemon features shape: {components_tuxemon.shape}")

        # For t-SNE/UMAP, it's generally better to use a higher-dimensional PCA or raw features
        # to preserve more variance before non-linear reduction.
        # Let's use 50 components as a common practice for t-SNE/UMAP input.
        # Adjust n_components if your feature space is smaller or you want more/less variance.
        if (
            features.shape[1] > 2
        ):  # Only do this if original features are higher than 2D
            pca_tuxemon_high_dim = PCA(n_components=min(50, features.shape[1]))
            components_tuxemon_high_dim = pca_tuxemon_high_dim.fit_transform(features)
            print(
                f"Higher-dim PCA for t-SNE/UMAP shape: {components_tuxemon_high_dim.shape}"
            )
        else:
            components_tuxemon_high_dim = features
            print(
                "Features are already low-dimensional, using them directly for t-SNE/UMAP."
            )

        # We fit KMeans here so cluster_assignments, kmeans, and tuxemon_df are always available.
        kmeans = KMeans(n_clusters=NUMBER_CLUSTERS, random_state=42, n_init=10)
        # K-Means on 2D PCA for consistency with viz
        cluster_assignments = kmeans.fit_predict(components_tuxemon)

        tuxemon_df = pd.DataFrame(
            {
                "name": tuxemon_names,
                "pc1": components_tuxemon[:, 0],
                "pc2": components_tuxemon[:, 1],
                "cluster": cluster_assignments,  # Default cluster from KMeans
            }
        )

        if ACTION == "tuxemon_pca_dots":
            plot_embedding(
                components_tuxemon,
                None,
                "PCA Scatter Plot of Tuxemon Images (Color Histograms)",
                "PCA",
                tuxemon_names,
            )

        elif ACTION == "tuxemon_clustering_viz":
            plot_embedding(
                components_tuxemon,
                cluster_assignments,
                f"Tuxemon Clusters in PCA Space (KMeans with {NUMBER_CLUSTERS} clusters)",
                "PCA",
                tuxemon_names,
            )

            cluster_counts = Counter(cluster_assignments)
            print("\n KMeans Cluster Summary:")
            for cluster_id, count in sorted(cluster_counts.items()):
                print(f"  Cluster {cluster_id}: {count} sprites")

            for cluster_id in range(NUMBER_CLUSTERS):
                cluster_tuxemon_names = tuxemon_df[tuxemon_df["cluster"] == cluster_id][
                    "name"
                ].tolist()

                if not cluster_tuxemon_names:
                    fig, ax = plt.subplots(figsize=(4, 2))
                    ax.text(
                        0.5,
                        0.5,
                        "No images to display for this cluster.",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=12,
                        color="gray",
                        transform=ax.transAxes,
                    )
                    ax.axis("off")
                    plt.title(f"--- Cluster {cluster_id} ---", fontsize=10, loc="left")
                    plt.show()
                    continue

                np.random.shuffle(cluster_tuxemon_names)
                num_to_display = min(5, len(cluster_tuxemon_names))

                if num_to_display > 0:
                    fig, axes = plt.subplots(
                        1, num_to_display, figsize=(2 * num_to_display, 2.5)
                    )
                    axes = axes.flatten() if num_to_display > 1 else [axes]
                    for i, name in enumerate(cluster_tuxemon_names[:num_to_display]):
                        img_path = tuxemon_data_dir / f"{name}.{IMAGE_EXTENSION}"
                        if img_path.exists():
                            try:
                                img = Image.open(img_path)
                                axes[i].imshow(img)
                                clean_name = name.replace(f"{IMAGE_TYPE}", "")
                                axes[i].set_title(
                                    f"C{cluster_id}: {clean_name}", fontsize=8
                                )
                                axes[i].axis("off")
                            except Exception as e:
                                print(
                                    f"Could not load image {name}.{IMAGE_EXTENSION} for display: {e}"
                                )
                                axes[i].set_title(f"Error loading {name}", fontsize=8)
                                axes[i].axis("off")
                        else:
                            axes[i].set_title(f"Missing {name}", fontsize=8)
                            axes[i].axis("off")
                    plt.tight_layout()
                    plt.show()

        elif ACTION == "tuxemon_heatmap":
            # Reduce high-dimensional features to 50 principal components for visual inspection
            pca_for_heatmap = PCA(n_components=min(50, features.shape[1]))
            components_for_heatmap = pca_for_heatmap.fit_transform(features)

            # Heatmap of PCA-reduced features
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                components_for_heatmap,
                cmap="viridis",
                yticklabels=False,
                xticklabels=True,
            )
            plt.title("Tuxemon Features Heatmap (Top PCA Components)")
            plt.xlabel("Principal Components")
            plt.ylabel("Tuxemon Images")
            plt.tight_layout()
            plt.show()

            # Printed Insight: Stats + Outlier Detection
            print("\nHeatmap overview:")
            print("→ Number of Tuxemon:", components_for_heatmap.shape[0])
            print("→ Number of PCA components:", components_for_heatmap.shape[1])

            top_component = components_for_heatmap[:, 0]
            print("→ First PCA component stats:")
            print("   Mean:", np.mean(top_component))
            print("   Std Dev:", np.std(top_component))
            print("   Min:", np.min(top_component))
            print("   Max:", np.max(top_component))

            # Highlight 5 most visually extreme sprites (based on PCA[0])
            most_positive = np.argsort(top_component)[-5:]
            most_negative = np.argsort(top_component)[:5]

            print("\nTop 5 visually intense Tuxemon (high PCA[0]):")
            for idx in reversed(most_positive):
                name = tuxemon_names[idx].replace(f"{IMAGE_TYPE}", "")
                print(f"  {name}: {top_component[idx]:.3f}")

            print("\nTop 5 visually muted Tuxemon (low PCA[0]):")
            for idx in most_negative:
                name = tuxemon_names[idx].replace(f"{IMAGE_TYPE}", "")
                print(f"  {name}: {top_component[idx]:.3f}")

        elif ACTION == "tuxemon_medoid_images":
            print(f"\n--- Tuxemon Medoid Images for {NUMBER_CLUSTERS} Clusters ---")
            centroids = kmeans.cluster_centers_

            for cluster_id in range(NUMBER_CLUSTERS):
                cluster_indices = np.where(cluster_assignments == cluster_id)[0]
                if len(cluster_indices) == 0:
                    print(f"Cluster {cluster_id} is empty, no medoid image.")
                    continue

                cluster_features = components_tuxemon[cluster_indices]
                distances_to_centroid = euclidean_distances(
                    cluster_features, centroids[cluster_id].reshape(1, -1)
                )

                medoid_local_idx = np.argmin(distances_to_centroid)
                medoid_global_idx = cluster_indices[medoid_local_idx]

                medoid_name = tuxemon_names[medoid_global_idx]
                medoid_img_path = tuxemon_data_dir / f"{medoid_name}.{IMAGE_EXTENSION}"

                if medoid_img_path.exists():
                    fig, ax = plt.subplots(figsize=(2.5, 2.5))
                    img = Image.open(medoid_img_path)
                    ax.imshow(img)
                    clean_medoid_name = medoid_name.replace(f"{IMAGE_TYPE}", "")
                    ax.set_title(
                        f"C{cluster_id} Medoid:\n{clean_medoid_name}", fontsize=9
                    )
                    print(f"C{cluster_id} Medoid: {clean_medoid_name}")
                    ax.axis("off")
                    plt.show()
                else:
                    print(
                        f"Medoid image '{medoid_name}.{IMAGE_EXTENSION}' for Cluster {cluster_id} not found."
                    )

        elif ACTION == "tuxemon_tsne_viz":
            print("\n--- Running t-SNE for Visualization ---")
            # Perplexity is crucial; adjust based on dataset size (often 5 to 50)
            # Using a fixed random_state for reproducibility
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(tuxemon_names) - 1),
            )

            # Use the higher-dimensional PCA output as input for t-SNE
            tsne_results = tsne.fit_transform(components_tuxemon_high_dim)

            plot_embedding(
                tsne_results,
                cluster_assignments,
                f"Tuxemon Clusters in t-SNE Space (KMeans assignments, Perplexity={tsne.perplexity})",
                "t-SNE",
                tuxemon_names,
            )
            print(
                "Note: t-SNE preserves local structure well, but distances on the plot don't reflect original distances accurately."
            )

        elif ACTION == "tuxemon_umap_viz":
            print("\n--- Running UMAP for Visualization ---")
            # Try `pip install umap-learn` if you get an import error.
            # n_neighbors and min_dist are key UMAP parameters
            reducer = umap.UMAP(
                n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
            )

            # Use the higher-dimensional PCA output as input for UMAP
            umap_results = reducer.fit_transform(components_tuxemon_high_dim)

            plot_embedding(
                umap_results,
                cluster_assignments,
                f"Tuxemon Clusters in UMAP Space (KMeans assignments)",
                "UMAP",
                tuxemon_names,
            )
            print(
                "Note: UMAP balances local and global structure preservation, often faster than t-SNE."
            )

        elif ACTION == "tuxemon_hdbscan_clustering":
            print("\n--- Running HDBSCAN Clustering ---")
            # HDBSCAN does not need 'eps'. Key parameters are min_cluster_size and min_samples.
            # min_cluster_size: The smallest size a cluster can be.
            # min_samples: Similar to min_cluster_size, but also impacts how conservative the clustering is.
            # Generally, min_samples should be <= min_cluster_size.
            # For 412 images, 5-15 is a good starting range for min_cluster_size.

            # Use components_tuxemon_high_dim as input for HDBSCAN
            # Set min_samples to None to default to min_cluster_size
            hdbscan_clusterer = HDBSCAN(
                min_cluster_size=5,  # less strict about cluster size
                min_samples=1,  # allows more flexible grouping
                prediction_data=True,
            )
            hdbscan_assignments = hdbscan_clusterer.fit_predict(
                components_tuxemon_high_dim
            )

            n_hdbscan_clusters = len(set(hdbscan_assignments)) - (
                1 if -1 in hdbscan_assignments else 0
            )
            n_noise_points_hdbscan = list(hdbscan_assignments).count(-1)

            print("\nHDBSCAN Cluster Summary")
            cluster_counts = Counter(hdbscan_assignments)  # Correctly count all labels
            for cluster_id, count in sorted(cluster_counts.items()):
                if cluster_id == -1:
                    print(f"  • Noise (outliers): {count} sprites")
                else:
                    print(f"  • Cluster {cluster_id}: {count} sprites")

            if n_hdbscan_clusters == 0 and n_noise_points_hdbscan > 0:
                print(
                    " → Only noise points found by HDBSCAN. Try adjusting 'min_cluster_size' or 'min_samples'."
                )
            elif n_hdbscan_clusters == 0:
                print(
                    " → No clusters found by HDBSCAN. Try adjusting 'min_cluster_size'."
                )
            elif n_noise_points_hdbscan > 0:
                print(
                    f" → Noise points (-1 label) indicate outliers. Consider them in your analysis."
                )

            # Plotting HDBSCAN results
            hdbscan_df = pd.DataFrame(
                {
                    "name": tuxemon_names,
                    "pc1": components_tuxemon[
                        :, 0
                    ],  # Use 2D PCA for plotting visualization
                    "pc2": components_tuxemon[:, 1],
                    "cluster": hdbscan_assignments,
                    "view": ["front" for _ in tuxemon_names],
                }
            )

            plt.figure(figsize=(12, 10))
            noise_df_hdbscan = hdbscan_df[hdbscan_df["cluster"] == -1]
            noise_names = noise_df_hdbscan["name"].tolist()

            print(f"\n Noise Tuxemon ({len(noise_names)}) examples:")
            print(
                "  "
                + ", ".join([name.replace(f"{IMAGE_TYPE}", "") for name in noise_names])
            )
            clustered_df_hdbscan = hdbscan_df[hdbscan_df["cluster"] != -1]

            if not clustered_df_hdbscan.empty:
                sns.scatterplot(
                    x="pc1",
                    y="pc2",
                    hue="cluster",
                    data=clustered_df_hdbscan,
                    palette="viridis",
                    legend="full",
                    s=100,
                    alpha=0.8,
                    hue_order=sorted(clustered_df_hdbscan["cluster"].unique()),
                )

            if not noise_df_hdbscan.empty:
                plt.scatter(
                    noise_df_hdbscan["pc1"],
                    noise_df_hdbscan["pc2"],
                    color="black",
                    marker="x",
                    s=50,
                    alpha=0.6,
                    label="Noise (-1)",
                )
                plt.legend()

            plt.title(
                f"Tuxemon Clusters (HDBSCAN: min_cluster_size={hdbscan_clusterer.min_cluster_size})"
            )
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Quantitative metrics for HDBSCAN results
            if n_hdbscan_clusters > 1:
                non_noise_indices_hdbscan = np.where(hdbscan_assignments != -1)[0]
                if (
                    len(non_noise_indices_hdbscan) > 1
                    and len(set(hdbscan_assignments[non_noise_indices_hdbscan])) > 1
                ):
                    hdbscan_silhouette = silhouette_score(
                        components_tuxemon_high_dim[non_noise_indices_hdbscan],
                        hdbscan_assignments[non_noise_indices_hdbscan],
                    )
                    print(
                        f"Silhouette Score (excluding noise): {hdbscan_silhouette:.3f}"
                    )
                else:
                    print(
                        "Not enough non-noise points or clusters to compute Silhouette Score for HDBSCAN."
                    )

            print("\nSample images from HDBSCAN clusters:")
            for cluster_id in sorted(set(hdbscan_assignments)):
                if cluster_id == -1:
                    print(
                        f"\n--- Noise Points ({list(hdbscan_assignments).count(-1)} Tuxemon) ---"
                    )
                    # Display noise points as well if desired
                    noise_tuxemon_names = [
                        tuxemon_names[i]
                        for i, cid in enumerate(hdbscan_assignments)
                        if cid == -1
                    ]
                    np.random.shuffle(noise_tuxemon_names)
                    num_to_display_noise = min(5, len(noise_tuxemon_names))

                    if num_to_display_noise > 0:
                        fig, axes = plt.subplots(
                            1,
                            num_to_display_noise,
                            figsize=(2 * num_to_display_noise, 2.5),
                        )
                        axes = axes.flatten() if num_to_display_noise > 1 else [axes]
                        for i, name in enumerate(
                            noise_tuxemon_names[:num_to_display_noise]
                        ):
                            img_path = tuxemon_data_dir / f"{name}.{IMAGE_EXTENSION}"
                            if img_path.exists():
                                try:
                                    img = Image.open(img_path)
                                    axes[i].imshow(img)
                                    clean_name = name.replace(f"{IMAGE_TYPE}", "")
                                    axes[i].set_title(
                                        f"Noise: {clean_name}", fontsize=8
                                    )
                                    axes[i].axis("off")
                                except Exception as e:
                                    print(
                                        f"Could not load image {name}.{IMAGE_EXTENSION} for display: {e}"
                                    )
                                    axes[i].set_title(
                                        f"Error loading {name}", fontsize=8
                                    )
                                    axes[i].axis("off")
                            else:
                                axes[i].set_title(f"Missing {name}", fontsize=8)
                                axes[i].axis("off")
                        plt.tight_layout()
                        plt.show()
                    continue

                cluster_tuxemon_names = [
                    tuxemon_names[i]
                    for i, cid in enumerate(hdbscan_assignments)
                    if cid == cluster_id
                ]

                if not cluster_tuxemon_names:
                    continue

                print(
                    f"\n--- Cluster {cluster_id} (contains {len(cluster_tuxemon_names)} Tuxemon) ---"
                )
                np.random.shuffle(cluster_tuxemon_names)
                num_to_display = min(5, len(cluster_tuxemon_names))

                fig, axes = plt.subplots(
                    1, num_to_display, figsize=(2 * num_to_display, 2.5)
                )
                axes = axes.flatten() if num_to_display > 1 else [axes]

                for i, name in enumerate(cluster_tuxemon_names[:num_to_display]):
                    img_path = tuxemon_data_dir / f"{name}.{IMAGE_EXTENSION}"
                    if img_path.exists():
                        try:
                            img = Image.open(img_path)
                            axes[i].imshow(img)
                            clean_name = name.replace(f"{IMAGE_TYPE}", "")
                            axes[i].set_title(
                                f"C{cluster_id}: {clean_name}", fontsize=8
                            )
                            axes[i].axis("off")
                        except Exception as e:
                            print(
                                f"Could not load image {name}.{IMAGE_EXTENSION} for display: {e}"
                            )
                            axes[i].set_title(f"Error loading {name}", fontsize=8)
                            axes[i].axis("off")
                    else:
                        axes[i].set_title(f"Missing {name}", fontsize=8)
                        axes[i].axis("off")
                plt.tight_layout()
                plt.show()

        elif ACTION == "tuxemon_hierarchical_clustering_viz":
            print("\n--- Running Agglomerative Hierarchical Clustering ---")
            # We'll use the higher-dimensional PCA output for clustering
            agg_clustering = AgglomerativeClustering(
                n_clusters=NUMBER_CLUSTERS, linkage=LINKAGE_OPTION
            )
            hierarchical_assignments = agg_clustering.fit_predict(
                components_tuxemon_high_dim
            )

            print("\nCluster Summary (Hierarchical):")
            hier_counts = Counter(hierarchical_assignments)
            for cluster_id, count in sorted(hier_counts.items()):
                print(f"  Cluster {cluster_id}: {count} sprites")

            plot_embedding(
                components_tuxemon,
                hierarchical_assignments,  # Plotting on 2D PCA for visual comparison
                f"Tuxemon Clusters in PCA Space (Hierarchical Clustering with {NUMBER_CLUSTERS} clusters)",
                "PCA",
                tuxemon_names,
            )

            # Optional: Display dendrogram (requires scipy linkage and dendrogram)
            print("\n--- Generating Dendrogram (Hierarchical Clustering) ---")
            try:
                # Calculate the linkage matrix
                Z = linkage(components_tuxemon_high_dim, method=LINKAGE_OPTION)

                plt.figure(figsize=(20, 10))
                plt.title("Hierarchical Clustering Dendrogram for Tuxemon Images")
                plt.xlabel("Tuxemon Index (or Cluster Size)")
                plt.ylabel("Distance")
                dendrogram(
                    Z,
                    leaf_rotation=90.0,  # rotates the x axis labels
                    leaf_font_size=8.0,  # font size for the x axis labels
                    # truncate_mode='lastp',  # show only the last p merged clusters
                    # p=30, # show only the last 30 merged clusters
                    show_leaf_counts=True,  # show the number of points in each leaf
                    labels=[
                        name.replace(f"{IMAGE_TYPE}", "") for name in tuxemon_names
                    ],  # Clean names for dendrogram
                )
                plt.tight_layout()
                plt.show()
                print(
                    "The dendrogram shows the merges at each step. You can 'cut' the tree at a certain height to define clusters."
                )

            except ImportError:
                print(
                    "SciPy is required for plotting dendrograms. Please install it (`pip install scipy`)."
                )
            except Exception as e:
                print(f"Error generating dendrogram: {e}")

            # You can also run quantitative metrics on Hierarchical results
            if NUMBER_CLUSTERS > 1:
                hierarchical_silhouette = silhouette_score(
                    components_tuxemon_high_dim, hierarchical_assignments
                )
                print(f"Silhouette Score: {hierarchical_silhouette:.3f}")
                if hierarchical_silhouette >= 0.7:
                    print(" Strong cluster separation.")
                elif hierarchical_silhouette >= 0.4:
                    print(" Moderate separation, but with some fuzziness.")
                else:
                    print(
                        " Weak clustering—consider adjusting linkage method or PCA dimensionality."
                    )
            else:
                print(
                    "Cannot compute Silhouette Score for Hierarchical Clustering with 1 or fewer clusters."
                )

        else:
            print(f"Unknown action: {ACTION}")


if __name__ == "__main__":
    main()
