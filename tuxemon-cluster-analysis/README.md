# Clusters & Colors – Tuxemon Sprite Analysis

This repo documents what happens when unsupervised learning meets pixel art-specifically, battle sprites from the Tuxemon GitHub repository.

Originally conceived as a color-based clustering experiment, it evolved into a miniature visual taxonomy: sprites mapped into 2D space, sorted by hue, shape, and whatever secrets lurk in 64×64 pixels. PCA reveals the dominant visual dimensions. UMAP and t-SNE tell stories of local similarity. KMeans, HDBSCAN, and hierarchical clustering sketch out families.

It's a dataset exploration, a clustering sandbox, and a playful tool to discover visual relationships that may (or may not) mean something. The sprites started as pixels, now some of them have cousins.

## What It Does

- Extracts pixel-level color features from Tuxemon battle sprites (`.png`, front-facing)
- Runs PCA to surface the most visually relevant dimensions
- Applies KMeans, HDBSCAN, and hierarchical clustering to detect sprite families
- Embeds sprites in 2D using t-SNE and UMAP for visually coherent maps
- Displays medoid images as "most typical" examples of each cluster
- Generates heatmaps for trait intensities across PCA components
- Builds dendrograms to trace sprite lineage (sort of)

## Files of Interest

- `tuxemon_analysis.py` - the main script, born of curiosity and powered by matplotlib therapy
- `battle/` - folder of 64×64 front-facing Tuxemon battle sprites (RGB PNGs, courtesy of the Tuxemon project)

## Supported Actions

Set `ACTION` in the script to any of the following:

- `"tuxemon_pca_dots"` - scatter plot of principal components, mapping sprite variation
- `"tuxemon_clustering_viz"` - PCA map with KMeans clusters and sample Tuxemon from each group
- `"tuxemon_heatmap"` - component-wise heatmap of extracted visual traits
- `"tuxemon_medoid_images"` - displays the most representative (medoid) image per cluster
- `"tuxemon_tsne_viz"` - t-SNE embedding for local structure preservation
- `"tuxemon_umap_viz"` - UMAP embedding balancing global and local cohesion
- `"tuxemon_hdbscan_clustering"` - density-based clustering with automatic detection of groups and outliers
- `"tuxemon_hierarchical_clustering_viz"` - agglomerative clustering visualized via dendrogram

## Dependencies

This script uses the following Python packages:

- numpy
- pandas
- matplotlib
- seaborn
- Pillow
- scikit-learn
- umap-learn
- hdbscan
- plotly
- scipy

## Why It Exists

It started after a feverish FTIR assignment, wine spectroscopy, PCA scatterplots, and late-night plotting rituals. But the delirium didn't end there. Somewhere between absorbance curves and visual embeddings, I started wondering: what other domains could benefit from dimensionality reduction and clustering?

That thought stuck around, softly buzzing in the background of my brain, perhaps the result of years studying academic philosophy, where even a PNG file can seem ontologically suspect. I was still half-convinced it wasn't worth doing. That line from Dignam "Maybe. Maybe not." landed long before The Departed reached its bloody climax. But it lingered. Something about its sharpness, its refusal to commit, matched the tone I needed to start this project.

Since I'd been contributing to the Tuxemon project, it felt natural to experiment with their sprites. Could color patterns and clustering make their visual ecosystem more navigable? Could this be genuinely useful - or at least weirdly satisfying?

This project sits somewhere between impulse and structure, curiosity and clustering. It's part fever dream, part philosophy residue, and part desire to make unsupervised learning actually fun.

## What I Discovered – Front vs Back Sprite Dynamics

Once I ran the same clustering and dimensionality reduction pipeline on the back-facing sprites, things unraveled, spectacularly. And right then, like a cinematic glitch in my brain, Willem Dafoe's voice popped into my head, "There was a firefight!", as he spun around, guns blazing in a chaotic ballet of bullets and drama. The Boondock Saints? Unforgettable.

### 1. HDBSCAN: When Uniformity Vanishes

HDBSCAN is most sensitive to shared visual traits, and its results told the loudest story:

- **Front Sprites**: Most grouped into meaningful clusters. Just 206 of 412 were noise.
- **Back Sprites**: A staggering 359 out of 412 were classified as noise. Less than 15% landed in clusters with more than 5 images. The rest formed scattered, tiny pockets - clusters of 5, 6, 8, maybe 10.

**Interpretation**: The back sprites are visually fragmented. They lack a dominant palette, structural motif, or shared silhouette. From HDBSCAN's point of view, many of them are loners - creatures without a visual family.

### 2. PCA: Scatter in Spectral Space

Dimensionality reduction through PCA gave another strong clue:

- **PC1 Range (Front Sprites)**: –3201.02 to +1672.17  
- **PC1 Range (Back Sprites)**: –3289.08 to +2010.50

**Interpretation**: The variance along the primary PCA axis is broader for back sprites. They exhibit higher contrast, broader ranges of brightness, and more erratic distribution. If front sprites sit on a visual spectrum, the back sprites dance off it.

### 3. KMeans Medoids: Who Represents Whom?

Even when forcing KMeans into five clusters, the medoid selections changed entirely:

- **Front Medoids**: weavifly (purple), nudikill (red-orange), trojerror (cool tones), toucanary (contrasting), rabbitosaur (earth tones).
- **Back Medoids**: dollfin, vividactil, possessun, agnite, brachifor - creatures with entirely new palette compositions.

**Interpretation**: Representation changed completely. The most "central" sprite in each group no longer shared color logic with its front-facing version. What front-view clustering once captured - mood, identity, even role - was reset from behind.

### Final Thought

I suspected early on that back sprites might resist clustering. This project confirms it with data. The PCA spread, HDBSCAN silence, and medoid shake-up all shout the same message: perspective matters. The reverse view isn't just a technical variation - it's a philosophical one, too.

When your dataset turns around, everything changes.

## What's Next

If curiosity keeps leading the way:

### 1. Enriched Feature Engineering  
The current analysis relies primarily on color histograms, which offer limited insight when visual diversity spikes - as seen in the back-facing sprites. Expanding to include:

- **Shape descriptors** to capture structural motifs in sprite outlines  
- **Texture analysis**, such as Gabor filters or Local Binary Patterns, to detect subtle patterns  
- **Edge detection**, which could highlight silhouette variance and compositional rhythm  

This enhanced feature set may yield tighter clusters, particularly for sprites that defy color-based categorization.

### 2. Practical Application & Quality Control  
Beyond academic interest, this pipeline has direct utility for game asset management. Proposed applications include:

- **Automated palette validation**: Flag assets that drift from canonical color schemes  
- **Unit tests for visual consistency**: Check number of unique colors, brightness profiles, or format compliance  
- **Curated cluster tagging**: Organize sprite assets by design families, enhancing modularity and reuse

These tools could serve as a lightweight form of visual QA - ensuring every sprite aligns with intended design standards.

---

In short, the sprite clustering toolkit isn't just exploratory - it could evolve into infrastructure. Whether helping artists maintain consistency or enabling designers to visualize taxonomies at scale, the possibilities feel far from pixel-sized.
