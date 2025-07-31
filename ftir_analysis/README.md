# PCA & Spectra – FTIR Wine Analysis

It started with an assignment, just another line in a syllabus, just another file in an inbox. I returned it, thought I'd move on. But curiosity refused to leave quietly.

There was a choice: analyze the spectral fingerprint of wine, or unravel the color chemistry hidden in a Renaissance painting (Da Vinci, I think-or was it someone else cloaked in chiaroscuro?). One path led to pigments and marble halls, the other to tannins and time-stamped absorbance curves.

I chose the wine.

Next thing I knew, PCA was running, plots were blooming, and Cabernet had coordinates.

This repo is the result of that rabbit hole. It's a mixture of science, impulse, and matplotlib therapy-an exploration of FTIR spectroscopy for red wine samples that somehow became a late-night ritual.

---

## What It Does

- Reads the `Wine_Cabernet_Shiraz_FTIR` dataset with the solemnity of a sommelier examining a vintage  
- Runs PCA to expose hidden patterns-tracing spectral structure between Cabernet Sauvignon and Shiraz  
- Synthesizes a fictional mixture of both wines to test where it falls in PCA space  
- Offers multiple visualizations: from raw spectra to principal component loadings to heatmaps glowing like lava

---

## Files of Interest

- `spectra_analysis.py` - the script born from curiosity and possibly sleep deprivation
- `Wine_FTIR_Triplicate_Spectra.csv` - 111 spectra across 37 wines (triplicate samples per bottle)

---

## Data Source

This project uses data from:

**QIBChemometrics** - Public datasets provided by Quadram Institute Bioscience Core Science Resources  
Licensed under **CC0 1.0 Universal**, free for research and analysis  
GitHub repository: [QIBChemometrics](https://github.com/QIBChemometrics)

Dataset originally published in:  
*Multivariate statistics: Considerations and confidences in food authenticity problems*  
Kemsley EK, Defernez M, Marini F (2019), *Food Control*, Volume 105, Pages 102–112

---

## Actions Available

Set `ACTION` in the script to any of the following:

- `"pca_dots"` - scatter PCA plot for Cabernet vs Shiraz  
- `"pca_dots_with_mixture"` - includes the synthetic wine blend  
- `"loadings"` - PCA component loadings across wavenumbers  
- `"spectra"` - absorbance plots for selected wine samples  
- `"mean_spectra"` - average spectra per grape variety  
- `"diff_spectrum"` - spectral subtraction (Cabernet minus Shiraz)  
- `"heatmap"` - visualization of absorbance values across all samples

---

## Why It Exists

After turning in my spectroscopy assignment, I couldn't let go. I had the choice between analyzing wine or exploring color in a Renaissance painting-but the dataset sealed it. Once you start parsing spectral patterns in wine, it's hard to stop. This project is part obsession, part improvisation, and part attempt to explain why FTIR has now become part of my personality.

---

## What's Next?

If curiosity keeps winning:

- I'll build a Streamlit dashboard to let others wander through spectra  
- Try classification models to guess grape variety from absorbance  
- Investigate spectral blending like digital viticulture  
- Maybe even write a paper titled *Cabernet–Syrah Confusion Matrix: A PCA Romance*

In the meantime, clone it, fork it, run it-just don't let your cabernet oxidize while debugging.
