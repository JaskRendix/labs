"""
https://www.crystallography.net/cod/1000227.html
"""

from pathlib import Path

import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure

AVOGADRO_CONSTANT = 6.02214076e23


CONFIG = {
    "wavelength": "CuKa",  # "MoKa" or "CoKa"
    "data_dir": Path(__file__).parent,
    "cif_file": "1000227.cif",
    "top_n_peaks": 10,  # Number of top intensity peaks to annotate
}


def load_structure(cif_path: Path) -> Structure:
    """Loads a pymatgen Structure object from a CIF file."""
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found at: {cif_path}")
    return Structure.from_file(cif_path)


def calculate_xrd_pattern(structure: Structure, wavelength: str) -> dict:
    """Calculates the XRD pattern using pymatgen."""
    xrd_calculator = XRDCalculator(wavelength=wavelength)
    pattern = xrd_calculator.get_pattern(structure)
    return pattern


def calculate_theoretical_density(structure: Structure) -> float:
    """
    Calculates the theoretical density in g/cm³.
    """
    formula_weight = structure.composition.weight
    volume_A3 = structure.lattice.volume
    density_g_per_A3 = formula_weight / (volume_A3 * AVOGADRO_CONSTANT)
    density_g_per_cm3 = density_g_per_A3 * 1e24
    return density_g_per_cm3


def analyze_coordination(structure, element):
    cnn = CrystalNN()
    for i, site in enumerate(structure):
        if site.species_string == element:
            neighbors = cnn.get_nn_info(structure, i)
            print(f"\n{element} site {i} coordination:")
            for n in neighbors:
                print(
                    f"  → {n['site'].species_string} at {n['site'].frac_coords}, weight = {n['weight']:.2f}"
                )


def plot_xrd_pattern(pattern, title: str):
    """Generates and displays a plot of the XRD pattern."""
    plt.figure(figsize=(12, 8))
    plt.plot(pattern.x, pattern.y, color="blue", linewidth=2, label="Simulated Pattern")

    # Add peak annotations for the top N peaks
    for i in range(min(CONFIG["top_n_peaks"], len(pattern.x))):
        x = pattern.x[i]
        y = pattern.y[i]
        hkls = pattern.hkls[i]
        label = ", ".join([f"({', '.join(map(str, hkl['hkl']))})" for hkl in hkls])

        # Adjust text position and style for readability
        plt.text(
            x,
            y + 5,
            label,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=9,
            color="red",
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"
            ),
        )

    plt.title(title, fontsize=16)
    plt.xlabel("2θ (degrees)", fontsize=14)
    plt.ylabel("Intensity (a.u.)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the XRD simulation and plotting."""
    cif_path = CONFIG["data_dir"] / CONFIG["cif_file"]

    try:
        structure = load_structure(cif_path)

        print("\nUnit Cell Volume:")
        print(f"  Volume = {structure.lattice.volume:.2f} Å³")
        print("\nTheoretical Density:")
        density = calculate_theoretical_density(structure)
        print(f"  Theoretical Density = {density:.4f} g/cm³")

        pattern = calculate_xrd_pattern(structure, CONFIG["wavelength"])

        # Get the compound's formula for the title
        formula = structure.composition.reduced_formula
        plot_title = f"Simulated XRD Pattern of {formula} with Miller Indices"

        plot_xrd_pattern(pattern, plot_title)

        # Analyze coordination environments
        print("\nCoordination Analysis:")
        analyze_coordination(structure, "Mn")
        analyze_coordination(structure, "Fe")

        # Print structure summary
        print("\nStructure Summary:")
        print(structure)

        # Print lattice parameters
        print("\nLattice Parameters:")
        print(f"  a = {structure.lattice.a:.4f} Å")
        print(f"  b = {structure.lattice.b:.4f} Å")
        print(f"  c = {structure.lattice.c:.4f} Å")
        print(f"  α = {structure.lattice.alpha:.2f}°")
        print(f"  β = {structure.lattice.beta:.2f}°")
        print(f"  γ = {structure.lattice.gamma:.2f}°")

        # Print atomic sites
        print("\nAtomic Sites:")
        for site in structure:
            print(f"  {site.species_string} at fractional coords {site.frac_coords}")

        # Print chemical formula
        print("\nChemical Formula:")
        print(f"  Reduced: {structure.composition.reduced_formula}")
        print(f"  Full: {structure.composition.formula}")

        # Print top XRD peaks
        print(f"\nTop {CONFIG['top_n_peaks']} XRD Peaks:")
        for i in range(min(CONFIG["top_n_peaks"], len(pattern.x))):
            x = pattern.x[i]
            y = pattern.y[i]
            hkls = pattern.hkls[i]
            hkl_labels = ", ".join(
                [f"({', '.join(map(str, hkl['hkl']))})" for hkl in hkls]
            )
            print(f"  2θ = {x:.2f}°, Intensity = {y:.2f}, HKL = {hkl_labels}")

        poscar_path = CONFIG["data_dir"] / "POSCAR"
        structure.to(filename=f"{str(poscar_path)}.json", fmt="json")
        print(f"\n POSCAR file saved to: {poscar_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
