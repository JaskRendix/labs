import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Element, Structure

AVOGADRO_CONSTANT = 6.02214076e23

CONFIG = {
    "wavelength": "CuKa",
    "data_dir": Path(__file__).parent,
    "cif_file": "1000227.cif",
    "dopants": ["Co", "Zn"],
    "target_element": "Mn",
    "doping_fraction": 0.1,
    "top_n_peaks": 10,
}


def load_structure(cif_path: Path) -> Structure:
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found at: {cif_path}")
    return Structure.from_file(cif_path)


def calculate_xrd_pattern(structure: Structure, wavelength: str):
    xrd_calculator = XRDCalculator(wavelength=wavelength)
    return xrd_calculator.get_pattern(structure)


def calculate_theoretical_density(structure: Structure) -> float:
    formula_weight = structure.composition.weight
    volume_A3 = structure.lattice.volume
    density_g_per_A3 = formula_weight / (volume_A3 * AVOGADRO_CONSTANT)
    return density_g_per_A3 * 1e24


def analyze_coordination(structure, element):
    cnn = CrystalNN()
    for i, site in enumerate(structure):
        if site.specie.symbol == element:
            neighbors = cnn.get_nn_info(structure, i)
            print(f"\n{element} site {i} coordination:")
            for n in neighbors:
                print(
                    f"  → {n['site'].species_string} at {n['site'].frac_coords}, weight = {n['weight']:.2f}"
                )


def simulate_doping(
    structure,
    target_element,
    dopant_element,
    fraction=0.1,
    min_atoms_to_replace=1,
    verbose=True,
):
    new_structure = structure.copy()
    target_indices = [
        i
        for i, site in enumerate(new_structure)
        if site.specie.symbol == target_element
    ]

    if not target_indices:
        raise ValueError(f"No atoms of type {target_element} found in structure.")

    num_to_replace = int(len(target_indices) * fraction)
    if num_to_replace < min_atoms_to_replace:
        if verbose:
            print(
                f"Doping fraction too small for {len(target_indices)} atoms of {target_element}. "
                f"Replacing {min_atoms_to_replace} atom(s) instead of {num_to_replace}."
            )
        num_to_replace = min_atoms_to_replace

    print(
        f"  Replaced {num_to_replace} out of {len(target_indices)} {target_element} atoms "
        f"({100 * num_to_replace / len(target_indices):.1f}%) with {dopant_element}"
    )

    indices_to_replace = random.sample(target_indices, num_to_replace)

    for i in indices_to_replace:
        new_structure[i] = Element(dopant_element)

    return new_structure


def compare_xrd_patterns(original, doped, label_doped, output_dir):
    plt.figure(figsize=(12, 8))
    plt.plot(original.x, original.y, label="Original", color="blue")
    plt.plot(
        doped.x,
        doped.y,
        label=f"Doped with {label_doped}",
        color="green",
        linestyle="--",
    )
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"XRD Comparison: Original vs {label_doped}-Doped")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"xrd_comparison_{label_doped}.png")
    plt.show()


def export_structure(structure, label: str, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    base_path = output_dir / f"doped_{label}"

    structure.to(filename=f"{base_path}.cif", fmt="cif")

    structure.to(filename=f"{base_path}.json", fmt="json")

    structure.to(filename=f"{base_path}.vasp", fmt="poscar")

    print(f"\nExported doped structure ({label}) to:")
    print(f"  → {base_path}.cif")
    print(f"  → {base_path}.json")
    print(f"  → {base_path}.vasp")


def main():
    cif_path = CONFIG["data_dir"] / CONFIG["cif_file"]
    output_dir = CONFIG["data_dir"] / "exports"
    summary_data = []

    try:
        structure = load_structure(cif_path)
        print(f"\nOriginal Structure: {structure.composition.formula}")
        original_density = calculate_theoretical_density(structure)
        print(f"  Density = {original_density:.4f} g/cm³")
        original_pattern = calculate_xrd_pattern(structure, CONFIG["wavelength"])

        for dopant in CONFIG["dopants"]:
            for fraction in [0.05, 0.1, 0.2]:
                print(
                    f"\n--- Simulating Doping with {dopant} at {fraction*100:.0f}% ---"
                )
                doped_structure = simulate_doping(
                    structure,
                    CONFIG["target_element"],
                    dopant,
                    fraction,
                    min_atoms_to_replace=1,
                    verbose=True,
                )

                doped_density = calculate_theoretical_density(doped_structure)
                print(f"  Doped Density = {doped_density:.4f} g/cm³")

                doped_pattern = calculate_xrd_pattern(
                    doped_structure, CONFIG["wavelength"]
                )
                label = f"{dopant}_{int(fraction*100)}pct"

                compare_xrd_patterns(original_pattern, doped_pattern, label, output_dir)

                print(f"\nCoordination Analysis for {label}-Doped Structure:")

                oxidized_doped_structure = doped_structure.copy()
                oxidized_doped_structure.add_oxidation_state_by_guess()
                analyze_coordination(oxidized_doped_structure, dopant)

                export_structure(doped_structure, label, output_dir)

                summary_data.append(
                    {
                        "Dopant": dopant,
                        "Fraction (%)": int(fraction * 100),
                        "Density (g/cm³)": round(doped_density, 4),
                        "Formula": doped_structure.composition.reduced_formula,
                    }
                )

        print("\n Doping Summary Table:")
        print(
            f"{'Dopant':<10} {'Fraction (%)':<13} {'Density (g/cm³)':<17} {'Formula':<20}"
        )
        print("-" * 60)
        for entry in summary_data:
            print(
                f"{entry['Dopant']:<10} {entry['Fraction (%)']:<13} {entry['Density (g/cm³)']:<17} {entry['Formula']:<20}"
            )

        csv_path = output_dir / "doping_summary.csv"
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\nSummary table saved to: {csv_path}")

        json_path = output_dir / "doping_summary.json"
        with open(json_path, mode="w") as file:
            json.dump(summary_data, file, indent=4)
        print(f"Summary data saved to: {json_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
