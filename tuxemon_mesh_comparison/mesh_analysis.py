from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from stl.mesh import Mesh


def export_to_txt(data: dict[str, Any], output_path: Path) -> None:
    def format_dict(d: dict[str, Any], indent: int = 0) -> str:
        lines = []
        for key, value in d.items():
            prefix = " " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(format_dict(value, indent + 2).splitlines())
            elif isinstance(value, np.ndarray):
                lines.append(f"{prefix}{key}: {np.round(value, 2).tolist()}")
            else:
                lines.append(
                    f"{prefix}{key}: {round(value, 2) if isinstance(value, float) else value}"
                )
        return "\n".join(lines)

    with output_path.open("w") as f:
        f.write(format_dict(data))


def get_mesh_stats(model: Mesh) -> dict[str, Any]:
    volume, center_of_mass, _ = model.get_mass_properties()
    surface_area = np.sum(model.areas)
    bounds = {
        "x": (model.x.min(), model.x.max()),
        "y": (model.y.min(), model.y.max()),
        "z": (model.z.min(), model.z.max()),
    }
    is_symmetric_x = np.allclose(model.x, model.x[::-1])

    return {
        "num_facets": len(model.vectors),
        "volume": volume,
        "surface_area": surface_area,
        "bounding_box": bounds,
        "center_of_mass": center_of_mass,
        "is_symmetric_x": is_symmetric_x,
    }


def analyze_stl_file(file_path: Path) -> dict[str, Any]:
    mesh_data = Mesh.from_file(file_path.as_posix())
    return get_mesh_stats(mesh_data)


def compare_stl_stats(
    stats_a: dict[str, Any], stats_b: dict[str, Any], name_a: str, name_b: str
) -> dict[str, dict[str, str]]:
    comparison: dict[str, dict[str, str]] = {}

    for key in stats_a:
        val_a = stats_a[key]
        val_b = stats_b[key]

        if isinstance(val_a, dict):
            comparison[key] = {
                axis: f"{val_a[axis]} vs {val_b[axis]}" for axis in val_a
            }
        elif isinstance(val_a, np.ndarray):
            comparison[key] = {
                "value": f"{np.round(val_a, 2).tolist()} vs {np.round(val_b, 2).tolist()}"
            }
        else:
            comparison[key] = {"value": f"{val_a} vs {val_b}"}

    return comparison


def analyze_single_png(image_path: Path) -> dict[str, Any]:
    image = cv2.imread(image_path.as_posix())
    if image is None:
        return {"error": f"Cannot read image: {image_path.name}"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    area = cv2.contourArea(contours[0]) if contours else 0
    aspect_ratio = round(width / height, 2)
    pixels = image.reshape(-1, 3)
    most_common = Counter(map(tuple, pixels)).most_common(1)[0][0]
    dominant_color = {"r": most_common[2], "g": most_common[1], "b": most_common[0]}

    return {
        "image_size": {"width": width, "height": height},
        "area_px": round(area, 2),
        "aspect_ratio": aspect_ratio,
        "dominant_color": dominant_color,
        "num_contours": len(contours),
    }


def analyze_views(front_path: Path, back_path: Path) -> dict[str, Any]:
    return {
        "front_view": analyze_single_png(front_path),
        "back_view": analyze_single_png(back_path),
    }


def main():
    folder = Path(__file__).parent

    rockitten_stl = folder / "rockitten.stl"
    rockitten_front = folder / "rockitten-front.png"
    rockitten_back = folder / "rockitten-back.png"

    miaownolith_stl = folder / "miaownolith.stl"
    miaownolith_front = folder / "miaownolith-front.png"
    miaownolith_back = folder / "miaownolith-back.png"

    if rockitten_stl.exists():
        stats_rockitten = analyze_stl_file(rockitten_stl)
        export_to_txt(stats_rockitten, folder / "rockitten_stats.txt")

        if rockitten_front.exists() and rockitten_back.exists():
            views_rockitten = analyze_views(rockitten_front, rockitten_back)
            export_to_txt(views_rockitten, folder / "rockitten_views.txt")

        if miaownolith_stl.exists():
            stats_miaownolith = analyze_stl_file(miaownolith_stl)
            export_to_txt(stats_miaownolith, folder / "miaownolith_stats.txt")

            comparison = compare_stl_stats(
                stats_rockitten, stats_miaownolith, "rockitten", "miaownolith"
            )
            export_to_txt(comparison, folder / "stl_comparison.txt")

            if miaownolith_front.exists() and miaownolith_back.exists():
                views_miaownolith = analyze_views(miaownolith_front, miaownolith_back)
                export_to_txt(views_miaownolith, folder / "miaownolith_views.txt")


if __name__ == "__main__":
    main()
