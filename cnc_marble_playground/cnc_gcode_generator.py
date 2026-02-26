import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

SAFE_Z = 5.0  # mm above workpiece
FEED_DEFAULT = 300  # mm/min


def multi_pass(depth, stepdown):
    passes = []
    current = -stepdown
    while abs(current) < depth:
        passes.append(current)
        current -= stepdown
    passes.append(-depth)
    return passes


def move_to_safe_z(gcode):
    gcode.append(f"G1 Z{SAFE_Z}")


def start_program():
    gcode = [
        "G90",  # Absolute positioning
        "G21",  # Units in mm
        "G17",  # XY plane selection
        "G94",  # Feed per minute
        "M3 S1000",  # Spindle on
    ]
    return gcode


def end_program(gcode):
    move_to_safe_z(gcode)
    gcode.append("M5")
    gcode.append("G0 X0 Y0")
    gcode.append("M30")
    return gcode


def generate_rectangle_outline(x, y, width, height, z, feed_rate):
    print(f"[RECT OUTLINE] origin=({x},{y}) size=({width}x{height}) depth={z}")

    g = []
    g.append(f"G0 X{x} Y{y}")
    g.append(f"G1 Z{z} F{feed_rate}")
    g.append(f"G1 X{x + width} Y{y} F{feed_rate}")
    g.append(f"G1 X{x + width} Y{y + height}")
    g.append(f"G1 X{x} Y{y + height}")
    g.append(f"G1 X{x} Y{y}")
    move_to_safe_z(g)

    print("[RECT OUTLINE] done.")
    return g


def generate_circle_outline(cx, cy, radius, z, feed_rate):
    print(f"[CIRCLE OUTLINE] center=({cx},{cy}) radius={radius} depth={z}")

    g = []
    start_x = cx + radius
    start_y = cy
    g.append(f"G0 X{start_x} Y{start_y}")
    g.append(f"G1 Z{z} F{feed_rate}")
    g.append(f"G2 X{start_x} Y{start_y} I{-radius} J0 F{feed_rate}")
    move_to_safe_z(g)

    print("[CIRCLE OUTLINE] done.")
    return g


def generate_polygon_outline(cx, cy, radius, sides, z, feed_rate):
    print(
        f"[POLYGON OUTLINE] center=({cx},{cy}) radius={radius} sides={sides} depth={z}"
    )

    assert sides >= 3
    points = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))

    print(f"[POLYGON OUTLINE] first point: {points[0]}")

    g = []
    x0, y0 = points[0]
    g.append(f"G0 X{x0} Y{y0}")
    g.append(f"G1 Z{z} F{feed_rate}")

    for idx, (x, y) in enumerate(points[1:], start=2):
        g.append(f"G1 X{x} Y{y} F{feed_rate}")
        if idx == sides // 2:
            print(f"[POLYGON OUTLINE] mid point: ({x:.2f}, {y:.2f})")

    g.append(f"G1 X{x0} Y{y0}")
    move_to_safe_z(g)

    print("[POLYGON OUTLINE] done.")
    return g


def generate_rectangle_pocket(
    x, y, width, height, depth, stepdown, stepover, feed_rate
):
    print(f"[RECT POCKET] origin=({x},{y}) size=({width}x{height}) depth={depth}")
    print(f"[RECT POCKET] stepdown={stepdown}, stepover={stepover}")

    g = []
    z_passes = multi_pass(depth, stepdown)
    print(f"[RECT POCKET] z-passes: {z_passes}")

    for pass_i, z in enumerate(z_passes):
        print(f"[RECT POCKET] pass {pass_i+1}/{len(z_passes)} at z={z:.3f}")

        current_y = y
        end_y = y + height
        direction = 1

        while current_y <= end_y + 1e-6:
            start_x = x if direction == 1 else x + width
            end_x = x + width if direction == 1 else x

            print(f"  row y={current_y:.2f} from x={start_x:.2f} to x={end_x:.2f}")

            g.append(f"G0 X{start_x} Y{current_y}")
            g.append(f"G1 Z{z} F{feed_rate}")
            g.append(f"G1 X{end_x} Y{current_y} F{feed_rate}")
            move_to_safe_z(g)

            current_y += stepover
            direction *= -1

    print("[RECT POCKET] done.")
    return g


def generate_relief_from_heightmap(
    image_path, origin_x, origin_y, size_x, size_y, max_depth, feed_rate
):
    img = Image.open(image_path).convert("L")
    width, height = img.size
    pixels = img.load()

    g = []

    dx = size_x / (width - 1)
    dy = size_y / (height - 1)

    for j in range(height):
        y = origin_y + j * dy
        direction = 1 if j % 2 == 0 else -1
        x_range = range(width) if direction == 1 else range(width - 1, -1, -1)

        first_point = True
        for i in x_range:
            x = origin_x + i * dx
            brightness = pixels[i, j]
            depth = (255 - brightness) / 255 * max_depth
            z = -depth

            if first_point:
                g.append(f"G0 X{x:.3f} Y{y:.3f}")
                g.append(f"G1 Z{z:.3f} F{feed_rate}")
                first_point = False
            else:
                g.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed_rate}")

        move_to_safe_z(g)

    return g


def generate_rosette(cx, cy, base_radius, depth, stepdown, feed_rate):
    g = []
    z_passes = multi_pass(depth, stepdown)

    for z in z_passes:
        g += generate_circle_outline(cx, cy, base_radius, z, feed_rate)
        g += generate_polygon_outline(cx, cy, base_radius * 0.8, 12, z, feed_rate)
        g += generate_circle_outline(cx, cy, base_radius * 0.5, z, feed_rate)
        sides = random.choice([5, 8])
        g += generate_polygon_outline(cx, cy, base_radius * 0.3, sides, z, feed_rate)

    g += generate_rectangle_pocket(
        cx - base_radius * 0.1,
        cy - base_radius * 0.1,
        base_radius * 0.2,
        base_radius * 0.2,
        depth * 0.5,
        stepdown * 0.5,
        stepover=base_radius * 0.05,
        feed_rate=feed_rate,
    )
    return g


def generate_hexagon(cx, cy, radius, z, feed_rate):
    return generate_polygon_outline(cx, cy, radius, 6, z, feed_rate)


def generate_hex_tessellation(
    x_min, x_max, y_min, y_max, radius, depth, stepdown, feed_rate
):
    g = []
    z_passes = multi_pass(depth, stepdown)

    hex_height = math.sqrt(3) * radius
    hex_width = 2 * radius
    row = 0
    y = y_min

    while y <= y_max + hex_height:
        offset = 0 if row % 2 == 0 else (hex_width * 0.75)
        x = x_min + offset
        while x <= x_max + hex_width:
            for z in z_passes:
                g += generate_hexagon(x, y, radius, z, feed_rate)
            x += hex_width * 1.5
        y += hex_height
        row += 1

    return g


def generate_nasrid_12point_rosette(
    origin_x, origin_y, size_x, size_y, max_depth, feed_rate
):
    g = []
    width = 300
    height = 300

    dx = size_x / (width - 1)
    dy = size_y / (height - 1)

    star_points = 12

    for j in range(height):
        y = origin_y + j * dy
        direction = 1 if j % 2 == 0 else -1
        x_range = range(width) if direction == 1 else range(width - 1, -1, -1)

        for i in x_range:

            x = origin_x + i * dx

            cx = width / 2
            cy = height / 2
            nx = (i - cx) / cx
            ny = (j - cy) / cy
            r = math.sqrt(nx * nx + ny * ny)
            r = min(1, r)
            theta = math.atan2(ny, nx)

            # --- Nasrid star geometry ---
            star = abs(math.cos(theta * star_points))
            petals = 0.5 + 0.5 * math.sin(theta * star_points * 0.5)
            band = 1 - min(1, r * 1.2)
            depth_factor = max(0, band * (0.6 * star + 0.4 * petals))
            dome = max(0, 1 - r * r * 1.5)
            depth = (0.7 * depth_factor + 0.3 * dome) * max_depth
            z = -depth

            if j % 50 == 0 and i in (0, width // 2, width - 1):
                print(f"[Row {j}, Col {i}] r={r:.3f}, depth={depth:.3f}")

            if i == x_range.start:
                g.append(f"G0 X{x:.3f} Y{y:.3f}")
                g.append(f"G1 Z{z:.3f} F{feed_rate}")
            else:
                g.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed_rate}")

        move_to_safe_z(g)

    return g


def parse_gcode(gcode_file):
    x = y = z = 0.0
    paths = []
    current_path = []

    with open(gcode_file, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith(("G0", "G1", "G2", "G3")):
                parts = line.split()
                for p in parts:
                    if p.startswith("X"):
                        x = float(p[1:])
                    elif p.startswith("Y"):
                        y = float(p[1:])
                    elif p.startswith("Z"):
                        z = float(p[1:])

                current_path.append((x, y, z))

            elif line.startswith("M30"):
                if current_path:
                    paths.append(current_path)
                    current_path = []

    if current_path:
        paths.append(current_path)

    return paths


def plot_paths(paths):
    plt.figure(figsize=(6, 6))
    for path in paths:
        xs, ys, colors = [], [], []
        for x, y, cutting in path:
            xs.append(x)
            ys.append(y)
            colors.append("red" if cutting else "blue")

        plt.scatter(xs, ys, c=colors, s=10)
        plt.plot(xs, ys, linestyle="--", alpha=0.4, color="gray")

    plt.title("CNC Toolpath Visualization")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_heightmap(paths):
    xs, ys, zs = [], [], []

    for path in paths:
        for x, y, z in path:
            xs.append(x)
            ys.append(y)
            zs.append(z)

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c=zs, cmap="viridis", s=5)
    plt.colorbar(label="Z depth (mm)")
    plt.title("Relief Heightmap Visualization")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def main():
    base_path = Path(__file__).parent
    output_path = base_path / "output.gcode"

    gcode = start_program()

    job_type = random.choice(
        [
            "relief",
            "rosette",
            "tessellation",
            "rect_outline",
            "circle_outline",
            "polygon_outline",
            "rect_pocket",
        ]
    )
    print(f"\n=== Marble job type selected: {job_type.upper()} ===")

    max_depth = random.uniform(2.0, 5.0)
    stepdown = min(random.uniform(0.3, 1.0), max_depth / 2)
    feed_rate = random.choice([150, 200, 250, 300])

    if job_type == "relief":
        image_path = base_path / "relief.png"

        if image_path.exists():
            print(f"Using heightmap: {image_path.name}")
            gcode += generate_relief_from_heightmap(
                image_path=image_path,
                origin_x=0,
                origin_y=0,
                size_x=80,
                size_y=80,
                max_depth=max_depth,
                feed_rate=feed_rate,
            )
        else:
            print("No relief.png found.")
            gcode += generate_nasrid_12point_rosette(
                origin_x=0,
                origin_y=0,
                size_x=80,
                size_y=80,
                max_depth=max_depth,
                feed_rate=feed_rate,
            )

    elif job_type == "rosette":
        cx, cy = 50, 50
        base_radius = random.uniform(20, 40)
        print(f"Rosette center: ({cx}, {cy}), base radius: {base_radius:.1f}")
        print(
            f"Depth: {max_depth:.2f} mm, stepdown: {stepdown:.2f} mm, feed: {feed_rate} mm/min"
        )
        gcode += generate_rosette(
            cx=cx,
            cy=cy,
            base_radius=base_radius,
            depth=max_depth,
            stepdown=stepdown,
            feed_rate=feed_rate,
        )

    elif job_type == "tessellation":
        print("Generating hexagonal marble floor tessellation.")
        print(
            f"Depth: {max_depth:.2f} mm, stepdown: {stepdown:.2f} mm, feed: {feed_rate} mm/min"
        )
        gcode += generate_hex_tessellation(
            x_min=0,
            x_max=100,
            y_min=0,
            y_max=100,
            radius=random.uniform(5, 12),
            depth=max_depth,
            stepdown=stepdown,
            feed_rate=feed_rate,
        )

    elif job_type == "rect_outline":
        print("Generating rectangle outline.")
        gcode += generate_rectangle_outline(
            x=10,
            y=10,
            width=60,
            height=40,
            z=-max_depth,
            feed_rate=feed_rate,
        )

    elif job_type == "circle_outline":
        print("Generating circle outline.")
        gcode += generate_circle_outline(
            cx=50,
            cy=50,
            radius=random.uniform(10, 40),
            z=-max_depth,
            feed_rate=feed_rate,
        )

    elif job_type == "polygon_outline":
        print("Generating polygon outline.")
        gcode += generate_polygon_outline(
            cx=50,
            cy=50,
            radius=random.uniform(10, 40),
            sides=random.choice([3, 4, 5, 6, 8, 12]),
            z=-max_depth,
            feed_rate=feed_rate,
        )

    elif job_type == "rect_pocket":
        print("Generating rectangle pocket.")
        gcode += generate_rectangle_pocket(
            x=10,
            y=10,
            width=60,
            height=40,
            depth=max_depth,
            stepdown=stepdown,
            stepover=2,
            feed_rate=feed_rate,
        )

    gcode = end_program(gcode)

    with output_path.open("w") as f:
        f.write("\n".join(gcode))

    print(f"\nG-code written to: {output_path}")
    print("Visualizing toolpath...")

    paths = parse_gcode(output_path)
    plot_paths(paths)
    plot_heightmap(paths)


if __name__ == "__main__":
    main()
