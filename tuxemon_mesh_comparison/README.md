# STL & Sprite – Tuxemon Geometry

So I ran another script. Not for a job this time, but for a question. A question tucked inside a mesh file, wrapped in a PNG, whispered by a creature that doesn't exist, except it does. In pixels. In triangles. In the quiet corners of a folder named after a feline pun.

I promised myself I'd analyze it properly. Not just admire the STL in a viewer, but ask it questions. What volume does it occupy in imaginary space? What symmetry does it hold in its paws? What color does it wear when facing forward, and what secrets does it hide from behind?

I started with the sprites. Front and back. They come from the **Tuxemon project**, a community-driven, open-source monster-catching game. The kind of project where creativity is currency and attribution is sacred. I downloaded the mesh from [Printables](https://www.printables.com/model/1352910-tuxemon-set-cute-creatures-for-a-monster-catching), where someone had used Maker World's “Image to 3D Model” tool to breathe volume into flat art. I didn't expect much. But the STL had other plans.

There was a choice: let the mesh sit quietly in a folder, or ask it questions. What bounding box does it stretch across? What surface area does it flaunt? Is it symmetric, or does it lean like a cat caught mid-pounce?

I chose to listen.

---

## What It Does

- Loads STL files of Tuxemon creatures and extracts geometric stats  
- Analyzes front and back PNG sprites to determine dominant color, contour area, and aspect ratio  
- Compares two meshes side-by-side, highlighting differences in volume, surface area, and bounding box  
- Outputs structured TXT files with clean formatting for further inspection or archival  
- Designed to be modular, readable, and extendable for batch analysis or integration into larger pipelines  

---

## Files of Interest

- `mesh_analysis.py` – the script that started with a sprite and ended with a spatial profile  
- `rockitten.stl`, `miaownolith.stl` – STL files of two Tuxemon creatures  
- `rockitten-front.png`, `rockitten-back.png` – sprite views from the Tuxemon project  
- `rockitten_stats.txt`, `miaownolith_stats.txt` – geometric summaries  
- `rockitten_views.txt`, `miaownolith_views.txt` – visual summaries from sprite analysis  
- `stl_comparison.txt` – side-by-side comparison of mesh metrics  

---

## Data Source

This project uses assets from:

**Tuxemon Project**  
Sprites: Openly licensed artworks contributed by the community  
Mesh: [Tuxemon Set – Cute Creatures for a Monster Catching Game](https://www.printables.com/model/1352910-tuxemon-set-cute-creatures-for-a-monster-catching)  
Converted using Maker World's “Image to 3D Model” tool

**Design Credits**  
- **Cairfrey**: Spalding004 (sprites), reimagined by TheOwl724  
- **Rockitten**: ShadowApex  
- **Nuenflu**: princess-phoenix  
- **Octabode**: fauxlens  
- **Miaownolith**: Sanglorian, reimagined by MTC-Studio  
- **Skwib**: fauxlens  
- **Budaye**: Leo, reimagined by ReallyDarkandWindy  
- **Tumblequill**: Serpexnessie (design), giovani_rubio (art)

**License**  
Creative Commons Attribution-ShareAlike 4.0 International  
✔ Remix Culture allowed  
✔ Commercial Use  
✔ Free Cultural Works  
✔ Meets Open Definition  
✖ Sharing without attribution not permitted

---

## Requirements

- Python 3.8+  
- `numpy`, `opencv-python`, `numpy-stl`  
- Optional: `matplotlib` if you want to visualize contours or meshes  

Install dependencies with:

```bash
pip install numpy opencv-python numpy-stl
```

## Actions Available

Run the script to:

- Analyze STL geometry: volume, surface area, bounding box, symmetry  
- Analyze PNG sprites: dominant color, contour area, aspect ratio  
- Compare two meshes and export the results to TXT  
- Generate clean, readable summaries for each creature  

---

## Why It Exists

Because it's the same doubt I used to ask myself during a monographic course on Aesthetics at university. Someone, whose name I've forgotten, but whose words stayed, spoke of liberating the form from a block of marble. It touched me. A reason why I'm still thinking about it, like some Ennio Morricone soundtrack that lives rent-free in my mind, looping quietly behind every moment of creative friction.

Sprites are more than just pixels. And meshes are more than just triangles. They're expressions, of design, of imagination, of community. This project is part geometry, part homage, part attempt to understand how fictional creatures occupy space when given form. It's not just technical curiosity, it's a kind of reverence.

It's a quiet celebration of open-source art, of tools that turn 2D into 3D, and of the people who make monsters not to scare, but to share. It's also a meditation on structure, on how even imaginary beings obey the laws of volume and symmetry, like dancers in a digital ballet. Their poses, their proportions, their bounding boxes, they all speak a language older than code.

To analyze a mesh is to ask: What does this creature weigh in the world of numbers? What does it look like from behind? What secrets does its silhouette hold? Is there a way to generate STL meshes automatically from two PNGs, even if not perfectly aligned? What's the pattern? What's the pattern?

The solution, the code, the grammar of form, it's written in the book of nature. Galileo said that. Or maybe someone quoting him. Either way, it needs decoding. And maybe this project is a small attempt to do just that.

---

## What's Next?

If curiosity keeps winning:

- Extend to batch analysis of all Tuxemon meshes  
- Visualize sprite contours and mesh overlays in a Streamlit dashboard  
- Export mesh metrics to CSV for clustering or classification  
- Explore symmetry detection across the entire set  

In the meantime, run the script, admire the bounding box, and maybe compare a few creatures.
