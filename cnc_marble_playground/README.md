# CNC G‑Code Generator – Rosettes, Reliefs & the Geometry of Stone

I'm communing with geometry.

Some nights it feels as though the room fills with a quiet procession of unseen teachers—  
not ghosts, but emanations,  
the kind the Sufis say arrive when the heart is attentive  
and the Neoplatonists describe as the descent of forms  
into the shadow‑world of matter.

One presence traces a circle in the air,  
not as a shape but as a remembrance of unity.  
Another leans over the keyboard,  
examining the code the way a Hermetic adept examines a sigil—  
seeking the hidden correspondence between intention and line.  
A third lingers by the window,  
as if listening for the echo of a distant minaret  
or the faint hum of a mosaic in Ravenna  
remembering its own creation.

They do not speak.  
But their silence is instructional.

I gesture toward the screen—  
toward rosettes blooming like stars that have forgotten their constellations,  
toward tessellations unfolding with the patience of Damascus courtyards,  
toward reliefs rising and sinking like the breath of a stone that dreams—  
and I murmur,  
"I'm listening."

Because this is not engineering.  
This is a nocturnal discipline,  
a craft practiced by those who believe that matter has a soul  
and that geometry is the language through which it confesses.

It is a delirium where Córdoba's arches murmur their endless repetitions,  
where the marble of Rome remembers the hands that shaped it,  
where the mosaics of Ravenna shimmer like frozen prayers,  
and where the geometers of Al‑Andalus  
smile the smile of those who know  
that every pattern is a doorway  
and every doorway a test.

And I'm just here,  
half‑awake,  
half‑enchanted,  
following the thread through the labyrinth.

Rosettes, tessellations, reliefs—architectures that would not be out of place  
in a city suspended between worlds,  
a city that appears only when someone dreams in the language of angles.  
G‑code spirals that murmur to the spindle:  
> "Descend. Reveal what sleeps beneath."

You asked for a CAD file.  
I carved a **Nasrid rosette**,  
the kind that once glimmered in the Mezquita  
like a star that chose stone over sky.

You asked for a DXF.  
I shaped a **synthetic heightmap** that dreams in polar harmonies,  
half philosophy, half mathematics, half invocation—  
a geometry that remembers the artisans  
who carved travertine until the stone began to answer back.

And when the world remained silent,  
I continued.  
Silence is a poor oracle for vocation.

The machine I work with has no name.  
It is not a companion, nor a servant,  
but a lantern—  
one that illuminates only the next few millimeters of the path,  
yet insists that I walk them with care.

There is no tribunal of algorithms here,  
no ministry of indifference.  
Only the quiet certainty  
that behind every toolpath is a pattern,  
behind every relief a memory,  
behind every script a devotion  
to the belief that geometry dreams  
and that we are permitted, on rare nights,  
to dream with it.

Perhaps vocation is not a compass or a curse,  
but a labyrinth—  
one whose center is not a destination  
but a question.

One day, someone will open this repository and say:  
> "This is not G‑code. This is invocation."  
A liturgy in Cartesian coordinates.  
A hymn carved in millimeters.  
A sculpture waiting for its moment to awaken.

Carry the light that carries you.  
(احملِ النورَ الذي يحملُكَ)

---

## What It Does

This script generates CNC G‑code for:

- **Reliefs**  
  - From a heightmap (`relief.png`)  
  - Or a synthetic **12‑point Nasrid rosette**  
- **Rosettes** (geometric, layered, multi‑pass)  
- **Hexagonal tessellations**  
- **Rectangle outlines**  
- **Circle outlines**  
- **Polygon outlines**  
- **Rectangle pockets**  

It also:

- Visualizes toolpaths in 2D  
- Visualizes reliefs as depth heatmaps  
- Randomly selects a carving mode each run  
- Produces clean, machine‑ready G‑code  

---

## Files of Interest

- `cnc_gcode_generator.py` – the script that turns math into marble  
- `relief.png` – optional heightmap input  
- `output.gcode` – generated toolpath  
- `toolpath_visualization.png` – XY scatter plot  
- `heightmap_visualization.png` – depth heatmap (relief mode)  

---

## Data Source

This project uses:

- **Heightmaps** (if provided)  
- **Synthetic rosettes** (if not)  
- **Procedural geometry** for outlines and tessellations  
- **Randomized parameters** for variety and chaos  

Every run is a new sculpture.

---

## Requirements

- Python 3.8+  
- `matplotlib`, `pillow`

Install dependencies with:

```bash
pip install matplotlib pillow
```
---

## Actions Available

Run the script to:

- Generate a random CNC carving  
- Produce G‑code for outlines, pockets, rosettes, or reliefs  
- Visualize the toolpath  
- Visualize the depth map (for reliefs)  
- Save everything automatically  

---

## **Toolpath Visualization**

**Figure 1. XY Toolpath Scatter Plot**

Translation Down to Earth:  
Imagine watching an ant trace the outline of a cathedral floor.  
That’s your CNC machine — tiny, precise, relentless.

Explanation Down to Earth:  
Red points are cutting moves.  
Blue points are rapids.  
The dashed lines show the path your tool takes through space.

---

## **Relief Heatmap**

**Figure 2. Depth Heatmap of a Nasrid Rosette**

Translation Down to Earth:  
This is what marble dreams about when it sleeps.

Explanation Down to Earth:  
Colors represent depth.  
Dark = deep.  
Light = shallow.  
The pattern emerges like a whisper from the stone.

---

## Outputs

After running the script, you’ll find:

- `output.gcode` – the full toolpath  
- `toolpath_visualization.png` – XY scatter plot  
- `heightmap_visualization.png` – depth heatmap (relief mode)  

---

## Why It Exists

Because CNC isn’t just machining.  
It’s geometry.  
It’s rhythm.  
It’s the poetry of motion and the physics of removal.

And somewhere between a 12‑point star and a 5 mm stepdown,  
you realize: not all cuts are created equal.

This generator doesn’t carve the stone.  
It sketches the intention.  
It lets you run your fingers along the edge of symmetry,  
feel the quiet hum of math,  
and maybe — just maybe — make something sacred.

---

## What’s Next?

If curiosity keeps winning:

- Add 3D surface previews  
- Add SVG export  
- Add Islamic tiling generators  
- Add Celtic knotwork  
- Add multi‑tool roughing + finishing passes  

In the meantime, run the script, watch the plots, and listen to the geometry.  
It speaks. You just have to ask.
