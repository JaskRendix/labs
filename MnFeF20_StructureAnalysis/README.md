# XRD & Coordination – Crystallographic Curiosity

So I sent another application, an interesting opening, tucked inside a job description that mentioned crystallography. While writing my email, I promised them (and myself) that I'd dive into the topic the moment I hit send. And I did.

I started with research. CIF files, they said. That's where the journey begins. The crystallographic fingerprint, the atomic blueprint. So I went looking.

No idea where to start. Too many choices. Summer of 69? I typed in "1987" - why not? Scrolled a bit. Saw Fe. Familiar. Suddenly I was back in chemistry class, staring at that massive periodic table on the wall, the one I used to get lost in. So yeah, let's pick this one. It felt like the wand at Ollivander's, the compound chose me.

I downloaded it, intending only to test a function. But the lattice had other plans.

There was a choice: let the structure sit quietly in a folder, or ask it questions. What angles do its atoms prefer? What peaks would it scatter in an X-ray dance? What density does it whisper in Å³? What would Quetzalcoatl have thought, staring into this symmetry?

I chose to listen.

But listen to... what? Miller! Miller indices. Memory. Though it also reminded me of that guy from The Expanse, Miller, right? The detective with the hat, the one who decided to wash his hair in that woman's apartment. Back when the show was still on Syfy. I'm not even sure why that stuck with me. No time to check. I'm sharing the pattern.

---

## What It Does

- Loads a CIF file from the [Crystallography Open Database](https://www.crystallography.net/cod/1000227.html)  
- Simulates an X-ray diffraction (XRD) pattern using `pymatgen`'s `XRDCalculator`  
- Annotates the top 10 intensity peaks with their corresponding Miller indices  
- Calculates theoretical density based on unit cell volume and formula weight  
- Analyzes local coordination environments for selected elements using `CrystalNN`  
- Outputs a POSCAR-style structure file in JSON format for further use  

---

## Files of Interest

- `xrd_analysis.py` – the script that started with a CIF and ended with a full crystallographic profile  
- `1000227.cif` – the CIF file for MnFeF₅·2H₂O, downloaded from the COD  
- `POSCAR.json` – exported structure file for use in other simulations or visualization tools  

---

## Data Source

This project uses crystallographic data from:

**Crystallography Open Database (COD)**  
Entry: [1000227](https://www.crystallography.net/cod/1000227.html)  
Compound: Manganese iron pentafluoride bis(hydrate)  
Formula: F₅FeH₄MnO₂  
Space Group: Imma (No. 74)  
Licensed under **CC0 1.0 Universal**, free for research and analysis

---

## Requirements

- Python 3.8+  
- `pymatgen`, `matplotlib`, `pathlib`  
- Optional: `pandas` if you plan to export XRD data to CSV  

Install dependencies with:

```bash
pip install pymatgen matplotlib
```

## Actions Available

Run the script to:

- Simulate and plot the XRD pattern with annotated peaks  
- Print lattice parameters, atomic sites, and chemical formula  
- Analyze coordination environments for Mn and Fe atoms  
- Calculate theoretical density in g/cm³  
- Save the structure as a JSON-formatted POSCAR file  

---

## Why It Exists

Because CIF files are more than just coordinates. They hold stories, of symmetry, of scattering, of atomic intimacy. That intimacy is a quiet elegance of repeating patterns, like Olympic ice skating performed in perfect synchronicity. Gold medal moments, choreographed not by coaches, but by Obi-Wan Kenobi singing his song, guiding two athletes toward the most beautiful and emotional performance in history.

This project is part crystallographic curiosity, part computational ritual, and part attempt to understand how atoms arrange themselves when no one's watching. That's the beauty of spontaneous order, like the blossoming of a flower. No audience, no applause, no rules. Just emergence. It reminds me I still need to read that book on Aztec metaphysics.

To understand the invisible is to read a book authored by nature itself. Ironically, it echoes the ambitions of 17th-century philosophers, those who believed the universe could be decoded, one principle at a time. But here, the decoding happens in Ångströms, in peaks and planes, in the silent language of the lattice.

---

## What's Next?

If curiosity keeps winning:

- Explore hidden symmetries in unlooked spots, those quiet corners of the atomic world where something revolutionary might be waiting, like Newton's forge still burning in the basement of time.
- Add support for batch CIF analysis across multiple compounds, because sometimes the real patterns only emerge when you zoom out and let the structures speak in chorus.
- Export XRD patterns to CSV for machine learning applications. Imagine teaching a machine to read diffraction, like a chimpanzee learning to recognize colors, surprised by the shimmer of order in the noise.
- Build a Streamlit dashboard to visualize coordination shells interactively, peeling back layers like an archaeologist of geometry.
- Extend to neutron diffraction or PDF analysis for amorphous systems, where the rules dissolve and the atoms improvise.

In the meantime, clone it, run it, rotate the unit cell, and maybe annotate a few peaks while sipping something crystalline. Not gin. Not air. A piña colada with Robert Holmes, somewhere between a Victorian murder mystery and a crystallographic fever dream.
