# SoD Simulation – ERP Role Conflicts & Login Behavior

So around 3 am, I randomly woke up with my brain already spiraling about this audit interview.
No coffee, no warning, just me, wide-eyed and haunted by role conflict matrices.
This script basically revealed itself to me in a moment of delirious clarity.

It's a homemade simulation of segregation of duties (SoD) risks inside an ERP setup.
I wanted something that looked smart, sounded audit-y, and maybe made me feel less doomed heading into the interview.

## What It Does

- Generates 100 fake users, each with 1–3 randomly assigned roles
- Checks for role combos that scream "internal control violation"
- Simulates login events with fake timestamps, IP addresses, and city names
- Outputs everything into CSV files so it looks like I know what I'm doing

## Files

- `sod_simulation.py` – the script that appeared to me in a vision
- `user_roles.csv` – user-to-role assignments
- `segregation_of_duties_risks.csv` – users with role conflicts
- `user_login_events.csv` – login logs pulled straight from my imagination

## Why It Exists

Mostly fear. But also curiosity. I wanted to make something audit-relevant, quick to build, and realistic enough to talk about.
You can take this and plug it into a dashboard, run visualizations, or use it to spark bigger ideas around internal control testing.

## What’s Next?

If I survive the interview, maybe I'll:
- Wrap this in Streamlit and call it a "risk monitoring tool"
- Add user behavior logic and trigger alerts
- Talk like I built a compliance platform instead of a 3am panic project

Use it, tweak it, laugh at it, at least it got me thinking under pressure.
