# F1 Lap Time Simulator

Point-mass F1 lap time model with tyre degradation, fuel burn-off, DRS zones and pit-stop strategy analysis. Built around a Silverstone track model after a placement with Williams in motorsport.

![Stint analysis](stint_analysis.png)

## What it does

| Feature | Detail |
|---|---|
| **Car model** | Power-limited acceleration, aero drag and downforce, brake performance, DRS drag reduction |
| **Tyre model** | Compound-specific grip (Soft/Medium/Hard), thermal warmup, linear degradation with distance |
| **Track model** | Segment-based layout — straights with optional DRS zones and corners with radius and arc length |
| **Fuel effect** | Mass decreases over the stint, improving lap times — exactly as in a real race |
| **Strategy** | Compare 1-stop vs 2-stop over a stint length and see which comes out ahead |
| **Silverstone** | 17-segment model of the current Silverstone layout with real corner radii |
| **Plots** | Lap times, tyre grip, fuel load and top speed across a stint, split by compound |

## Example output

```
  Track    : Silverstone (5.36 km)
  Car      : F1 2024-spec
  Fuel     : 80 kg
  Stint    : 20 laps

  Soft      best 83.21s  worst 91.44s  degradation 8.23s  total 1731.2s
  Medium    best 84.89s  worst 88.16s  degradation 3.27s  total 1714.8s
  Hard      best 86.44s  worst 88.55s  degradation 2.11s  total 1741.3s

--- Strategy: 1-stop vs 2-stop (20 laps, Medium tyres) ---
  1-stop (12+8): 1732.4s
  2-stop (7+7+6): 1762.1s
  Difference: 29.7s favour 1-stop
```

## Usage

```bash
pip install -r requirements.txt
python lap_sim.py
```

## Built with

- Python 3.10+
- NumPy
- Matplotlib

## What I learned

How tyre degradation and fuel mass create a natural lap-time evolution curve through a stint. Why soft tyres aren't always faster over a race distance. How DRS works as a drag-reduction mechanism and where it matters most (Hangar Straight at Silverstone). The trade-off between fresh-tyre grip and pit-stop time loss that defines race strategy.
