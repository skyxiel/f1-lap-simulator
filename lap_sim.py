"""
F1 lap time simulator — point-mass model with tyre degradation,
fuel load, DRS, aero balance and sector-level strategy analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# ── Car specification ────────────────────────────────────────────────────────

@dataclass
class CarSpec:
    name: str
    mass_dry: float         # kg (without fuel)
    power: float            # W (peak)
    cd: float               # drag coefficient
    cl: float               # downforce coefficient (negative lift)
    frontal_area: float     # m^2
    drivetrain_eff: float   # powertrain efficiency
    brake_mu: float         # brake friction coefficient
    fuel_rate: float        # kg/s at full power
    drs_cd_reduction: float # Cd reduction when DRS is open

    def drag_force(self, v: float, rho: float = 1.225, drs: bool = False) -> float:
        cd = self.cd - self.drs_cd_reduction if drs else self.cd
        return 0.5 * rho * cd * self.frontal_area * v**2

    def downforce(self, v: float, rho: float = 1.225, drs: bool = False) -> float:
        cl = self.cl * 0.75 if drs else self.cl  # DRS reduces rear downforce
        return 0.5 * rho * cl * self.frontal_area * v**2

    def max_traction_force(self, v: float, mass: float, mu: float) -> float:
        normal = mass * 9.81 + self.downforce(v)
        return mu * normal

    def max_brake_force(self, v: float, mass: float) -> float:
        normal = mass * 9.81 + self.downforce(v)
        return self.brake_mu * normal

# ── Tyre model ───────────────────────────────────────────────────────────────

@dataclass
class TyreCompound:
    name: str
    mu_peak: float         # peak grip coefficient
    degradation_rate: float  # grip loss per km
    thermal_window: float  # laps to reach peak (warmup)

    def grip_at(self, distance_km: float, lap: int) -> float:
        warmup = min(1.0, lap / max(self.thermal_window, 0.5))
        wear = max(0.0, 1.0 - self.degradation_rate * distance_km)
        return self.mu_peak * warmup * wear

# ── Track definition ─────────────────────────────────────────────────────────

@dataclass
class Corner:
    name: str
    radius: float         # m
    length: float         # m (arc length through corner)
    drs_after: bool = False

@dataclass
class Straight:
    name: str
    length: float         # m
    drs_zone: bool = False

@dataclass
class Track:
    name: str
    segments: list
    pit_loss: float = 22.0  # seconds lost in pit stop

    @property
    def lap_distance(self) -> float:
        return sum(s.length for s in self.segments)

# ── Simulation ───────────────────────────────────────────────────────────────

@dataclass
class LapResult:
    lap_time: float
    sector_times: list
    top_speed: float
    fuel_used: float
    tyre_grip: float
    segment_speeds: list


@dataclass
class RaceResult:
    lap_times: list = field(default_factory=list)
    fuel_mass: list = field(default_factory=list)
    tyre_grip: list = field(default_factory=list)
    top_speeds: list = field(default_factory=list)
    total_time: float = 0.0


def simulate_segment(seg, car: CarSpec, mass: float, mu: float,
                     entry_speed: float, drs: bool = False) -> tuple:
    dt = 0.01
    v = entry_speed
    t = 0.0
    dist = 0.0
    max_speed = v
    fuel_used = 0.0

    if isinstance(seg, Corner):
        v_corner_max = np.sqrt(mu * (mass * 9.81 + car.downforce(v)) * seg.radius / mass)
        v = min(v, v_corner_max)

    while dist < seg.length:
        if isinstance(seg, Straight):
            # accelerate up to power limit
            f_drive = min(
                car.power * car.drivetrain_eff / max(v, 1.0),
                car.max_traction_force(v, mass, mu),
            )
            f_drag = car.drag_force(v, drs=drs)
            f_net = f_drive - f_drag
        else:
            # in corner: maintain max cornering speed
            v_max = np.sqrt(mu * (mass * 9.81 + car.downforce(v)) * seg.radius / mass)
            v = min(v, v_max)
            f_net = 0.0  # constant speed through corner (simplified)

        a = f_net / mass
        v += a * dt
        v = max(v, 5.0)  # minimum speed
        dist += v * dt
        t += dt
        fuel_used += car.fuel_rate * dt * (0.3 if isinstance(seg, Corner) else 1.0)
        max_speed = max(max_speed, v)

    return t, v, max_speed, fuel_used


def simulate_lap(track: Track, car: CarSpec, mass: float,
                 tyre: TyreCompound, distance_km: float, lap: int) -> LapResult:
    mu = tyre.grip_at(distance_km, lap)
    v = 60.0  # start/finish speed estimate
    total_time = 0.0
    total_fuel = 0.0
    top_speed = 0.0
    segment_speeds = []

    n_sectors = 3
    seg_per_sector = len(track.segments) // n_sectors
    sector_times = [0.0] * n_sectors

    for i, seg in enumerate(track.segments):
        drs = isinstance(seg, Straight) and seg.drs_zone
        t_seg, v, v_max, fuel = simulate_segment(seg, car, mass, mu, v, drs)
        total_time += t_seg
        total_fuel += fuel
        top_speed = max(top_speed, v_max)
        segment_speeds.append(v_max)

        sector_idx = min(i // max(seg_per_sector, 1), n_sectors - 1)
        sector_times[sector_idx] += t_seg

    return LapResult(total_time, sector_times, top_speed, total_fuel, mu, segment_speeds)


def simulate_stint(track: Track, car: CarSpec, tyre: TyreCompound,
                   n_laps: int, fuel_load: float) -> RaceResult:
    result = RaceResult()
    mass = car.mass_dry + fuel_load
    distance_km = 0.0

    for lap in range(1, n_laps + 1):
        lap_result = simulate_lap(track, car, mass, tyre, distance_km, lap)
        result.lap_times.append(lap_result.lap_time)
        result.fuel_mass.append(mass - car.mass_dry)
        result.tyre_grip.append(lap_result.tyre_grip)
        result.top_speeds.append(lap_result.top_speed)

        mass -= lap_result.fuel_used
        mass = max(mass, car.mass_dry)
        distance_km += track.lap_distance / 1000

    result.total_time = sum(result.lap_times)
    return result

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_stint_analysis(results: dict[str, RaceResult], track_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Stint Analysis — {track_name}", color="white", fontsize=15, fontweight="bold")

    colors = {"Soft": "#e74c3c", "Medium": "#f1c40f", "Hard": "#ecf0f1"}

    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.15, color="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    # lap times
    ax = axes[0, 0]
    for name, res in results.items():
        laps = range(1, len(res.lap_times) + 1)
        ax.plot(laps, res.lap_times, color=colors.get(name, "white"), linewidth=2, label=name, marker="o", markersize=3)
    ax.set_xlabel("Lap", color="white")
    ax.set_ylabel("Lap Time (s)", color="white")
    ax.set_title("Lap Times", color="white")
    ax.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")

    # tyre grip
    ax = axes[0, 1]
    for name, res in results.items():
        laps = range(1, len(res.tyre_grip) + 1)
        ax.plot(laps, [g * 100 for g in res.tyre_grip], color=colors.get(name, "white"), linewidth=2, label=name)
    ax.set_xlabel("Lap", color="white")
    ax.set_ylabel("Tyre Grip (%)", color="white")
    ax.set_title("Tyre Degradation", color="white")
    ax.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")

    # fuel mass
    ax = axes[1, 0]
    for name, res in results.items():
        laps = range(1, len(res.fuel_mass) + 1)
        ax.plot(laps, res.fuel_mass, color=colors.get(name, "white"), linewidth=2, label=name)
    ax.set_xlabel("Lap", color="white")
    ax.set_ylabel("Fuel Remaining (kg)", color="white")
    ax.set_title("Fuel Load", color="white")
    ax.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")

    # top speed
    ax = axes[1, 1]
    for name, res in results.items():
        laps = range(1, len(res.top_speeds) + 1)
        ax.plot(laps, [v * 3.6 for v in res.top_speeds], color=colors.get(name, "white"), linewidth=2, label=name)
    ax.set_xlabel("Lap", color="white")
    ax.set_ylabel("Top Speed (km/h)", color="white")
    ax.set_title("Top Speed per Lap", color="white")
    ax.legend(facecolor="#2a2a3e", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    return fig

# ── Presets ───────────────────────────────────────────────────────────────────

def f1_car_2024() -> CarSpec:
    return CarSpec(
        name="F1 2024-spec",
        mass_dry=798,
        power=735_000,       # ~1000 bhp
        cd=1.0,
        cl=3.5,
        frontal_area=1.5,
        drivetrain_eff=0.92,
        brake_mu=1.8,
        fuel_rate=0.033,     # ~2 kg/lap at Silverstone
        drs_cd_reduction=0.15,
    )


def silverstone() -> Track:
    return Track(
        name="Silverstone",
        segments=[
            Straight("Wellington Straight", 600, drs_zone=True),
            Corner("Brooklands", 60, 120),
            Corner("Luffield", 45, 150),
            Straight("National Pit Straight", 450),
            Corner("Copse", 180, 200),
            Straight("Maggotts approach", 300),
            Corner("Maggotts", 80, 100),
            Corner("Becketts", 65, 120),
            Corner("Chapel", 120, 150),
            Straight("Hangar Straight", 770, drs_zone=True),
            Corner("Stowe", 90, 130),
            Corner("Vale", 35, 80),
            Corner("Club", 55, 120),
            Straight("Farm Straight", 500),
            Corner("Abbey", 100, 150),
            Corner("Bridge", 45, 80),
            Straight("Start/Finish", 350),
        ],
        pit_loss=22.0,
    )


def tyre_compounds():
    return {
        "Soft": TyreCompound("Soft", mu_peak=1.65, degradation_rate=0.012, thermal_window=1.5),
        "Medium": TyreCompound("Medium", mu_peak=1.55, degradation_rate=0.006, thermal_window=2.5),
        "Hard": TyreCompound("Hard", mu_peak=1.45, degradation_rate=0.003, thermal_window=3.5),
    }

# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  F1 LAP TIME SIMULATOR")
    print("=" * 60)

    car = f1_car_2024()
    track = silverstone()
    tyres = tyre_compounds()
    n_laps = 20
    fuel_kg = 80.0

    print(f"\n  Track    : {track.name} ({track.lap_distance/1e3:.2f} km)")
    print(f"  Car      : {car.name}")
    print(f"  Fuel     : {fuel_kg} kg")
    print(f"  Stint    : {n_laps} laps\n")

    results = {}
    for name, tyre in tyres.items():
        res = simulate_stint(track, car, tyre, n_laps, fuel_kg)
        results[name] = res
        best = min(res.lap_times)
        worst = max(res.lap_times)
        print(f"  {name:8s}  best {best:.2f}s  worst {worst:.2f}s  "
              f"degradation {(worst-best):.2f}s  "
              f"total {res.total_time:.1f}s")

    # strategy comparison
    print(f"\n--- Strategy: 1-stop vs 2-stop ({n_laps} laps, Medium tyres) ---")
    med = tyres["Medium"]

    # 1-stop: 12 laps + 8 laps
    stint1a = simulate_stint(track, car, med, 12, fuel_kg)
    stint1b = simulate_stint(track, car, med, 8, fuel_kg * 8 / n_laps)
    one_stop = stint1a.total_time + stint1b.total_time + track.pit_loss
    print(f"  1-stop (12+8): {one_stop:.1f}s")

    # 2-stop: 7+7+6
    stint2a = simulate_stint(track, car, med, 7, fuel_kg)
    stint2b = simulate_stint(track, car, med, 7, fuel_kg * 7 / n_laps)
    stint2c = simulate_stint(track, car, med, 6, fuel_kg * 6 / n_laps)
    two_stop = stint2a.total_time + stint2b.total_time + stint2c.total_time + 2 * track.pit_loss
    print(f"  2-stop (7+7+6): {two_stop:.1f}s")
    print(f"  Difference: {abs(one_stop - two_stop):.1f}s favour {'1-stop' if one_stop < two_stop else '2-stop'}")

    print("\nGenerating plots...")
    fig = plot_stint_analysis(results, track.name)
    fig.savefig("stint_analysis.png", dpi=150, bbox_inches="tight")
    print("  → stint_analysis.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
