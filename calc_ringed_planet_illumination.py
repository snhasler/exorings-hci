"""
author: S. Hasler
Script to simulate and visualize the illumination of a ringed planet across orbital epochs.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Tuple
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

import geometry as g
import ray_casting as rc

# Set up simulation parameters below for the planet, ring, orbit, system, and 
# output directories
# ----------------------------------------------------------------------------
class SimulationParams:
    """
    Simulation parameters for planet, ring, orbit, and 
    output directories for writing files/plots.
    """

    def __init__(
        self,
        # Orbit parameters
        a_planet_au: float = 5.2,           # true planet separation (AU)
        e: float = 0.0,                     # eccentricity
        inc_deg: float = 60.0,              # orbit inclination (deg)
        Omega_deg: float = 0.0,             # longitude of ascending node (deg)
        omega_deg: float = 170.0,           # argument of periastron (deg)
        n_epochs: int = 18,                 # number of orbital positions to sample -- 250 generally good for phase curves
        # Planet & rings
        planet_radius_km: float = 69911.0,  # Jupiter radius [km]
        ring_inner: float = 1.43,           # inner ring radius [R_planet]
        ring_outer: float = 3.52,           # outer ring radius [R_planet]
        obliquity_deg: float = 90,        # ring obliquity (epsilon, deg) (for i=90°, face-on =)
        spin_longitude_deg: float = -90.0,    # planet spin longitude (phi_s, deg)
        # Observer
        distance_pc: float = 10.0,          # observer distance (pc)
        # Sampling resolution for planet and ring
        n_theta_planet: int = 260,
        n_phi_planet: int = 260,
        n_r_ring: int = 260,
        n_phi_ring: int = 360,
        # What to run
        plot: bool = False,                 # combined 3D plot of all epochs
        plot_closeup: bool = False,          # close-up view of ringed planet per epoch
        save_files: bool = False,           # write illumination-fraction .txt files
        save_figs: bool = False,             # save figures 
        # Output locations (only used if save_files / save_figs = True)
        output_dir: str = ("/Users/shasler/Documents/Research/Projects/exorings/fraction_lit_output/2xSat_faceon_ring/"),
        plot_dir: str = "/Users/shasler/Documents/Research/Projects/exorings/fraction_lit_output/2xSat_faceon_ring/plots/"
        ):
        
        self.a_planet_au = a_planet_au
        self.e = e
        self.inc_deg = inc_deg
        self.Omega_deg = Omega_deg
        self.omega_deg = omega_deg
        self.n_epochs = n_epochs
        self.planet_radius_km = planet_radius_km
        self.ring_inner = ring_inner
        self.ring_outer = ring_outer
        self.obliquity_deg = obliquity_deg
        self.spin_longitude_deg = spin_longitude_deg
        self.distance_pc = distance_pc
        self.n_theta_planet = n_theta_planet
        self.n_phi_planet = n_phi_planet
        self.n_r_ring = n_r_ring
        self.n_phi_ring = n_phi_ring
        self.plot = plot
        self.plot_closeup = plot_closeup
        self.save_files = save_files
        self.save_figs = save_figs

        self.output_dir = output_dir
        self.plot_dir = plot_dir

class DerivedGeometry:
    """Derived quantities, computed once from a config."""

    def __init__(self, e, inc, Omega, omega, obliquity, spin_longitude,
                 planet_radius, ring_inner, ring_outer, a, d):
        self.e = e
        self.inc = inc
        self.Omega = Omega
        self.omega = omega
        self.obliquity = obliquity
        self.spin_longitude = spin_longitude
        self.planet_radius = planet_radius
        self.ring_inner = ring_inner
        self.ring_outer = ring_outer
        self.a = a
        self.d = d

    @classmethod
    def from_config(cls, cfg: SimulationParams) -> "DerivedGeometry":
        return cls(
            e=cfg.e,
            inc=np.radians(cfg.inc_deg),
            Omega=np.radians(cfg.Omega_deg),
            omega=np.radians(cfg.omega_deg),
            obliquity=np.radians(cfg.obliquity_deg),
            spin_longitude=np.radians(cfg.spin_longitude_deg),
            planet_radius=cfg.planet_radius_km,
            ring_inner=cfg.ring_inner * cfg.planet_radius_km,
            ring_outer=cfg.ring_outer * cfg.planet_radius_km,
            a=(0.05 * u.AU).to(u.km).value, # use scaled separation here to reduce computation time, does not affect orbit calculations
            d=(cfg.distance_pc * u.pc).to(u.km).value)


class EpochResult:
    """Everything computed for one orbital epoch."""

    def __init__(self, epoch, true_anomaly, planet_center, star_dir, ring_normal,
                 phase_angle, planet_pts, planet_lit, planet_observable,
                 ring_pts, ring_lit, ring_observable,
                 planet_frac, ring_frac, ring_observable_frac, planet_observable_frac):
        self.epoch = epoch
        self.true_anomaly = true_anomaly
        self.planet_center = planet_center
        self.star_dir = star_dir
        self.ring_normal = ring_normal
        self.phase_angle = phase_angle
        self.planet_pts = planet_pts
        self.planet_lit = planet_lit
        self.planet_observable = planet_observable
        self.ring_pts = ring_pts
        self.ring_lit = ring_lit
        self.ring_observable = ring_observable
        self.planet_frac = planet_frac
        self.ring_frac = ring_frac
        self.ring_observable_frac = ring_observable_frac
        self.planet_observable_frac = planet_observable_frac


class SimulationResults:
    def __init__(self, epochs: List["EpochResult"]):
        self.epochs = epochs

# Orbit / reference frame (computed once, shared across all epochs)
def get_orbit_frame(geom: DerivedGeometry):
    """Full orbit track and the orbit-plane vectors (v1, v2, h_hat)."""
    orbit_coords = g.full_orbit(geom.a, geom.e, geom.inc, geom.Omega, geom.omega)
    R_pCi = g.R_pCi(geom.inc, geom.Omega, geom.omega)
    h = g.ang_momentum_vector(R_pCi)
    v1, rotation = g.orthogonal_to_h(h)
    v2 = rotation[:, 1]
    h_hat = rotation[:, 2]
    return orbit_coords, v1, v2, h_hat

def get_spin_axis(geom: DerivedGeometry, v1, v2, h_hat):
    """Unit ring normal/spin axis; obliquity is measured from the orbit normal h_hat."""
    tilt_dir = np.cos(geom.spin_longitude) * v1 + np.sin(geom.spin_longitude) * v2
    spin_axis = np.cos(geom.obliquity) * h_hat + np.sin(geom.obliquity) * tilt_dir
    return spin_axis / np.linalg.norm(spin_axis)


def ring_tilt_to_los(geom: DerivedGeometry, v1, v2, h_hat, observer_dir) -> float:
    """Angle between ring normal and line of sight; 0 = face-on."""
    return np.arccos(abs(np.dot(get_spin_axis(geom, v1, v2, h_hat), observer_dir)))

# Illumination tests
# ----------------------------------------------------------------------------
def planet_illumination(planet_pts, planet_normals, star_position, observer_dir,
                         planet_center, planet_radius, ring_normal, ring_inner, 
                         ring_outer) -> Tuple[np.ndarray, np.ndarray]:
    """
    Illumination/visibility test for the planet.
    """
    planet_lit = []
    planet_observable = []
    for p, n in zip(planet_pts, planet_normals):
        ray_origin = star_position
        ray_dir = p - star_position
        dist = np.linalg.norm(ray_dir)

        # Is this point on the visible hemisphere w.r.t. the observer?
        visible = np.dot(n, observer_dir) > 0
        planet_observable.append(visible)

        if not visible:
            planet_lit.append(False)
            continue

        lit = True  # assume illuminated until an intersection is found

        # Self-shadowing test: does another part of the planet block this point?
        t_planet = rc.intersect_sphere_distance(ray_origin, ray_dir, planet_center, planet_radius)
        if t_planet is not None and t_planet < dist - 1e-6:
            lit = False

        # Does the ring cast a shadow on this point?
        t_ring = rc.intersect_ring_distance(ray_origin, ray_dir, planet_center, 
                                            ring_normal, ring_inner, ring_outer)
        if t_ring is not None and t_ring < dist - 1e-6:
            lit = False

        planet_lit.append(lit)

    return np.array(planet_lit), np.array(planet_observable)


def ring_illumination(ring_pts, star_position, observer_dir, 
                       planet_center, planet_radius, 
                       ring_normal) -> Tuple[np.ndarray, np.ndarray]:
    """
    Illumination/visibility test for the ring.
    """
    ring_lit = []
    ring_observable = []
    for p in ring_pts:
        ray_origin = star_position
        ray_dir = p - star_position
        dist = np.linalg.norm(ray_dir)

        lit = True

        # Does the planet cast a shadow on the ring point?
        t_planet = rc.intersect_sphere_distance(ray_origin, ray_dir, planet_center, planet_radius)
        if t_planet is not None and t_planet < dist - 1e-6:
            lit = False

        ring_lit.append(lit)

        # Does this ring point face the star (same side as the observer)?
        to_star = star_position - p
        to_star /= np.linalg.norm(to_star)
        if np.sign(np.dot(ring_normal, to_star)) != np.sign(np.dot(ring_normal, observer_dir)):
            lit = False

        # Is this ring point hidden behind the planet w.r.t. the observer?
        t_planet_obs = rc.intersect_sphere_distance(p, observer_dir, planet_center, planet_radius)
        if t_planet_obs is not None:
            lit = False

        ring_observable.append(lit)

    return np.array(ring_lit), np.array(ring_observable)


# Per-epoch computation
def process_epoch(epoch: int, true_anomaly: float, 
                  cfg: SimulationParams, geom: DerivedGeometry, v1, 
                  v2, h_hat, star_position, observer_dir) -> EpochResult:

    theta = true_anomaly + geom.omega

    # compute spin axis and ring tilt direction relative to orbital plane
    ring_normal = get_spin_axis(geom, v1, v2, h_hat)

    # get planet position at epoch
    planet_center, _s = g.kepler_orbit(geom.a, geom.e, geom.inc, geom.Omega, geom.omega, true_anomaly, theta)
    planet_center_real, s_real = g.kepler_orbit((cfg.a_planet_au * u.AU).to(u.km).value, geom.e, 
                                                geom.inc, geom.Omega, geom.omega, true_anomaly, theta)
    print(f"\nEpoch {epoch}: Planet position (AU): {(s_real * u.km).to(u.AU):.4f} "
        f"(~{(((s_real * u.km).to(u.AU) / (geom.d * u.km).to(u.pc))).value * 1000:.4f} mas)")

    # vector from planet to star
    star_dir = star_position - planet_center
    star_dir /= np.linalg.norm(star_dir)

    # calculate phase angle with planet-star and planet-observer vectors
    to_star = star_position - planet_center
    phase_angle = np.degrees(np.arccos(np.dot(to_star, observer_dir) / (np.linalg.norm(to_star) * np.linalg.norm(observer_dir))))
    # Positive phase angle if the planet is on the right side of the star w.r.t. the observer, else negative
    if planet_center[0] < 0:
        phase_angle = phase_angle - 2 * phase_angle

    print(f"phase angle (from vectors): {phase_angle:.2f}")

    # Planet sampling
    planet_unit_pts, planet_normals = g.sample_sphere(n_theta=cfg.n_theta_planet, n_phi=cfg.n_phi_planet)
    planet_pts = planet_center + geom.planet_radius * planet_unit_pts # scale to planet position/size

    # Get reference for rotating ring points
    ref = v1
    if abs(np.dot(v1, ring_normal)) > 0.99:
        ref = v2  # fall back to v2 if ring normal is nearly parallel to v1
    e1 = np.cross(ring_normal, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ring_normal, e1)
    B = np.column_stack((e1, e2, ring_normal))  # local-to-inertial rotation

    # Ring sampling, generate points in ring plane 
    ring_pts0 = g.sample_ring(geom.ring_inner, geom.ring_outer, n_r=cfg.n_r_ring, n_phi=cfg.n_phi_ring)
    ring_pts = planet_center + ring_pts0 @ B.T 

    # Find ring normal to ring plane using ring points
    ring_normal = g.get_ring_normal(ring_pts)

    # Illumination tests
    planet_lit, planet_observable = planet_illumination(planet_pts, planet_normals, star_position, 
                                                        observer_dir, planet_center, geom.planet_radius, 
                                                        ring_normal, geom.ring_inner, geom.ring_outer)
    ring_lit, ring_observable = ring_illumination(ring_pts, star_position, observer_dir, planet_center, 
                                                  geom.planet_radius, ring_normal)

    planet_frac = planet_lit.mean()
    ring_frac = ring_lit.mean()
    ring_observable_frac = ring_observable.mean()
    planet_observable_frac = planet_lit[planet_observable].mean()

    print(f"Illuminated planet fraction: {planet_frac:.3f}")
    print(f"Illuminated ring fraction (from star --> ring):   {ring_frac:.3f}")
    print(f"Observable ring fraction:    {ring_observable_frac:.3f}")
    print("illuminated fraction of observable planet: ", planet_lit[planet_observable].mean())

    return EpochResult(epoch=epoch, true_anomaly=true_anomaly, planet_center=planet_center,
                       star_dir=star_dir, ring_normal=ring_normal, phase_angle=phase_angle, 
                       planet_pts=planet_pts, planet_lit=planet_lit, 
                       planet_observable=planet_observable, ring_pts=ring_pts, 
                       ring_lit=ring_lit, ring_observable=ring_observable, planet_frac=planet_frac,
                       ring_frac=ring_frac, ring_observable_frac=ring_observable_frac, 
                       planet_observable_frac=planet_observable_frac)

# Plotting
def init_3d_plot(orbit_coords):
    """
    Initialize the 3D orbit plot    
    """
    fig1 = plt.figure(figsize=(7, 7))
    ax = fig1.add_subplot(111, projection="3d")
    ax.plot(orbit_coords[:, 0], orbit_coords[:, 1], orbit_coords[:, 2],
            color="k", linestyle="dashed", alpha=0.6, linewidth=0.5)
    return fig1, ax


def plot_epoch_on_3d(ax, res: EpochResult, cfg: SimulationParams, geom: DerivedGeometry, 
                     star_position, observer_dir):
    """
    Plot all epochs together with the illuminated/shadowed planet/ring points
    shown. This should really only be done for a few epochs at a time.
    """
    ax.scatter(*res.planet_pts[res.planet_lit].T, s=1, alpha=0.6, c="gold")
    ax.scatter(*res.planet_pts[~res.planet_lit].T, s=1, alpha=0.5, c="black")

    if cfg.n_epochs < 20:
        ax.text(*res.planet_center + 1e6, f"{np.degrees(res.true_anomaly):.1f}\u00b0",
                color="k", fontsize=8, ha="center", va="center")

    ax.scatter(*res.ring_pts[res.ring_observable].T, s=1, alpha=0.6, c="goldenrod")
    ax.scatter(*res.ring_pts[~res.ring_observable].T, s=1, alpha=0.5, c="darkgray")

    ax.scatter(*star_position, color="gold", marker="*", s=100, edgecolors="k", linewidths=0.5)
    ax.quiver(*res.planet_center, *observer_dir, length=3 * geom.ring_inner, color="k", linewidth=1,
              label="Observer dir" if res.epoch == 0 else "")
    ax.quiver(*res.planet_center, *res.star_dir, length=3 * geom.ring_inner, color="gold", linewidth=1,
              linestyle="dashed", label="Star dir" if res.epoch == 0 else "")


def finalize_3d_plot(fig1, ax, cfg: SimulationParams, geom: DerivedGeometry, last_planet_center):
    """
    Finalize the plot parameters -- set the axes limits, labels, aspect ratio, title, 
    save figure(s). 
    """
    planet_sep = np.linalg.norm(last_planet_center)
    max_radius = geom.ring_outer + planet_sep + 1e6
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_zlim([-max_radius, max_radius])

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect(aspect="equal")
    ax.set_title(f"Incl. = {cfg.inc_deg:.1f}\u00b0\nRing obliquity = {cfg.obliquity_deg:.1f}\u00b0")
    ax.grid(False)
    ax.view_init(elev=-90, azim=90)  # view from "below" -- observer's perspective

    if cfg.save_figs:
        out_path = Path(cfg.plot_dir) / (
            f"all_epochs_inc{cfg.inc_deg:.1f}deg_obliq{cfg.obliquity_deg:.1f}deg_"
            f"{geom.ring_outer - geom.ring_inner:.2f}Rring.png"
        )
        fig1.savefig(out_path, dpi=400, bbox_inches="tight")
    fig1.show()


def plot_closeup(res: EpochResult, cfg: SimulationParams, geom: DerivedGeometry):
    """
    Plot the close-up view of each epoch to visualize the ringed planet system.
    """
    fig2, ax2 = plt.subplots(figsize=(7, 7))

    # Depth split so the ring and planet don't visually overlap -- compare
    # each ring point's z to the planet center's z to prevent ring/planet
    # overlapping each other on the plot.
    planet_z = res.planet_center[2]
    ring_z = res.ring_pts[:, 2]
    back_mask = ring_z > planet_z    # further from observer -> draw first
    front_mask = ring_z <= planet_z  # closer to observer -> draw last

    # Layer 1: back ring (behind planet disk)
    ax2.scatter(res.ring_pts[back_mask & ~res.ring_observable, 0],
                res.ring_pts[back_mask & ~res.ring_observable, 1],
                s=1, c="dimgray", alpha=0.8, zorder=1)
    ax2.scatter(res.ring_pts[back_mask & res.ring_observable, 0],
                res.ring_pts[back_mask & res.ring_observable, 1],
                s=1, c="goldenrod", alpha=0.7, zorder=1)

    # Layer 2: planet disk
    ax2.scatter(res.planet_pts[~res.planet_lit, 0], res.planet_pts[~res.planet_lit, 1],
                s=1, c="black", alpha=0.2, zorder=2)
    ax2.scatter(res.planet_pts[res.planet_lit, 0], res.planet_pts[res.planet_lit, 1],
                s=1, c="gold", alpha=0.9, zorder=2, label=f"phase={res.phase_angle:.1f}\u00b0")

    # Layer 3: front ring (on top of planet disk)
    ax2.scatter(res.ring_pts[front_mask & ~res.ring_observable, 0],
                res.ring_pts[front_mask & ~res.ring_observable, 1],
                s=1, c="dimgray", alpha=0.8, zorder=3)
    ax2.scatter(res.ring_pts[res.ring_lit & res.ring_observable, 0],
                res.ring_pts[res.ring_lit & res.ring_observable, 1],
                s=1, c="goldenrod", alpha=0.7, zorder=3)

    ax2.set_xlim([res.planet_center[0] - geom.ring_outer, res.planet_center[0] + geom.ring_outer])
    ax2.set_ylim([res.planet_center[1] - geom.ring_outer, res.planet_center[1] + geom.ring_outer])
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig2.patch.set_alpha(0)
    ax2.patch.set_alpha(0)

    if cfg.save_figs:
        out_path = Path(cfg.plot_dir) / (
            f"inc{cfg.inc_deg:.1f}_obl{cfg.obliquity_deg:.1f}_"
            f"v{np.degrees(res.true_anomaly):.1f}_phase{res.phase_angle:.1f}.png"
        )
        fig2.savefig(out_path, dpi=250, transparent=True, format="png", bbox_inches="tight")

    plt.tight_layout()
    fig2.show()


# File output
def output_filenames(cfg: SimulationParams, geom: DerivedGeometry, timestamp: str) -> Tuple[Path, Path]:
    out_dir = Path(cfg.output_dir)
    ring_width = geom.ring_outer - geom.ring_inner
    file_stem = (f"e{cfg.e}_inc{cfg.inc_deg:.1f}deg_obliq{cfg.obliquity_deg:.1f}deg_"
                 f"{ring_width:.2f}Rring_{cfg.n_epochs}epochs_{timestamp}.txt")
    
    return out_dir / f"planet_illumination_{file_stem}", out_dir / f"ring_illumination_{file_stem}"


def write_header(cfg: SimulationParams, geom: DerivedGeometry, f_planet_illum: Path, f_ring_illum: Path):
    header = (
        "# Ringed planet illumination simulation\n"
        f"# Semi-major axis (km): {geom.a}\n"
        f"# Eccentricity: {cfg.e}\n"
        f"# Inclination (deg): {cfg.inc_deg:.2f}\n"
        f"# Longitude of ascending node (deg): {cfg.Omega_deg:.2f}\n"
        f"# Argument of periastron (deg): {cfg.omega_deg:.2f}\n"
        f"# Planet radius (km): {geom.planet_radius}\n"
        f"# Ring inner radius (km): {geom.ring_inner}\n"
        f"# Ring outer radius (km): {geom.ring_outer}\n"
        f"# Ring obliquity (deg): {cfg.obliquity_deg:.2f}\n"
        f"# Number of epochs saved (i.e., number of samples around the orbit): {cfg.n_epochs}\n\n"
    )
    with open(f_planet_illum, "w") as f_planet, open(f_ring_illum, "w") as f_ring:
        f_planet.write(header)
        f_ring.write(header)


def save_outputs(cfg: SimulationParams, epochs: List[EpochResult], f_planet_illum: Path, f_ring_illum: Path):
    if not cfg.save_files:
        return

    phases = [r.phase_angle for r in epochs]
    planet_observable_lit = [r.planet_observable_frac for r in epochs]
    ring_observable_lit = [r.ring_observable_frac for r in epochs]
    true_anomaly_deg = [np.degrees(r.true_anomaly) for r in epochs]

    phases_sorted, planet_observable_lit = zip(*sorted(zip(phases, planet_observable_lit)))
    phases_sorted, ring_observable_lit = zip(*sorted(zip(phases, ring_observable_lit)))
    phases_sorted, true_anomaly_sorted = zip(*sorted(zip(phases, true_anomaly_deg)))

    with open(f_planet_illum, "a") as f_planet, open(f_ring_illum, "a") as f_ring:
        f_planet.write("Phase_angle(deg),true_anomaly(deg),Fraction_illuminated\n")
        for phase, nu, frac in zip(phases_sorted, true_anomaly_sorted, planet_observable_lit):
            f_planet.write(f"{phase},{nu},{frac:.4f}\n")

        f_ring.write("Phase_angle(deg),true_anomaly(deg),Fraction_illuminated\n")
        for phase, nu, frac in zip(phases_sorted, true_anomaly_sorted, ring_observable_lit):
            f_ring.write(f"{phase},{nu},{frac:.4f}\n")


# Run the full simulation to calculate the illumination of the ringed planet
def run_simulation(cfg: SimulationParams = None) -> SimulationResults:
    """Run the full simulation (all epochs) for the given system configuration."""
    cfg = cfg or SimulationParams()
    geom = DerivedGeometry.from_config(cfg)

    print(f"Ring inner radius (km): {geom.ring_inner} \t Ring outer radius (km): {geom.ring_outer}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f_planet_illum, f_ring_illum = output_filenames(cfg, geom, timestamp)
    if cfg.save_files:
        write_header(cfg, geom, f_planet_illum, f_ring_illum)

    star_position = np.array([0.0, 0.0, 0.0])     # Star at origin
    observer_dir = np.array([0.0, 0.0, -geom.d])  # observer in -z-direction
    observer_dir /= np.linalg.norm(observer_dir)

    orbit_coords, v1, v2, h_hat = get_orbit_frame(geom)

    # Initialize the 3D fig
    fig1, ax = (init_3d_plot(orbit_coords) if cfg.plot else (None, None))

    true_anomalies = np.arange(0, 2 * np.pi, 2 * np.pi / cfg.n_epochs)

    epochs: List[EpochResult] = []
    for epoch, true_anomaly in enumerate(true_anomalies):
        res = process_epoch(epoch, true_anomaly, cfg, geom, v1, v2, h_hat, star_position, observer_dir)
        epochs.append(res)

        if cfg.plot:
            plot_epoch_on_3d(ax, res, cfg, geom, star_position, observer_dir)
        if cfg.plot_closeup:
            plot_closeup(res, cfg, geom)

    if cfg.plot:
        finalize_3d_plot(fig1, ax, cfg, geom, epochs[-1].planet_center)

    save_outputs(cfg, epochs, f_planet_illum, f_ring_illum)

    return SimulationResults(epochs=epochs)


def main():
    run_simulation(SimulationParams())


if __name__ == "__main__":
    main()