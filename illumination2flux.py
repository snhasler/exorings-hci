"""
author: S. Hasler
Apply the f_planet and f_ring illumination fractions to the planet
and ring spectra.

This script:
1. Reads precomputed illumination-fraction files for the planet and rings
   for one or more simulation configurations (different inclinations / ring
   sizes), using output files from calc_ringed_planet_illumination.py.
2. Applies f_planet and f_ring to the input albedo spectra.
3. Applies a Lambertian phase function to the planet spectrum.
4. Converts albedo spectra to flux / contrast space.
5. Combines F_planet and F_ring to produce F_total.
6. Computes band-integrated flux in the Roman bandpasses.
7. Saves output spectra and phase curves.
8. Overlays phase curves from all configurations.
"""

import re
from pathlib import Path
from dataclasses import dataclass    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u

import utils

plt.rcParams.update({"font.size": 14})

ROMAN_FILTER_DIR = Path("filters/roman/")

STELLAR_SPECTRUM_FILE = Path("spectra/star/SOLARSPECTRUM.DAT")
KARKOSCHKA_SPECTRA  = Path("spectra/planet/karkoschka1995low.dat")
RINGS_SPECTRA_FILE  = Path("spectra/rings/Bring_110-111e3km_Hedman2013_nm.txt")

B1_FILTER = ROMAN_FILTER_DIR / "transmission_ID-01_1F_v0.csv"
B2_FILTER = ROMAN_FILTER_DIR / "transmission_ID-02_2F_v0.csv"
B3_FILTER = ROMAN_FILTER_DIR / "transmission_ID-03_3F_v0.csv"
B4_FILTER = ROMAN_FILTER_DIR / "transmission_ID-04_4F_v0.csv"
B3C_FILTER = ROMAN_FILTER_DIR / "transmission_ID-21_3C_v0.csv"
B4C_FILTER = ROMAN_FILTER_DIR / "transmission_ID-18_4C_v0.csv"

# Physical constants fixed for every run
RPLANET    = 69911 * u.km
SEPARATION = 5.2   * u.AU # planet-star separation

roman_filter     = "1" # Roman filter name (1, 3C, 4, 4C)
SAVE_OUTPUTS     = True
MAKE_PLOTS       = True
# Phase function string for text file
PLANET_PHASEFUNC = "Lambertian"
PLOT_DIR = "fraction_lit_output/plots/" # where plots will be output

# Define all configurations to process.
# Point RUN_DIRS at one directory per ring-size/obliquity combination;
# all inclinations found inside each directory will be picked up
RUN_DIRS = [
        Path("fraction_lit_output/1xSat_obl26.73_allIncl/"), # <-- this should be the path where your output files are stored
            ]

RINNER = 1.43 * RPLANET # inner ring radius
ROUTER = 2.47 * RPLANET # outer ring radius
OBLIQ = 26.73           # ring obliquity in degrees

FILTER_FILE = B1_FILTER # filter to use for processing

@dataclass
class RunConfig:
    """
    All parameters that vary between runs.

    Each entry below should be one RunConfig. Parameters that
    are the same for every run (RPLANET, SEPARATION, file paths, etc.) stay as global 
    variables above.
    """
    label: str               # Legend text used in plots
    file_dir: Path           # Directory that holds the illumination files
    planet_illum_file: str   # Planet illumination filename
    ring_illum_file:   str   # Ring illumination filename

    # Orbital & ring geometry
    inc:       float        # Orbital inclination to the line of sight (deg)
    obliq:     float        # Ring obliquity (deg)
    ring_tilt: float        # Ring tilt to the line of sight (rad; from file header)
    ecc:       float        # Orbital eccentricity
    ring_size: float        # Ring size used in filename (km)
    epochs:    int          # Number of orbital epochs simulated
    rinner:    u.Quantity   # Ring inner radius
    router:    u.Quantity   # Ring outer radius

    # Plot styling
    color:     str          
    linestyle: str = "solid"   # linestyle
    filter_name: str = roman_filter     # Roman filter name

    def build_output_filenames(self) -> dict:
        """Build output filenames from the simulation parameters."""
        stem = (f"e{self.ecc}_inc{self.inc}deg_obliq{self.obliq}deg_"
                f"{self.ring_size}Rring_{self.epochs}epochs_BAND{self.filter_name}")
        return {
            "planet_albedo":
                f"planet_spectra_{stem}_{PLANET_PHASEFUNC}.txt",
            "ring_albedo":
                f"ring_spectra_{stem}.txt",
            "planet_flux":
                f"planet_flux_spectra_{stem}_{PLANET_PHASEFUNC}.txt",
            "ring_flux":
                f"ring_flux_spectra_{stem}.txt",
            "combined_flux":
                f"combined_flux_spectra_{stem}_{PLANET_PHASEFUNC}planet.txt",
            "combined_contrast":
                f"combined_contrast_spectra_{stem}_{PLANET_PHASEFUNC}planet.txt",
            "combined_phasecurve":
                f"combined_phasecurve_{stem}_{PLANET_PHASEFUNC}planet.txt",
            }

# Each run directory (one ring size + obliquity) holds pairs of
# planet_illumination_e{ecc}_inc{inc}deg_obliq{obliq}deg_{ring_size}Rring_{epochs}epochs_{timestamp}.txt
# ring_illumination_e{ecc}_inc{inc}deg_obliq{obliq}deg_{ring_size}Rring_{epochs}epochs_{timestamp}.txt
# files that differ only by inclination. The regex below recovers ecc/inc/obliq/ring_size/epochs 
# from the filename so the pairs can be matched and a RunConfig can be built automatically.
ILLUM_FILENAME_RE = re.compile(
    r"^(?P<kind>planet|ring)_illumination_"
    r"e(?P<ecc>[\d.]+)_"
    r"inc(?P<inc>[\d.]+)deg_"
    r"obliq(?P<obliq>[\d.]+)deg_"
    r"(?P<ring_size>[\d.]+)Rring_"
    r"(?P<epochs>\d+)epochs_"
    r"(?P<timestamp>\d{8}_\d{6})"
    r"\.txt$"
)

# Color cycle, applied in order of increasing inclination.
DEFAULT_COLORS = [
    "#a76c9b", "#4f9fab", "#4cd0c5", "#e790ab",
    "#f2a154", "#5159a8", "#9bd85a", "#f25f5f",
]

def parse_illum_filename(filename: str) -> dict:
    """Extract simulation parameters froman illumination filename.

    Raises ValueError if filename doesn't match the expected pattern.
    """
    match = ILLUM_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    d = match.groupdict()
    return {
        "kind":      d["kind"],
        "ecc":       float(d["ecc"]),
        "inc":       float(d["inc"]),
        "obliq":     float(d["obliq"]),
        "ring_size": float(d["ring_size"]),
        "epochs":    int(d["epochs"]),
        "timestamp": d["timestamp"],
    }

def build_configs_from_dir(
    file_dir,
    rinner: u.Quantity,
    router: u.Quantity,
    obliq: float = None,
    colors: list = None,
    linestyle: str = "solid",
) -> list:
    """
    Scan file_dir for planet/ring illumination file pairs and build one
    RunConfig per inclination found.

    Assumes that one directory corresponds to one ring size + obliquity, 
    with separate planet/ring illumination files for each inclination. 
    rinner and router are passed explicitly since they aren't uniquely recoverable from the 
    filenames

    Parameters
    ----------
    file_dir : Path or str
        Directory containing planet_illumination_*.txt and
        ring_illumination_*.txt files (e.g.
        "../2xSat_obl26.73_varyingIncl/").
    rinner, router : astropy.units.Quantity
        Ring inner/outer radii, applied to every config built from this
        directory.
    obliq : float, optional
        If given, overrides the obliquity parsed from filenames (in the event that 
        filenames carry a rounded value — e.g. "obliq26.7deg").
    colors : list of str, optional
        Matplotlib colors, assigned in order of increasing inclination.
        Defaults to DEFAULT_COLORS, cycled if there are more files than
        colors.
    linestyle : str
        Linestyle applied to every run built from this directory.

    Returns
    -------
    list of RunConfig, sorted by inclination.
    """
    file_dir = Path(file_dir)
    colors   = colors or DEFAULT_COLORS

    planet_files, ring_files = {}, {}
    for f in sorted(file_dir.glob("*_illumination_*.txt")):
        try:
            meta = parse_illum_filename(f.name)
        except ValueError:
            print(f"[WARNING] Skipping file with unrecognized name: {f.name}")
            continue

        key = (meta["ecc"], meta["inc"], meta["obliq"],
               meta["ring_size"], meta["epochs"])
        target = planet_files if meta["kind"] == "planet" else ring_files

        if key in target:
            print(
                f"[WARNING] Multiple {meta['kind']} files match inc={meta['inc']}; "
                f"keeping {target[key].name}, ignoring {f.name}"
            )
            continue
        target[key] = f

    matched_keys      = set(planet_files) & set(ring_files)
    unmatched_ring    = set(ring_files) - matched_keys
    unmatched_planet  = set(planet_files) - matched_keys
    for key in unmatched_ring:
        print(f"[WARNING] Ring file has no matching planet file: {ring_files[key].name}")
    for key in unmatched_planet:
        print(f"[WARNING] Planet file has no matching ring file: {planet_files[key].name}")
    if not matched_keys:
        raise FileNotFoundError(
            f"No matched planet/ring illumination file pairs found in {file_dir}"
        )

    configs = []
    for i, key in enumerate(sorted(matched_keys, key=lambda k: k[1])):  # sort by inc
        ecc, inc, file_obliq, ring_size, epochs = key
        configs.append(RunConfig(
            label=fr"$i={inc:g}°$",
            file_dir=file_dir,
            planet_illum_file=planet_files[key].name,
            ring_illum_file=ring_files[key].name,
            inc=inc,
            obliq=obliq if obliq is not None else file_obliq,
            ring_tilt=utils.read_ring_tilt(ring_files[key]),
            ecc=ecc,
            ring_size=ring_size,
            epochs=epochs,
            rinner=rinner,
            router=router,
            color=colors[i % len(colors)],
            linestyle=linestyle,
        ))
    return configs

# Set up configurations to process
CONFIGS = []
for _dir in RUN_DIRS:
    CONFIGS.extend(
        build_configs_from_dir( 
            file_dir=_dir,
            rinner=RINNER,
            router=ROUTER, 
            obliq=OBLIQ,   # overrides rounded obliquities in the filenames
        )
    )

# Utility functions
# -------------------------------------------------------------------
def sort_by_phase(phases, nus, lit_fracs):
    """
    Sort arrays by phase angle and ensure all angles are positive
    values.
    
    Parameters
    ----------
    phases : array-like
        Phase angles
    nus : array-like
        True anomalies
    lit_fracs : array-like
        Illumination fractions
    Returns
    -------
    tuple of np.ndarray
        Sorted (phases, nus, lit_fracs) arrays
    """
    phases    = np.abs(np.asarray(phases))
    nus       = np.asarray(nus)
    lit_fracs = np.asarray(lit_fracs)

    sorted_triplets = sorted(zip(phases, nus, lit_fracs), key=lambda x: x[0])
    phases_s, nus_s, lit_s = map(np.array, zip(*sorted_triplets))
    return phases_s, nus_s, lit_s

def get_filter_lims_for_plot(filt_name):
    '''Returns min, max wavelenghths of Roman filter of interest.'''
    # read in filter data
    wavel, transm = utils.read_roman_filter_curve(filt_name)
    # get range over which filter has non-zero transmission
    mask = transm > 0

    return wavel[mask].min(), wavel[mask].max()

def plot_spectra_dict(wavelength, spectra_dict, title, 
                      ylabel="Albedo spectrum", cmap_name="viridis"):
    """Plot spectra stored in the dictionary."""

    # read in some filters for plotting
    b1_min, b1_max = get_filter_lims_for_plot("1F")
    b2_min, b2_max = get_filter_lims_for_plot("2F")
    b3_min, b3_max = get_filter_lims_for_plot("3F")
    b4_min, b4_max = get_filter_lims_for_plot("4F")

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap    = plt.get_cmap(cmap_name)
    phases  = np.array(list(spectra_dict.keys()))

    for i, (phase, spectrum) in enumerate(spectra_dict.items()):
        ax.plot(wavelength, spectrum, color=cmap(i))

    norm = plt.Normalize(np.min(phases), np.max(phases))
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=np.linspace(0, 180, 10))
    cbar.set_label("Phase angle (degrees)")

    ax.axvspan(b1_min, b1_max, color="powderblue", alpha=0.5, label="B1")     
    ax.axvspan(b2_min, b2_max, color="lightgreen", alpha=0.35, label="B2")     
    ax.axvspan(b3_min, b3_max, color="orange", alpha=0.35, label="B3")     
    ax.axvspan(b4_min, b4_max, color="palevioletred", alpha=0.45, label="B4") 
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(300, 1005)
    ax.set_ylim(-0.01, None)
    plt.tight_layout()
    return fig, ax

def compute_band_albedo_per_phase(spectra_dict, wavelength_nm, filter_file):
    """Compute band-averaged albedo for each angle."""
    return {
        phase: utils.calc_albedo_in_band(
            filter_file, spectrum, wavelength_nm * u.nm,
        )
        for phase, spectrum in spectra_dict.items()
    }

def convert_planet_spectra_to_flux(spectra_dict, wavelength_angstrom, 
                                   stellar_spectrum, stellar_wavelength):
    """
    Convert planet albedo spectra to flux and planet-star flux ratio.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary of planet albedo spectra keyed by angle.
    wavelength_angstrom : array-like
        Wavelength grid in Angstroms.
    stellar_spectrum : array-like
        Stellar flux spectrum.
    stellar_wavelength : array-like
        Stellar wavelength grid in Angstroms.

    Returns
    -------
    flux_dict : dict
        Dictionary of planet flux spectra
    fpfs_dict : dict
        Dictionary of planet-star flux ratio
    fs_at_planet : array-like
        Stellar flux at the planet's location
    """
    flux_dict, fpfs_dict, fs_at_planet = {}, {}, None
    for phase, spectrum in spectra_dict.items():
        flux, fpfs, fs_at_planet = utils.convert_planet_albedo_to_flux(
            spectrum, wavelength_angstrom,
            stellar_spectrum, stellar_wavelength,
            separation=SEPARATION.to(u.km),
            Rp=RPLANET,
        )
        flux_dict[phase] = flux
        fpfs_dict[phase] = fpfs
    return flux_dict, fpfs_dict, fs_at_planet


def convert_ring_spectra_to_flux(spectra_dict, wavelength_angstrom, 
                                 stellar_spectrum, stellar_wavelength,
                                 cfg: RunConfig):
    """
    Convert ring albedo spectra per angle to flux and Fp/Fs.

    Ring geometry (inner/outer radius, obliquity) is taken from *cfg* so that
    configurations with different ring sizes or obliquities are handled
    correctly.
    """
    flux_dict, fpfs_dict, fs_at_planet = {}, {}, None
    for phase, spectrum in spectra_dict.items():
        flux, fpfs, fs_at_planet = utils.convert_ring_albedo_to_flux(
            spectrum, wavelength_angstrom,
            stellar_spectrum, stellar_wavelength,
            separation=SEPARATION.to(u.km),
            R_inner=cfg.rinner,                          
            R_outer=cfg.router,                          
            ringinc_rad=np.radians(cfg.ring_tilt),
        )
        flux_dict[phase] = flux
        fpfs_dict[phase] = fpfs
    return flux_dict, fpfs_dict, fs_at_planet

def compute_band_fluxes_from_albedo(spectra_dict, wavelength_angstrom,
                                    stellar_spectrum, stellar_wavelength, 
                                    filter_file):
    """
    Compute band-integrated planet flux and flux ratio from albedo spectra.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary of albedo spectra keyed by angle.
    wavelength_angstrom : array-like
        Wavelenegth grid for albedo spectra in Angstroms.
    stellar_spectrum : array-like
        Stellar flux spectrum.
    stellar_wavelength : array-like
        Stellar wavelength grid in Angstroms.
    filter_file : str or Path
        Filter file for band integration.
    """
    flux_band, fpfs_band, albedo_band = {}, {}, {}
    for phase, spectrum in spectra_dict.items():
        fp_band, fpfs_val = utils.calc_flux_in_band(
            filter_file, spectrum, wavelength_angstrom,
            stellar_spectrum=stellar_spectrum,
            stellar_wavelength=stellar_wavelength,
            separation=SEPARATION,
            Rp=RPLANET,
        )
        ag_band = utils.calc_albedo_in_band(
            filter_file, spectrum, wavelength_angstrom,
        )
        flux_band[phase]   = fp_band
        fpfs_band[phase]   = fpfs_val
        albedo_band[phase] = ag_band
    return flux_band, fpfs_band, albedo_band

def compute_band_fluxes_from_flux_spectra(flux_spectra_dict, wavelength_angstrom,
                                          stellar_spectrum, stellar_wavelength, 
                                          filter_file):
    """
    Compute band-integrated flux and flux ratio from flux spectra.
    
    Parameters
    ----------
    flux_spectra_dict : dict
        Dictionary of flux spectra keyed by angle.
    wavelength_angstrom : array-like
        Wavelenegth grid for flux spectra in Angstroms.
    stellar_spectrum : array-like
        Stellar flux spectrum.
    stellar_wavelength : array-like
        Stellar wavelength grid in Angstroms.
    filter_file : str or Path
        Filter file for band integration.
    
    Returns
    -------
    flux_band : dict
        Dictionary of band-integrated fluxes
    fpfs_band : dict
        Dictionary of band-integrated flux ratios
    """
    flux_band, fpfs_band = {}, {}
    for phase, flux_spectrum in flux_spectra_dict.items():
        fp_band, fpfs_val = utils.flux_in_band_from_fluxSpec(
            filter_file, flux_spectrum, wavelength_angstrom,
            stellar_spectrum=stellar_spectrum,
            stellar_wavelength=stellar_wavelength,
            separation=SEPARATION,
        )
        flux_band[phase] = fp_band
        fpfs_band[phase] = fpfs_val
    return flux_band, fpfs_band

def process_config(cfg: RunConfig, lambda_vac_nm, jup_albedo, 
                   ring_albedo, stellar_wavel, stellar_spectrum, 
                   filter_file) -> dict:
    """
    Run the full illumination-fraction pipeline for one RunConfig.

    Spectral data (jup_albedo, ring_albedo, stellar_) are passed in from
    main() so they are loaded only once and shared across all configs.

    Parameters
    ----------
    cfg              : RunConfig   - simulation configuration
    lambda_vac_nm    : ndarray     - wavelength grid (nm)
    jup_albedo       : ndarray     - Jupiter's albedo spectrum
    ring_albedo      : ndarray     - ring albedo spectrum on lambda_vac_nm grid
    stellar_wavel    : ndarray     - stellar wavelength array (Angstrom)
    stellar_spectrum : ndarray     - stellar flux spectrum
    filter_file      : str or Path - filter file for band integration

    Returns
    -------
    dict containing processed spectra, band-integrated quantities, and
    metadata needed by the plotting functions.
    """
    lambda_vac_A  = (lambda_vac_nm * u.nm).to(u.Angstrom)
    output_files  = cfg.build_output_filenames()

    # ── Load illumination fractions ──────────────────────────────────
    planet_phases, planet_nu, planet_lit = utils.read_illumination_file(cfg.file_dir / cfg.planet_illum_file)
    ring_phases, ring_nu, ring_lit = utils.read_illumination_file(cfg.file_dir / cfg.ring_illum_file)

    planet_phases, planet_nu, planet_lit = sort_by_phase(planet_phases, planet_nu, planet_lit)
    ring_phases, ring_nu, ring_lit = sort_by_phase(ring_phases, ring_nu, ring_lit)

    # ── Apply illumination fractions & Lambertian phase function ─────
    spectra_per_phase_planet      = {}   # planet with ring shadowing
    spectra_per_phase_planet_only = {}   # ringless Lambertian for reference

    for i, phase in enumerate(planet_phases):
        lambert = utils.phi_lambert(phase)
        spectra_per_phase_planet_only[i] = jup_albedo * lambert
        spectra_per_phase_planet[i] = jup_albedo * planet_lit[i] * lambert

    rings_spectra_per_phase, _ = utils.apply_litFraction_to_spectrum(ring_phases, ring_nu, 
                                                                     ring_lit, ring_albedo)

    # ── Save albedo spectra ──────────────────────────────────────────
    if SAVE_OUTPUTS:
        utils.save_spectra_per_phase_to_file(spectra_per_phase_planet,
                                             lambda_vac_nm, 
                                             cfg.file_dir / output_files["planet_albedo"])
        utils.save_spectra_per_phase_to_file(rings_spectra_per_phase,
                                             lambda_vac_nm,
                                             cfg.file_dir / output_files["ring_albedo"])

    # ── albedo spectra plots ────────────────────────
    if MAKE_PLOTS:
        plot_spectra_dict(lambda_vac_nm, spectra_per_phase_planet, 
                          title=f"Planet albedo spectra - {cfg.label}",
                          cmap_name="PuRd_r")
        plot_spectra_dict(lambda_vac_nm, rings_spectra_per_phase,
                          title=f"Ring albedo spectra - {cfg.label}", 
                          cmap_name="PuBu_r")

    # ── Band-averaged albedo ─────────────────────────────────────────
    planet_band_albedo = compute_band_albedo_per_phase(spectra_per_phase_planet, 
                                                       lambda_vac_nm, filter_file)
    ring_band_albedo = compute_band_albedo_per_phase(rings_spectra_per_phase, 
                                                     lambda_vac_nm, filter_file)

    # ── Convert albedo to flux ──────────────────────────────────
    planet_flux_dict, _, fs_at_planet = convert_planet_spectra_to_flux(spectra_per_phase_planet, 
                                                                       lambda_vac_A, stellar_spectrum, 
                                                                       stellar_wavel)
    planet_only_flux_dict, _, _ = convert_planet_spectra_to_flux(spectra_per_phase_planet_only, 
                                                                 lambda_vac_A, stellar_spectrum, 
                                                                 stellar_wavel)
    ring_flux_dict, _, _ = convert_ring_spectra_to_flux(rings_spectra_per_phase, lambda_vac_A, 
                                                        stellar_spectrum, stellar_wavel, cfg)

    if SAVE_OUTPUTS: # Save files
        utils.save_spectra_per_phase_to_file(planet_flux_dict, lambda_vac_nm, 
                                             cfg.file_dir / output_files["planet_flux"])
        utils.save_spectra_per_phase_to_file(ring_flux_dict, lambda_vac_nm, 
                                             cfg.file_dir / output_files["ring_flux"])

    # ── Combine planet + ring ────────────────────────────────────────
    total_flux_dict = {phase: planet_flux_dict[phase] + ring_flux_dict[phase] for phase in planet_flux_dict}
    total_contrast_dict = {phase: total_flux_dict[phase] / fs_at_planet for phase in total_flux_dict}

    if SAVE_OUTPUTS:
        utils.save_spectra_per_phase_to_file(total_flux_dict, lambda_vac_A,
                                             cfg.file_dir / output_files["combined_flux"])
        utils.save_spectra_per_phase_to_file(total_contrast_dict, lambda_vac_nm,
                                             cfg.file_dir / output_files["combined_contrast"])

    # ── Band-integrated flux ratio ──────────────────────────────────
    _, fpfsBand_planet_dict, _ = compute_band_fluxes_from_albedo(spectra_per_phase_planet,
                                                                 lambda_vac_A, stellar_spectrum, 
                                                                 stellar_wavel, filter_file)
    _, fpfsBand_planet_only_dict, _ = compute_band_fluxes_from_albedo(spectra_per_phase_planet_only,
                                                                      lambda_vac_A, stellar_spectrum, 
                                                                      stellar_wavel, filter_file)
    _, fpfsBand_combined_dict = compute_band_fluxes_from_flux_spectra(total_flux_dict, lambda_vac_A, 
                                                                      stellar_spectrum, stellar_wavel, 
                                                                      filter_file)

    # ── Save phase-curve info ───────────────────────────────────────
    phasecurve_df = pd.DataFrame(
            {
                "phase_angle": [float(planet_phases[i]) for i in total_flux_dict.keys()],
                "true_anomaly": [float(planet_nu[i]) for i in total_flux_dict.keys()],
                "fpfs_ringedplanet": [float(fpfsBand_combined_dict[i]) for i in total_flux_dict.keys()],
            }
        )
    if SAVE_OUTPUTS:
        phasecurve_df.to_csv(cfg.file_dir / output_files["combined_phasecurve"], index=False)

    # ── Return everything the plots will need ───────────
    return {
        "cfg":                       cfg,
        "planet_nu":                 planet_nu,
        "planet_phases":             planet_phases,
        "lambda_vac_A":              lambda_vac_A,
        "spectra_per_phase_planet":  spectra_per_phase_planet,
        "rings_spectra_per_phase":   rings_spectra_per_phase,
        "planet_band_albedo":        planet_band_albedo,
        "ring_band_albedo":          ring_band_albedo,
        "fpfsBand_planet_dict":        fpfsBand_planet_dict,
        "fpfsBand_planet_only_dict":   fpfsBand_planet_only_dict,
        "fpfsBand_combined_dict":      fpfsBand_combined_dict,
    }

# phase-curve plots
def plot_all_phase_curves(all_results: list) -> tuple:
    """
    Overlay phase curves for every processed configuration.

    The ringless Lambertian reference is plotted once in gray, 
    plotted from the first config's phase coverage.  Each configuration's combined 
    (planet + ring) curve is plotted with the color and linestyle specified in its RunConfig.

    Parameters
    ----------
    all_results : list of dict
        List of result dicts returned by process_config().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Ringless Lambertian reference –
    first       = all_results[0]
    sorted_idx  = np.argsort(first["planet_nu"])
    nu_sorted   = np.array(first["planet_nu"])[sorted_idx]
    fpfs_ringless = np.array(
        list(first["fpfsBand_planet_only_dict"].values())
    )[sorted_idx]
    ax.plot(nu_sorted, fpfs_ringless, color="gray", ls="dashed", lw=2), #label="Ringless planet")

    # One combined (planet + ring) curve per 
    for result in all_results:
        cfg         = result["cfg"]
        sorted_idx  = np.argsort(result["planet_nu"])
        nu_sorted   = np.array(result["planet_nu"])[sorted_idx]
        fpfs_combined = np.array(
            list(result["fpfsBand_combined_dict"].values())
        )[sorted_idx]

        ax.plot(nu_sorted, fpfs_combined, color=cfg.color, lw=2, ls=cfg.linestyle,
                label=cfg.label, alpha=0.9)

    ax.set_xlabel("True anomaly (deg)")
    ax.set_ylabel(r"$F_p/F_s$")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f'obl{OBLIQ}_Rin{RINNER}_Rout{ROUTER}_all_phase_curves.png', 
                dpi=300, bbox_inches='tight')
    return fig, ax

def main():
    # Load spectral data once 
    lambda_vac_nm, jup_albedo = utils.get_karkoschka_Jupiter_spectrum()
    ring_albedo               = utils.load_ring_albedo(lambda_vac_nm)
    stellar_wavel, stellar_spectrum = utils.read_stellar_spectrum(STELLAR_SPECTRUM_FILE, 
                                                                  wavel_units=u.Angstrom)
    
    # Process every configuration defined 
    all_results = []
    for cfg in CONFIGS:
        print(f"Processing: {cfg.label}")
        result = process_config(cfg, lambda_vac_nm, jup_albedo, ring_albedo,
                                stellar_wavel, stellar_spectrum, filter_file=FILTER_FILE) 
        all_results.append(result)

    # Overlay all configurations on shared plots
    if MAKE_PLOTS:
        plot_all_phase_curves(all_results)
        plt.show()

if __name__ == "__main__":
    main()