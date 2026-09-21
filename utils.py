# S. Hasler
# Useful functions for exorings model

import numpy as np
import astropy.units as u
import os
import pandas as pd

def normalize(vector):
    '''
    Normalize vector

    Parameters
    ----------
    vector : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    return vector / np.linalg.norm(vector)

def on_sky_separation(system_distance_pc, planet_separation_AU):
    '''
    Calculate the on-sky separation of the planet from its host star in arcseconds

    Parameters
    ----------
    system_distance_pc : astropy.units.quantity.Quantity
        Distance of system from observer
    planet_separation_AU : astropy.units.quantity.Quantity
        Planet-star separation

    '''
    # Check that input values are astropy quantities
    if not isinstance(system_distance_pc, u.Quantity):
        raise TypeError("system_distance_pc must be an astropy.units.Quantity.")
    if not isinstance(planet_separation_AU, u.Quantity):
        raise TypeError("planet_separation_AU must be an astropy.units.Quantity.")
    
    # Check for proper units
    if not system_distance_pc.unit.is_equivalent(u.pc):
        raise u.UnitTypeError("system_distance_pc must be in units of pc.")
    if not planet_separation_AU.unit.is_equivalent(u.AU):
        raise u.UnitTypeError("planet_separation_AU must be in units of AU.")
    
    return (planet_separation_AU.value / system_distance_pc.value) * u.arcsec # [arcseconds]

def ring_area(R_inner, R_outer):
    '''
    Calculate area of a ring given the inner and outer radii

    Parameters
    ----------
    R_inner : float
        Inner radius
    R_outer : float
        Outer radius

    Returns
    -------
    Area of ring in input units squared

    '''
    return np.pi * (R_outer**2 - R_inner**2) 

def lambda_over_D(wavelength, diameter):
    '''
    Calculate angular resolution (lambda / D) in milliarcseconds

    Parameters
    ----------
    wavelength : astropy.units.quantity.Quantity
        Wavelength of observation [m]
    diameter : astropy.units.quantity.Quantity
        Diameter of telescope aperture [m]

    '''
    # Check that input is an astropy quantity
    if not isinstance(wavelength, u.Quantity):
        raise TypeError("wavelength must be an astropy.units.Quantity.")
    if not isinstance(diameter, u.Quantity):
        raise TypeError("diameter must be an astropy.units.Quantity.")

    # Check that it has compatible units
    if not wavelength.unit.is_equivalent(u.m):
        raise u.UnitTypeError("wavelength must be in units of m.")
    if not diameter.unit.is_equivalent(u.m):
        raise u.UnitTypeError("diameter must be in units of m.")

    return 1.22 * (wavelength / diameter * u.rad).to(u.mas)

def ring_angular_extent(R_outer, system_distance):
    '''
    Calculate angular extent of rings in milliarcseconds.
    
    Parameters
    ----------
    R_inner : astropy.units.quantity.Quantity
        Inner radius of rings [km]
    R_outer : astropy.units.quantity.Quantity
        Outer radius of rings [km]
    system_distance : astropy.units.quantity.Quantity
        Distance to system [pc]
    
    '''
    # Check if system distance has units and is entered in pc
    if not isinstance(system_distance, u.Quantity):
        raise TypeError("system_distance must be an astropy.units.Quantity.")
    if not system_distance.unit.is_equivalent(u.pc):
        raise u.UnitTypeError("system_distance must be in units of pc.")
    
    # Check if ring radii have units and are entered in km
    if not isinstance(R_outer, u.Quantity):
        raise TypeError("R_outer must be an astropy.units.Quantity.")
    
    d_ringUnits = system_distance.to(R_outer.unit) # convert system distance to ring units
    
    return ( (R_outer*2) / d_ringUnits * u.rad).to(u.mas) # angular extent in mas
    

def get_karkoschka_Jupiter_spectrum(filepath="spectra/planet/karkoschka1995low.dat"):
    """
    Read in Karkoschka Jupiter albedo spectrum from file

    Parameters
    ----------
    filepath : str
        Path to Karkoschka Jupiter albedo spectrum file
    """
    vacuum_lambda_nm, air_lambda_nm, ch4_abs, alb_Jup, \
        alb_Sat, alb_Uranus, alb_Nep, alb_Titan = np.loadtxt(filepath, unpack=True)

    return vacuum_lambda_nm, alb_Jup

def load_ring_albedo(target_wavelength_nm, 
                     filepath="spectra/rings/Bring_110-111e3km_Hedman2013_nm.txt"):
    """
    Load Hedman+2013 ring spectrum and interpolate onto the planet
    wavelength grid.
    """
    lambda_rings, if_rings = np.loadtxt(filepath, skiprows=1, dtype=float, 
                                        unpack=True, delimiter=",")
    albedo_rings    = np.interp(target_wavelength_nm, lambda_rings, if_rings)

    return albedo_rings

def phi_lambert(alpha_deg):
    """ Lambertian phase function normalized to 1 at alpha=0.
    
    Parameters
    ----------
    alpha_deg : float
        Phase angle in degrees.
    
    Returns
    -------
    float
        Value of Lambertian phase function at given phase angle.
    """
    alpha = np.deg2rad(alpha_deg)
    return (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha)) / np.pi

def FpFs_planet(Ag_planet, R_planet_AU, a_AU, alpha_deg):
    """
    Calculates planet contrast at phase angle with a Lambertian.

    Parameters
    ----------
    Ag_planet : float
        Geometric albedo of the planet.
    R_planet_AU : float
        Radius of the planet in AU.
    a_AU : float
        Semi-major axis of the planet's orbit in AU.
    alpha_deg : float
        Phase angle in degrees.

    Returns
    -------
    float
        Planet flux ratio (Fp/Fs)
    """
    # Planet brightness (Lambertian)
    phi_planet = phi_lambert(alpha_deg)

    return  Ag_planet * (R_planet_AU / a_AU)**2 * phi_planet

def read_roman_filter_curve(filter_name, filepath="filters/roman/"):
    """
    Read in Roman CGI filter transmission curves from data files

    Parameters
    ----------
    filepath : str
        Path to directory containing filter files
    filter_name : str
        Name of filter to read (e.g., '1F', '2F', '3F', '4F')
    
    Returns
    -------
    wavel : np.array
        Wavelength array of filter transmission curve
    transmission : np.array
        Transmission values of filter curve
    """
    filter_files = [f for f in os.listdir(filepath) if '.csv' in f and filter_name in f]

    if len(filter_files) == 0:
        raise FileNotFoundError(f"No filter files found for filter name '{filter_name}' in directory '{filepath}'.")
    
    wavel, transmission = np.loadtxt(filepath + filter_files[0], delimiter=',', skiprows=4, unpack=True)

    return wavel, transmission


def FpFs_band(FpFs_spectrum, lambda_planet, filter_transmission, filter_wavelengths):
    """
    Calculate the band-averaged flux given a spectrum in Fp/Fs and filter transmission

    Parameters
    ----------
    FpFs_spectrum : np.array
        Array of Fp/Fs values from FpFs_planet or other
    wavelength : np.array
        Wavelength array corresponding to FpFs_spectrum
    filter_transmission : np.array
        Transmission values of the filter
    filter_wavelengths : np.array
        Wavelength array corresponding to filter_transmission

    Returns
    -------
    float
        Band-averaged Fp/Fs value   
    """

    # Get spectrum only in wavelength regions of filter transmission
    filter_spec = FpFs_spectrum[(lambda_planet >= filter_wavelengths.min()) & (lambda_planet <= filter_wavelengths.max())]
    planet_filt_lambda = lambda_planet[(lambda_planet >= filter_wavelengths.min()) & (lambda_planet <= filter_wavelengths.max())]

    # Interpolate filter transmission to match spectrum wavelengths
    filt_transmission_interp = np.interp(planet_filt_lambda, filter_wavelengths, filter_transmission)

    # Calculate band-averaged flux
    FpFs_band = np.trapz(filter_spec * filt_transmission_interp, planet_filt_lambda) / np.trapz(filt_transmission_interp, planet_filt_lambda)

    return FpFs_band

def separation_AU(a, e, f):
    '''
    Calculate separation of planet from star at a given epoch from semi-major
    axis, eccentricity, and true anomaly.

    Parameters
    ----------
    a : float
        Semi-major axis in AU
    e : float
        Eccentricity
    f : float
        True anomaly in radians

    Returns
    -------
    float        Separation in AU
    '''
    s = a * (1 - e**2) / (1 + e * np.cos(f))

    return s

def phase_angle(I, theta):
    '''
    Calculate planet phase angle 
    Assumes observer located in -z direction of reference frame

    Parameters
    ----------
    I : float
        Inclination in rad or with astropy u.deg
    theta : float
        Argument of latitude (true anomaly + arg. of periastron)

    Returns
    -------
    float
        Phase angle in radians
    '''
    return np.arccos( np.sin(I) * np.sin(theta) )

def read_illumination_file(file_path, rows_to_skip=14):
    """
    Read ringed_planet_illumination.py output file with headers for phase, true anomaly, 
    and fraction of planet/ring illuminated.

    Parameters
    ----------
    file_path : str
        Full path to file with name "planet_illumination_e{}.txt"
    rows_to_skip : int
        Number of header rows to skip in file before data starts (default=13)

    Returns
    -------
    np.arrays
        Arrays of phase, true anomaly, and fraction illuminated.

    """
    phases, true_anomaly, fraction_lit = np.loadtxt(file_path, delimiter=",", skiprows=rows_to_skip, unpack=True)
    return phases, true_anomaly, fraction_lit

def apply_litFraction_to_spectrum(phases, true_anom, lit_fraction, albedo_spectrum):
    """
    Apply fraction of illumination to a spectrum. 
    lit_fraction is equivalent to a phase function here.

    Parameters
    ----------
    phases : np.ndarray
        Phase angles corresponding to each fraction of illumination in degrees
    true_anom : np.ndarray
        True anomalies corresponding to each fraction of illumination in degrees
    lit_fraction : np.ndarray
        Fraction of the planet/ring illuminated corresponding to each phase angle
    albedo_spectrum : np.ndarray 
        Albedo spectrum of the planet/ring at full phase.

    Returns
    -------
    dict
        Dictionary with phase angles as keys and spectra (albedo_spectrum multiplied by lit_fraction) as values.
    """

    # Make sure all values in the phase angle array are positive
    phases = np.abs(phases)
    # Sort arrays by phase angle
    sorted_lists = sorted(zip(phases, true_anom, lit_fraction))
    phases, true_anom, lit_fraction = zip(*sorted_lists)

    spectra_per_phase, spectra_per_trueanom = {}, {}
    
    for i, lit_frac in enumerate(lit_fraction):
        new_spectrum = lit_frac * albedo_spectrum # Apply lit fraction to albedo spectrum
        # lit_fraction is equivalent to a phase function
        
        # Save new spectrum in dictionary with phase angle as key
        spectra_per_phase[i] = new_spectrum
        spectra_per_trueanom[i] = new_spectrum

    return spectra_per_phase, spectra_per_trueanom

def save_spectra_per_phase_to_file(spectra_per_phase_dict, wavelength_arr, filename):
    """
    Save spectra as a function of phase angle to output file with matching
    wavelength array.

    Parameters
    ----------
    spectra_per_phase_dict : dict
        Dictionary containing phase angles as keys and spectra as values.
    wavelength_arr : np.ndarray
        Array of wavelengths corresponding to the spectra.
    filename : str
        Full path to the output file.
    """
    with open(filename, 'w') as f:
        # Write header row
        header = "wavelength"
        for phase in spectra_per_phase_dict.keys():
            header += f",{phase:.1f}"
        f.write(header + "\n")

        # Write data rows
        for i in range(len(wavelength_arr)):
            row = f"{wavelength_arr[i]}"
            for spectra in spectra_per_phase_dict.values():
                row += f",{spectra[i]}"
            f.write(row + "\n")

    print(f"Spectra per phase saved to {filename}")

def read_spectra_per_phase_file(file_path):
    """
    Read in wavelength array, phase angles, and spectra as a function of phase angle
    from files produced by save_spectra_per_phase_to_file().

    Parameters
    ----------
    file_path : str
        Full path to file containing file name

    Returns
    -------
    dict
        Dictionary containing phase angles as keys
        and spectra as values. 
    """

    data = pd.read_csv(file_path)
    wavelength = data['wavelength']
    phases = data.columns[1:]
    spectra_per_phase_dict = {}
    for phase in data.columns[1:]:
        spectra_per_phase_dict[float(phase)] = data[phase]

    return spectra_per_phase_dict, wavelength, phases

def read_stellar_spectrum(file_path, wavel_units=u.Angstrom, spec_units=u.erg/u.s/u.cm**2/u.Angstrom):
    """
    Reads in stellar spectrum and returns wavelength and flux arrays.

    Parameters
    ----------
    file_path : str
        Full path to file containing wavelength array and spectrum. 
    wavel_units : astropy.units, optional
        Units of wavelength in the input file. Default is Angstroms.

    Returns
    -------
    wavelength : np.ndarray
        Array of wavelengths.
    flux : np.ndarray
        Array of flux values corresponding to the wavelengths.
    """
    star_wavel, star_flux = np.loadtxt(file_path, unpack=True)

    # Convert wavelength to Angstroms if necessary
    if wavel_units != u.Angstrom:
        star_wavel = (star_wavel * wavel_units).to(u.Angstrom).value
    else:
        star_wavel *= wavel_units

    star_flux *= spec_units

    return star_wavel, star_flux

def calc_flux_in_band(filter_file, albedo_spectrum, wavelength_arr, stellar_spectrum, 
                      stellar_wavelength, separation, Rp, file_skiprows=3):
    """
    Compute the band-average flux of a spectrum given a filter transmission curve.
    All wavelengths must be in Angstroms.
    TODO: update this to take in only spectrum in units of flux, not albedo space

    Parameters
    ----------
    filter_file : str
        Path to the filter transmission curve file (CSV format).
    albedo_spectrum : np.ndarray
        Array of spectral values (e.g., albedo or I/F) corresponding to the wavelength array.
    wavelength_arr : np.ndarray
        Array of wavelengths corresponding to the spectrum_arr.
    stellar_spectrum : np.ndarray
        Array of the stellar flux values corresponding to the stellar_wavelength array.
        Should be in units of flux (e.g., W/m2/nm)
    stellar_wavelength : np.ndarray
        Array of wavelengths corresponding to the stellar_spectrum.
    separation : astropy Quantity
        The distance between the star and the planet (e.g., in AU or km).
    Rp : astropy Quantity
        The radius of the planet (e.g., in km).
    file_skiprows : int, optional
        Number of rows to skip at the beginning of the filter file (default is 3 for
        Roman filter files).

    Returns
    -------
    F_band : float
        The band-averaged flux of the spectrum through the filter.
    fpfs: np.ndarray
        The planet-to-star flux ratio spectrum (Fp/Fs) across the input wavelength array.

    """    
    df_band = pd.read_csv(filter_file, skiprows=file_skiprows, sep=',')
    wavel_band = (df_band['lambda_nm'].to_numpy() * u.nm).to(u.Angstrom)
    transmission_band = df_band['%T'].to_numpy()
    
    # Scale stellar spectrum for distance of planet from star
    separation_km = separation.to(u.km) # units of km
    star_spec_atplanet = stellar_spectrum / separation_km.value**2
    
    # Interpolate stellar spectrum on object wavelength grid
    stellar_flux_at_planet = np.interp(wavelength_arr, stellar_wavelength, star_spec_atplanet)

    # Convert source spectrum from albedo to flux space
    target_flux = albedo_spectrum * stellar_flux_at_planet * (Rp / separation_km)**2 # Ag(lambda) * F_star(lambda) * (Rp / r)**2
    # fpfs = target_flux / stellar_flux_at_planet                                   # Fp/Fs
    
    # Get wavelength range of filter
    min_lambda, max_lambda = np.min(wavel_band), np.max(wavel_band)

    # Select spectrum within filter band
    flux_in_band = target_flux[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]
    wavel_in_band = wavelength_arr[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]

    # Interpolate filter wavelegth grid to match spectrum wavelength grid
    band_transmission_interp = np.interp(wavel_in_band, wavel_band, transmission_band)

    # Calculate band-averaged flux
    F_band = np.trapezoid(flux_in_band * band_transmission_interp, wavel_in_band) / np.trapezoid(band_transmission_interp, wavel_in_band) 

    # ------ Get stellar flux in filter to calculate Fp/Fs --------
    stellar_flux_in_band = stellar_flux_at_planet[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]
    # Calculate band-averaged flux for star
    F_star_band = np.trapezoid(stellar_flux_in_band * band_transmission_interp, wavel_in_band) / np.trapezoid(band_transmission_interp, wavel_in_band)

    # Calculate Fp/Fs in band
    Fp_Fs_band = F_band / F_star_band

    return F_band, Fp_Fs_band

def flux_in_band_from_fluxSpec(filter_file, flux_spectrum, wavelength_arr, stellar_spectrum,
                               stellar_wavelength, separation, file_skiprows=3):
    """
    Calculate flux within a Roman bandpass given the target's spectrum in flux units (erg/s/cm2/A)
    and the stellar flux spectrum.

    Parameters
    ----------
    filter_file : str
        Full path to Roman filter file for either bands 1-4
    flux_spectrum : np.ndarray
        Flux spectrum of the target in units of erg/s/cm2/A
    wavelength_arr : np.ndarray
        Array of wavelengths corresponding to the flux_spectrum.
    stellar_spectrum : np.ndarray
        Array of the stellar flux values corresponding to the stellar_wavelength array.
        Units of erg/s/cm2/A
    stellar_wavelength : np.ndarray
        Array of wavelengths corresponding to the stellar_spectrum.
    separation : astropy.units.Quantity
        Distance between the target and the star in units of AU.
    file_skiprows : int, optional
        Number of rows to skip at the beginning of the filter file (default is 3 for
        Roman filter files).

    Returns
    -------
    F_band : float
        The band-averaged flux of the spectrum through the filter.
    fpfs: np.ndarray
        The planet-to-star flux ratio spectrum (Fp/Fs) in the bandpass of interest.

    """

    # Read in filter file
    df_band = pd.read_csv(filter_file, skiprows=file_skiprows, sep=',')
    wavel_filter = (df_band['lambda_nm'].to_numpy() * u.nm).to(u.Angstrom)
    transmission_band = df_band['%T'].to_numpy()

    # Get wavelength range of filter 
    min_lambda, max_lambda = np.min(wavel_filter), np.max(wavel_filter)

    # Scale stellar spectrum for distance of target from star
    separation_km = separation.to(u.km) # units of km
    star_spec_at_target = stellar_spectrum / separation_km.value**2
    # Interpolate stellar spectrum on object wavelength grid
    stellar_flux_at_target = np.interp(wavelength_arr, stellar_wavelength, star_spec_at_target)

    # Select spectrum within filter band
    flux_in_band = flux_spectrum[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]
    wavel_in_band = wavelength_arr[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]

    # Interpolate filter wavelegth grid to match spectrum wavelength grid
    band_transmission_interp = np.interp(wavel_in_band, wavel_filter, transmission_band)

    # Calculate band-averaged flux
    F_band = np.trapezoid(flux_in_band * band_transmission_interp, wavel_in_band) / np.trapezoid(band_transmission_interp, wavel_in_band)

    # Get stellar flux in filter to calculate Fp/Fs
    stellar_flux_in_band = stellar_flux_at_target[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]
    F_star_band = np.trapezoid(stellar_flux_in_band * band_transmission_interp, wavel_in_band) / np.trapezoid(band_transmission_interp, wavel_in_band)

    # Calculate Fp/Fs in band
    Fp_Fs_band = F_band / F_star_band

    return F_band, Fp_Fs_band

def calc_albedo_in_band(filter_file, albedo_spectrum, wavelength_arr, file_skiprows=3):
    """
    Calculate albedo in Roman filter of interest. 

    Parameters
    ----------
    filter_file : str
        Full path to Roman filter file for either band 1-4
    albedo_spectrum : np.ndarray
        Albedo spectrum of object
    wavelength_arr : np.ndarray
        Array of wavelengths corresponding to the albedo_spectrum.
    file_skiprows : int, optional
        Number of rows to skip at the beginning of the filter file (default is 3 for
        Roman filter files).

    Returns
    -------
    A_band : float
        Band-averaged albedo of the object in the filter of interest.
    """

    df_band = pd.read_csv(filter_file, skiprows=file_skiprows, sep=',')
    wavel_band = (df_band['lambda_nm'].to_numpy() * u.nm).to(u.Angstrom)
    transmission_band = df_band['%T'].to_numpy()

    # Get wavelength range of filter
    min_lambda, max_lambda = np.min(wavel_band), np.max(wavel_band)

    # Select spectrum within filter band
    spec_in_band = albedo_spectrum[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]
    wavel_in_band = wavelength_arr[(wavelength_arr >= min_lambda) & (wavelength_arr <= max_lambda)]

    # Interpolate filter wavelegth grid to match spectrum wavelength grid
    band_transmission_interp = np.interp(wavel_in_band, wavel_band, transmission_band)

    # Calculate band-averaged albedo
    A_band = np.trapezoid(spec_in_band * band_transmission_interp, wavel_in_band) / np.trapezoid(band_transmission_interp, wavel_in_band)

    return A_band

def convert_planet_albedo_to_flux(albedo_spectrum, albedo_wavel, stellar_spectrum, 
                                  stellar_wavel, separation, Rp):
    """
    Convert planet albedo spectrum to flux space.

    Parameters
    ----------
    albedo_spectrum : np.ndarray
        Albedo spectrum of the planet. 
    albedo_wavel : np.ndarray
        Wavelength array corresponding to albedo_spectrum.
    stellar_spectrum : np.ndarray
        Spectrum of host star in flux units (e.g., W/m2/nm) corresponding to stellar_wavel
    stellar_wavel : np.ndarray
        Wavelength array corresponding to stellar_spectrum.
    separation : astropy.units.Quantity
        Distance between the planet and the star.
    Rp : astropy.units.Quantity
        Radius of the planet in km
    Returns
    -------
    target_flux : np.ndarray
        Flux spectrum of the planet in the same units as the stellar_spectrum.
    """
    # Scale stellar spectrum for distance of planet from star
    separation_km = separation.to(u.km) # units of km
    star_spec_atplanet = stellar_spectrum / separation_km.value**2

    # Interpolate stellar spectrum on object wavelength grid
    stellar_flux_at_planet = np.interp(albedo_wavel, stellar_wavel, star_spec_atplanet)

    # Convert source spectrum from albedo to flux space
    target_flux = albedo_spectrum * stellar_flux_at_planet * (Rp / separation_km)**2 

    FpFs = target_flux / stellar_flux_at_planet

    return target_flux, FpFs, stellar_flux_at_planet

def convert_planet_albedo_to_flux_wRingShadowing(albedo_spectrum, albedo_wavel, frac_planet_illuminated, 
                                                 stellar_spectrum, stellar_wavel, separation, Rp):
    """
    Convert planet albedo spectrum to flux space.

    Parameters
    ----------
    albedo_spectrum : np.ndarray
        Albedo spectrum of the planet. 
    albedo_wavel : np.ndarray
        Wavelength array corresponding to albedo_spectrum.
    frac_planet_illuminated : np.ndarray
        Fraction of the planet illuminated at each phase angle w.r.t. observer, accounting for ring shadowing.
    stellar_spectrum : np.ndarray
        Spectrum of host star in flux units (e.g., W/m2/nm) corresponding to stellar_wavel
    stellar_wavel : np.ndarray
        Wavelength array corresponding to stellar_spectrum.
    separation : astropy.units.Quantity
        Distance between the planet and the star.
    Rp : astropy.units.Quantity
        Radius of the planet in km
    Returns
    -------
    target_flux : np.ndarray
        Flux spectrum of the planet in the same units as the stellar_spectrum.
    """
    # Scale stellar spectrum for distance of planet from star
    separation_km = separation.to(u.km) # units of km
    star_spec_atplanet = stellar_spectrum / separation_km.value**2

    # Interpolate stellar spectrum on object wavelength grid
    stellar_flux_at_planet = np.interp(albedo_wavel, stellar_wavel, star_spec_atplanet)

    # Convert source spectrum from albedo to flux space
    target_flux = albedo_spectrum * frac_planet_illuminated * stellar_flux_at_planet * (Rp / separation_km)**2 

    FpFs = target_flux / stellar_flux_at_planet

    return target_flux, FpFs, stellar_flux_at_planet

def convert_ring_albedo_to_flux(albedo_spectrum, albedo_wavel, stellar_spectrum, 
                                stellar_wavel, separation, R_inner, R_outer, inc_rad):
    """
    Convert ring albedo spectrum (I/F) to flux space.

    Parameters
    ----------
    albedo_spectrum : np.ndarray
        Albedo spectrum of the rings.
    albedo_wavel : np.ndarray
        Wavelength array corresponding to albedo_spectrum.
    stellar_spectrum : np.ndarray
        Spectrum of host star in flux units (e.g., W/m2/nm) corresponding to stellar_wavel
    stellar_wavel : np.ndarray
        Wavelength array corresponding to stellar_spectrum.
    separation : astropy.units.Quantity
        Distance between the planet and the star in km
    R_inner : astropy.units.Quantity
        Inner radius of the ring in km
    R_outer : astropy.units.Quantity
        Outer radius of the ring in km
    inc_rad : astropy.units.Quantity
        Inclination of the ring in radians (obliquity)

    Returns
    -------
    target_flux : np.ndarray
        Flux spectrum of the rings in the same units as the stellar_spectrum.
    """
    
    # Scale stellar spectrum for distance of planet from star
    separation_km = separation.to(u.km) # units of km
    star_spec_at_target = stellar_spectrum / separation_km.value**2

    # Interpolate stellar spectrum on object wavelength grid
    stellar_flux_at_target = np.interp(albedo_wavel, stellar_wavel, star_spec_at_target)

    # Convert source spectrum from albedo to flux space
    target_flux = albedo_spectrum * stellar_flux_at_target * ( (R_outer**2 - R_inner**2) / separation_km**2 ) * np.cos(inc_rad) # albedo * F_star * (A_ring,projected / r**2)

    FpFs = target_flux / stellar_flux_at_target

    return target_flux, FpFs, stellar_flux_at_target

def convert_flux_to_albedo(flux_spectrum, flux_wavel, stellar_spectrum, 
                           stellar_wavel, separation, R_system):
    """
    Convert the flux spectrum in units of erg/s/cm^2/Angstrom to the albedo spectrum
    as a function of phase angle.

    Parameters
    ----------
    flux_spectrum : np.ndarray
        Flux spectrum of the target in units of erg/s/cm^2/Angstrom.
    flux_wavel : np.ndarray
        Wavelength array corresponding to flux_spectrum.
    stellar_spectrum : np.ndarray
        Spectrum of the host star in units of erg/s/cm^2/Angstrom.
    stellar_wavel : np.ndarray
        Wavelength array corresponding to stellar_spectrum.
    separation : astropy.units.Quantity
        Distance between the planet and the star in km.
    R_system : astropy.units.Quantity
        Radius of the planetary system in km.

    Returns
    -------
    albedo_spectrum : np.ndarray
        Albedo spectrum of the target as a function of phase angle.
    """
    
    # Scale stellar spectrum for distance of planet from star
    separation_km = separation.to(u.km) # units of km
    star_spec_at_target = stellar_spectrum / separation_km.value**2
    
    # Interpolate stellar spectrum on object wavelength grid
    stellar_flux_at_target = np.interp(flux_wavel, stellar_wavel, star_spec_at_target)

    # Get Fp/Fs(phi)
    FpFs = flux_spectrum / stellar_flux_at_target 

    # Convert flux spectrum to albedo spectrum
    # Ag = Fp/Fs(phi) * (r / Rp)**2 
    albedo_spectrum = FpFs * (separation_km / R_system)**2

    return albedo_spectrum, FpFs

def sort_trueanom_flux(df):
    """
    Sort a dataframe of true anomaly and flux ratio values 
    by true anomaly. Usually used for plotting purposes.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe generated from a combined phase curve file.
    """
    sorted_ids = np.argsort(df["true_anomaly"])
    true_anom_sorted = df["true_anomaly"][sorted_ids]
    fpfs_sorted = df["fpfs_ringedplanet"][sorted_ids]
    return true_anom_sorted, fpfs_sorted, sorted_ids

