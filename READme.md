# exorings

A forward model for simulating the illumination, shadowing, and reflected-light phase curves of ringed exoplanets.

## Overview

<img src="docs/images/ringed_planet.gif"
     alt="Animation of the ringed-exoplanet illumination simulation"
     align="right"
     width="300">

`exorings` is a Python-based modeling framework designed to investigate the observable signatures of planetary rings in directly imaged exoplanet systems. The code computes the illumination state of a ringed planet throughout its orbit while accounting for:

- Planet illumination by the host star
- Ring illumination by the host star
- Mutual shadowing between the planet and rings
- Observer visibility of illuminated regions
- Orbital geometry
- Ring orientation and obliquity
- Wavelength-dependent reflected-light spectra
- Instrument bandpasses (currently: Roman Coronagraph filters)

The code produces broadband and spectrally resolved planet-star flux ratios as a function of orbital phase.

<br clear="right">


## Installation

Clone the repository:

```bash
git clone https://github.com/snhasler/exorings.git
cd exorings
```

Install required dependencies:

```bash
pip install numpy scipy matplotlib pandas astropy
```


## Main Components

### `calc_ringed_planet_illumination.py`

Primary illumination and shadowing simulation script.

This script computes the fraction of the planet and rings that are:

- Illuminated by the host star
- Visible to the observer
- Mutually shadowed by the rings/planet

The simulation is performed over a user-defined set of orbital epochs and produces illumination-fraction files that are later converted into observable phase curves by `illumination2flux.py`.

#### Outputs

```text
planet_illumination_*.txt
ring_illumination_*.txt
```


### `illumination2flux.py`

Converts illumination fractions into observable spectra and phase curves.

For each orbital epoch, the script:

1. Reads illumination-fraction files.
2. Applies illumination fractions to planetary and ring albedo spectra.
3. Applies a Lambertian phase function to the planet.
4. Converts albedo spectra into reflected flux spectra.
5. Combines planetary and ring contributions.
6. Calculates planet-star flux ratios.
7. Computes band-integrated values for Roman filters.
8. Generates phase-curve outputs and plots.

#### Outputs

- Spectral flux ratios
- Broadband phase curves
- Comparison plots between different system geometries (if multiple geometries are passed)

### `geometry.py`

Orbital and ring geometry module.

Provides functions for: (1) Orbital coordinate generation, (2) coordinate transformations, (3) rotation matrices, (4) angular momentum vectors, (5) ring orientation calculations, (6) observer and star viewing geometry

This module defines the geometric framework used throughout the simulations.


### `ray_casting.py`

Contains ray-intersection methods used to determine illumination and shadowing.

Provides functions for determining: (1) planet self-shadowing, (2) planet shadowing of the rings, (3) ring shadowing of the planet, (4) ring occulatations (of the planet)


### `utils.py`

Collection of useful functions used throughout.

Examples include: (1) Reading Roman filter transmission curves, (2) Reading stellar spectra, (3) Loading albedo spectra, (4) Bandpass integration, (5) Flux conversions

---

## Simulation Workflow

### Step 1: Compute Illumination and Shadowing of Ringed Planet System

Update desired system parameters and run:

```bash
python calc_ringed_planet_illumination.py
```

### Step 2: Generate Spectra and Phase Curves

Update desired system paramters and run:

**Note**: The planet and ring parameters that you run here should match the run parameters in Step 1.

```bash
python illumination2flux.py
```

- **Input Data**

  - Planet Spectrum : The default planetary spectrum is a Jupiter geometric albedo spectrum based on: [Karkoschka (1998)](https://ui.adsabs.harvard.edu/abs/1998Icar..133..134K/abstract)

      Located in: ```spectra/planet/```

 - **Ring Spectrum**

    - The default ring spectrum is derived from Cassini VIMS observations of Saturn's rings: [Hedman et al. (2013)](https://ui.adsabs.harvard.edu/abs/2013Icar..223..105H/abstract)

      Located in: ```spectra/rings/```


 - **Stellar Spectrum**
    - The default spectrum is a Solar spectrum. 

      Located in: ```spectra/star/```


  - **Roman Coronagraph Instrument Filters**
    - Roman filter curves were retreived from the [Roman Science Support Center at IPAC at Caltech](https://roman.ipac.caltech.edu/page/additional-coronagraph-instrument-parameters-model-and-data-html#Color_Filter_Curves).
    
      Located in: ```filters/roman/```

    - Currently supported filters in the code include: 1F, 2F, 3F, 4F, 3C, 4C 


<!-- ## Assumptions and Current Limitations

Current implementation assumptions include:

- Lambertian scattering for the planetary surface
- Optically thick rings
- Single-scattering reflected-light treatment
- Static ring geometry
- No wavelength-dependent ring transmission

Future releases may include:

- Non-Lambertian planetary scattering
- Expansion to include a wide range of ring optical depths/compositions
- Wavelength-dependent transmission through rings
- Varying ring components/scattering properties
- Additional instrument bandpasses

--- -->

## Citation

*If you use this code, please cite:*

```text
Hasler, S. N., Greenbaum, A. Z., Bryden, G., Bailey, V. B., Llop-Sayson, J., Lane, E., Limbach, M. A., 
Pearce, L., Ingalls, J., and Lowrance, P. (submitted)
```

*and the spectral datasets used by the model:*


Karkoschka, E. (1998). Methane, ammonia, and temperature measurements of the Jovian planets and Titan from CCD–spectrophotometry. [Icarus, 133(1), 134-146.](https://ui.adsabs.harvard.edu/abs/1998pds..data...82K/abstract)

Hedman, M. M., Nicholson, P. D., Cuzzi, J. N., Clark, R. N., Filacchione, G., Capaccioni, F., & Ciarniello, M. (2013). Connections between spectra and structure in Saturn’s main rings based on Cassini VIMS data. [Icarus, 223(1), 105-130.](https://ui.adsabs.harvard.edu/abs/2013Icar..223..105H/abstract)


## Author

**Samantha Hasler**, [shasler@stsci.edu](mailto:shasler@stsci.edu)  
Space Telescope Science Institute (STScI)


## License

[MIT](https://choosealicense.com/licenses/mit/)