# Physical constants
const gas_constant  = 8.3144598         # Universal gas constant (J/mol/K)
const light_speed   = 2.99792458e8      # Speed of light in vacuum (m/s)
const h_Planck      = 6.626e-34         # Planck constant (m^2 kg/s)
const k_Boltzmann   = 1.381e-23         # Boltzmann constant (m^2 kg/s^2/K)
const Stefan        = 5.670e-8       	# Stefan-Boltzmann constant (W/m^2/K^4)
const AU            = 1.495978707e11    # Astronomical unit (m)

# Properties of dry air
const molmass_air   = 28.97e-3          # Molecular weight dry air (kg/mol)
const R_d           = gas_constant/
        molmass_air                     # Gas constant dry air (J/kg/K)
const kappa_d       = 2/7               # Adiabatic exponent dry air
const cp_d          = R_d/kappa_d       # Isobaric specific heat dry air

# Properties of water
const dens_liquid   = 1e3               # Density of liquid water (kg/m^3)
const molmass_water = 18.01528e-3       # Molecular weight (kg/mol)
const R_v           = gas_constant/
        molmass_water                   # Gas constant water vapor (J/kg/K)
const cp_v          = 1859              # Isobaric specific heat vapor (J/kg/K)
const cp_l          = 4181              # Isobaric specific heat liquid (J/kg/K)
const cp_i          = 2100              # Isobaric specific heat ice (J/kg/K)
const Tfreeze 	    = 273.15            # Freezing point temperature (K)
const T0      	    = 273.16            # Reference temperature [triple point] (K)
const L_v0          = 2.5008e6          # Latent heat vaporization at T0 (J/kg)
const L_s0          = 2.8341e6          # Latent heat sublimation at T0 (J/kg)
const L_f0          = L_s0 - L_v0       # Latent heat of fusion at T0 (J/kg)

# Properties of sea water
const dens_ocean    = 1.035e3           # Reference density sea water (kg/m^3)
const cp_ocean      = 3989.25           # Specific heat sea water (J/kg/K)

# Planetary parameters
const planet_radius = 6.371e6           # Mean planetary radius (m)
const day           = 86400             # Length of day (s)
const Omega         = 7.2921159e-5      # Ang. velocity planetary rotation (1/s)
const grav          = 9.81              # Gravitational acceleration (m/s^2)
const year_anom     = 365.26*day        # Length of anomalistic year (s)
const orbit_semimaj = 1*AU              # Length of semimajor orbital axis (m)
const TSI           = 1362              # Total solar irradiance (W/m^2)
const mslp          = 1.01325e5         # Mean sea level pressure (Pa)
