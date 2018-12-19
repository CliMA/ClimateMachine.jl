"""
    Parameter{sym} <: Irrational{sym}

Number type representing a constant parameter value denoted by the symbol `sym`.
"""
struct Parameter{sym} <: Base.AbstractIrrational end

Base.show(io::IO, x::Parameter{S}) where {S} = print(io, "$S = $(string(x))")

==(::Parameter{s}, ::Parameter{s}) where {s} = true
<(::Parameter{s}, ::Parameter{s}) where {s} = false
<=(::Parameter{s}, ::Parameter{s}) where {s} = true
hash(x::Parameter, h::UInt) = 3*objectid(x) - h
widen(::Type{T}) where {T<:Parameter} = T
round(x::Parameter, r::RoundingMode) = round(float(x), r)

"""
    @parameter sym val [units=()]
    @parameter sym val [units=()]
    @parameter(sym, val, [units=()])

Define a new `Parameter` value, `sym`, with value `val`. The units of the
parameter can be specified with the optional string 'units'
"""
macro parameter(sym, val, units=())
  esym = esc(sym)
  qsym = esc(Expr(:quote, sym))

  if ~isempty(units)
    units = " [$units]"
  else
    units = ""
  end

  quote
    const $esym = Parameter{$qsym}()
    Base.BigFloat(::Parameter{$qsym}) = $(esc(BigFloat(eval(val))))
    Base.Float64(::Parameter{$qsym}) = $(esc(Float64(eval(val))))
    Base.Float32(::Parameter{$qsym}) = $(esc(Float32(eval(val))))
    Base.string(::Parameter{$qsym}) = $(esc(string(eval(val), "$units")))
  end
end

# Physical constants
@parameter gas_constant 8.3144598     "J/mol/K"
@parameter light_speed  2.99792458e8  "m/s"
@parameter h_Planck     6.626e-34     "m^2 kg/s"
@parameter k_Boltzmann  1.381e-23     "m^2 kg/s^2/K"
@parameter Stefan       5.670e-8      "W/m^2/K^4"
@parameter astro_unit   1.4959787e11  "m"

# Properties of dry air
@parameter molmass_air 28.97e-3                 "kg/mol"
@parameter R_d         gas_constant/molmass_air "J/kg/K"
@parameter kappa_d     2//7
@parameter cp_d        R_d/kappa_d

# Properties of water
@parameter dens_liquid        1e3                        "kg/m^3"
@parameter molmass_water      18.01528e-3                "kg/mol"
@parameter R_v                gas_constant/molmass_water "J/kg/K"
@parameter cp_v               1859                       "J/kg/K"
@parameter cp_l               4181                       "J/kg/K"
@parameter cp_i               2100                       "J/kg/K"
@parameter Tfreeze            273.15                     "K"
@parameter T0                 273.16                     "K"
@parameter L_v0               2.5008e6                   "J/kg"
@parameter L_s0               2.8341e6                   "J/kg"
@parameter L_f0               L_s0 - L_v0                "J/kg"
@parameter sat_vapor_press_0  611.657                    "Pa"

# Properties of sea water
@parameter dens_ocean         1.035e3 "kg/m^3"
@parameter cp_ocean           3989.25 "J/kg/K"

# Planetary parameters
@parameter planet_radius      6.371e6       "m"
@parameter day                86400         "s"
@parameter Omega              7.2921159e-5  "1/s"
@parameter grav               9.81          "m/s^2"
@parameter year_anom          365.26*day    "s"
@parameter orbit_semimaj      1*astro_unit  "m"
@parameter TSI                1362          "W/m^2"
@parameter mslp               1.01325e5     "Pa"

#=
These are the docs for the parameters so that user can do something like:
```
julia> ?gas_constant
```
=#

# Physical constants
"""
    gas_constant

Universal gas constant (J/mol/K)

# Examples
```jldoctest
julia> gas_constant
gas_constant = 8.3144598 [J/mol/K]
```
"""
gas_constant

"""
    light_speed

Speed of light in vacuum (m/s)

# Examples
```jldoctest
julia> light_speed
light_speed = 2.99792458e8 [m/s]
```
"""
light_speed

"""
    h_Planck

Planck constant (m^2 kg/s)

# Examples
```jldoctest
julia> h_Planck
h_Planck = 6.626e-34 [m^2 kg/s]
```
"""
h_Planck

"""
    k_Boltzmann

Boltzmann constant (m^2 kg/s^2/K)

# Examples
```jldoctest
julia> k_Boltzmann
k_Boltzmann = 1.381e-23 [m^2 kg/s^2/K]
```
"""
k_Boltzmann

"""
    Stefan

Stefan-Boltzmann constant (W/m^2/K^4)

# Examples
```jldoctest
julia> Stefan
Stefan = 5.67e-8 [W/m^2/K^4]
```
"""
Stefan

"""
    astro_unit

Astronomical unit (m)

# Examples
```jldoctest
julia> astro_unit
astro_unit = 1.4959787e11 [m]
```
"""
astro_unit


# Properties of dry air
"""
    molmass_air

Molecular weight dry air (kg/mol)

# Examples
```jldoctest
julia> molmass_air
molmass_air = 0.02897 [kg/mol]
```
"""
molmass_air

"""
    R_d

Gas constant dry air (J/kg/K)

# Examples
```jldoctest
julia> R_d
R_d = 287.0024093890231 [J/kg/K]
```
"""
R_d

"""
    kappa_d

Adiabatic exponent dry air

# Examples
```jldoctest
julia> kappa_d
kappa_d = 2//7
```
"""
kappa_d

"""
    cp_d

Isobaric specific heat dry air

# Examples
```jldoctest
julia> cp_d
cp_d = 1004.5084328615809
```
"""
cp_d


# Properties of water
"""
    dens_liquid

Density of liquid water (kg/m^3)

# Examples
```jldoctest
julia> dens_liquid
dens_liquid = 1000.0 [kg/m^3]
```
"""
dens_liquid

"""
    molmass_water

Molecular weight (kg/mol)

# Examples
```jldoctest
julia> molmass_water
molmass_water = 0.01801528 [kg/mol]
```
"""
molmass_water

"""
    R_v

Gas constant water vapor (J/kg/K)

# Examples
```jldoctest
julia> R_v
R_v = 461.52265188217996 [J/kg/K]
```
"""
R_v

"""
    cp_v

Isobaric specific heat vapor (J/kg/K)

# Examples
```jldoctest
julia> cp_v
cp_v = 1859 [J/kg/K]
```
"""
cp_v

"""
    cp_l

Isobaric specific heat liquid (J/kg/K)

# Examples
```jldoctest
julia> cp_l
cp_l = 4181 [J/kg/K]
```
"""
cp_l

"""
    cp_i

Isobaric specific heat ice (J/kg/K)

# Examples
```jldoctest
julia> cp_i
cp_i = 2100 [J/kg/K]
```
"""
cp_i

"""
    Tfreeze

Freezing point temperature (K)

# Examples
```jldoctest
julia> Tfreeze
Tfreeze = 273.15 [K]
```
"""
Tfreeze

"""
    T0

Triple point temperature (K)

# Examples
```jldoctest
julia> T0
T0 = 273.16 [K]
```
"""
T0

"""
    L_v0

Latent heat vaporization at T0 (J/kg)

# Examples
```jldoctest
julia> L_v0
L_v0 = 2.5008e6 [J/kg]
```
"""
L_v0

"""
    L_s0

Latent heat sublimation at T0 (J/kg)

# Examples
```jldoctest
julia> L_s0
L_s0 = 2.8341e6 [J/kg]
```
"""
L_s0

"""
    L_f0

Latent heat of fusion at T0 (J/kg)

# Examples
```jldoctest
julia> L_f0
L_f0 = 333300.0 [J/kg]
```
"""
L_f0

"""
    sat_vapor_press_0

Saturation vapor pressure at T0 (Pa)

# Examples
```jldoctest
julia> sat_vapor_press_0
sat_vapor_press_0 = 611.657 [Pa]
```
"""
sat_vapor_press_0


# Properties of sea water
"""
    dens_ocean

Reference density sea water (kg/m^3)

# Examples
```jldoctest
julia> dens_ocean
dens_ocean = 1035.0 [kg/m^3]
```
"""
dens_ocean

"""
    cp_ocean

Specific heat sea water (J/kg/K)

# Examples
```jldoctest
julia> cp_ocean
cp_ocean = 3989.25 [J/kg/K]
```
"""
cp_ocean


# Planetary parameters
"""
    planet_radius

Mean planetary radius (m)

# Examples
```jldoctest
julia> planet_radius
planet_radius = 6.371e6 [m]
```
"""
planet_radius

"""
    day

Length of day (s)

# Examples
```jldoctest
julia> day
day = 86400 [s]
```
"""
day

"""
    Omega

Ang. velocity planetary rotation (1/s)

# Examples
```jldoctest
julia> Omega
Omega = 7.2921159e-5 [1/s]
```
"""
Omega

"""
    grav

Gravitational acceleration (m/s^2)

# Examples
```jldoctest
julia> grav
grav = 9.81 [m/s^2]
```
"""
grav

"""
    year_anom

Length of anomalistic year (s)

# Examples
```jldoctest
julia> year_anom
year_anom = 3.1558464e7 [s]
```
"""
year_anom

"""
    orbit_semimaj

Length of semimajor orbital axis (m)

# Examples
```jldoctest
julia> orbit_semimaj
orbit_semimaj = 1.4959787e11 [m]
```
"""
orbit_semimaj

"""
    TSI

Total solar irradiance (W/m^2)

# Examples
```jldoctest
julia> TSI
TSI = 1362 [W/m^2]
```
"""
TSI

"""
    mslp

Mean sea level pressure (Pa)

# Examples
```jldoctest
julia> mslp
mslp = 101325.0 [Pa]
```
"""
mslp
