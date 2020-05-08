# ClimateMachine Variable List

This document is currently for collaborators within the project with
access to the Overleaf CliMA-Atmos docs. The purpose of this page is to
unify the naming conventions used in the Overleaf document in a manner
useful for coding. This document suggests 'reserved' variable names in
<property>_<species> format with the default working fluid (no-subscript)
being moist air. Contributors to the CliMA repository are welcome to
suggest changes when necessary.

## Type parameters

The Julia code typically uses `T` as a type parameter, however this
conflicts with the typical usage for temperature. Instead, good choices are:
- `FT` for floating point values

### Names reserved for 'use and throw' / debug variables
```
dummy
scratch
```

### 2.1  Working Fluid and Equation of State
```
q_dry = dry air mass fraction
q_vap = specific humidity, vapour
q_liq = specific humidity, liquid
q_ice = specific humidity, ice
q_con = specific humidity, condensate
q_tot = specific humidity, total

P_<species>     = pressure, species (no subscript == default working fluid moist air)
ρ_<species>     = density, species (no subscript == default working fluid moist air)
R_m             = gas constant, moist
R_d             = gas constant, dry
R_v             = gas constant, water vapour
T               = temperature, moist air
T_<species>     = temperature, species
```

### 2.2 Mass Balance
```
dt              = time increment
u               = x-velocity
v               = y-velocity
w               = z-velocity
U               = x-momentum
V               = y-momentum
W               = z=momentum
```
### 2.3 Moisture balances
```
source_qt           = local source/sink of water mass [S_qt]
diffusiveflux_vap   = diffusive flux, water vapour
diffusiveflux_liq   = diffusive flux, cloud liquid
diffusiveflux_ice   = diffusive flux, cloud ice
diffusiveflux_tot   = diffusive flux, total
```

### 2.4 Momentum balances
```
U               = x-momentum
V               = y-momentum
W               = z-momentum (2D/3D: this is the vertical coordinate)
Ω_x             = x-angular momentum
Ω_y             = y-angular momentum
Ω_z             = z-angular momentum
τ_xx            = stress tensor ((1,1) component)
τ_<ij>          = replace ij with combination of x/y/z to recover appropriate value
λ_stokes        = Stokes parameter
```

### 2.5 Energy balance
```
<Lower case e_<type> suggests specific (per unit mass) quantities>
e_kin_<spe>      = specific energy per unit volume, kinetic
e_pot_<spe>      = specific energy per unit volume, potential
e_int_<spe>      = specific energy per unit volume, internal
e_tot_<spe>      = specific energy per unit volume, total

E_kin_<spe>      = energy, kinetic
E_pot_<spe>      = energy, potential
E_int_<spe>      = energy, internal
E_tot_<spe>      = energy, total

cv_m             = isochoric specific heat, moist air
cv_d             = isochoric specific heat, dry air
cv_l             = isochoric specific heat, liquid water
cv_v             = isochoric specific heat, water vapour
cv_i             = isochoric specific heat, ice

cp_m             = isobaric specific heat, moist air
cp_d             = isobaric specific heat, dry air
cp_l             = isobaric specific heat, liquid water
cp_v             = isobaric specific heat, water vapour
cp_i             = isobaric specific heat, ice
```

### 2.6 Microphysics
```
q_rai = specific humidity, rain [kg/kg]

terminal_velocity = mass weighted average rain fall speed [m/s]

conv_q_vap_to_q_liq      = tendency to q_liq and q_ice due to
                           condensation/evaporation and
                           sublimation/resublimation from q_vap [1/s]
conv_q_liq_to_q_rai_acnv = tendency to q_rai due to autoconversion from q_liq [1/s]
conv_q_liq_to_q_rai_accr = tendency to q_rai due to accretion from q_liq [1/s]
conv_q_rai_to_q_vap      = tendency to q_vap due to evaporation from q_rai [1/s]
```

### 2.7 Diagnostics

Please see `CliMA Diagnostic Variable List`.

### TODO
```
Update with list of additional parameters / source terms as necessary
```
