# CliMA Variable List

This document is currently for collaborators within the project with access to the Overleaf CliMA-Atmos docs. The purpose of this page is to unify the naming conventions used in the Overleaf document in a manner useful for coding. This document suggests 'reserved' variable names in <property>_<species> format with the default working fluid (no-subscript) being moist air. Contributors to the CliMA repository are welcome to suggest changes when necessary.

## Type parameters
The Julia code typically uses `T` as a type parameter, however this conflicts with the typical usage for temperature. Instead, good choices are:
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
Here are suggested names in the output. Format: short name (code name) = long name
```
rho(ρ) = density
u(u) = x_velocity
v(v) = y_velocity
w(w) = z_velocity
qt(q_tot) = total specific humidity
et(e_tot) = total specific energy
qv(q_vap) = water vapor specific humidity
ql(q_liq) = liquid water specific humidity
ei(e_int) = specific internal energy
thd(θ_dry) = potential temperature
thl(θ_liq_ice) = liquid-ice potential temperature
thv(θ_vir) = virtual potential temperature
hm(h_moi) = specific enthalpy
ht(h_tot) = total specific enthalpy

var_u(u′u′) = variance of x-velocity
var_v(v′v′) = variance of y-velocity
var_w(w'w') = variance of z-velocity
TKE(TKE) = turbulence kinetic energy
var_qt(q_tot'q_tot') = variance of total specific humidity
var_ei(e_int'e_int') = variance of specific internal energy
var_thl(θ_liq_ice'θ_liq_ice') = variance of liquid-ice potential temperature
cov_qt_thl(q_tot'θ_liq_ice') = covariance of total specific humidity and liquid-ice potential temperature
cov_qt_ei(q_tot'e_int') = covariance of total specific humidity and specific internal energy
w3(w′w′w′) = the third moment of z-velocity

cov_w_rho(w′ρ′) = vertical flux of mass
cov_w_u(w′u′) = vertical flux of x-velocity
cov_w_v(w′v′) = vertical flux of y-velocity
cov_w_qt(w'q_tot') = vertical flux of total specific humidity
cov_w_qv(w′q_vap′) = vertical flux of water vapor specific humidity
cov_w_ql(w′q_liq′) = vertical flux of liuqid water specific humidity
cov_w_thd(w′θ_dry′) = vertical flux of potential temperature
cov_w_thv(w′θ_vir′) = vertical flux of virtual temperature
cov_w_thl(w′θ_liq_ice′) = vertical flux of liquid-ice potential temperature
w_qt_sgs(d_q_tot) = vertical sgs flux of total specific humidity
w_ht_sgs(d_h_tot) = vertical sgs flux of total specific enthalpy

cl(cl) = cloud fraction
clt(clt) = cloud cover
```

### TODO
```
Update with list of additional parameters / source terms as necessary
```
