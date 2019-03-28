# CliMA Variable List: 

This document is currently for collaborators within the project with access to the Overleaf CliMA-Atmos docs. The purpose of this page is to unify the naming conventions used in the Overleaf document in a manner useful for coding. This document suggests 'reserved' variable names in <property>_<species> format with the default working fluid (no-subscript) being moist air. Contributors to the CliMA repository are welcome to suggest changes when necessary.

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

P_air           = pressure, air 
ρ_air           = density, air
R_moist         = gas constant, moist
R_dry           = gas constant, dry
R_vap           = gas constant, water vapour
T_air           = temperature, air [temp_air]
T_<species>     = temperature, species 
ε_vd            = ratio of molar masses (== R_airv/R_aird)
```

### 2.2 Mass Balance
```
dt              = time increment
u               = x-velocity 
v               = y-velocity
w               = z-velocity
U		= x-momentum 
V		= y-momentum
W		= z=momentum 
```
### 2.3 Moisture balances 
```
source_qt           = local source/sink of water mass [S_qt]
diffusiveflux_vap   = diffusive flux, water vapour
diffusiveflux_liq   = diffusive flux, cloud liquid
diffusiveflux_ice   = diffusive flux, cloud ice
diffusiveflux_tot   = diffusive flux, total
<Suggestions from the microphysics side needed for section 2.3>
```

### 2.4 Momentum balances
```
U               = x-momentum 
V               = y-momentum 
W               = z-momentum (2D/3D: this is the vertical coordinate)
Ω_x             = x-angular momentum
Ω_y             = y-angular momentum
Ω_z             = z-angular momentum
τ_xx            = viscous and SGS turbulent stress tensor ( (1,1) component)
τ_yy            = viscous and SGS turbulent stress tensor ( (2,2) component)
τ_zz            = viscous and SGS turbulent stress tensor ( (3,3) component)
τ_<ij>          = replace ij with combination of x/y/z to recover appropriate value
λ_stokes        = Stokes parameter
```

### 2.5 Energy balance
```
<Lower case e_<type> suggests specific (per unit mass) quantities>
e_kin_<spe>      = energy, kinetic
e_pot_<spe>      = energy, potential
e_int_<spe>      = energy, internal
e_tot_<spe>      = energy, total

E_kin_<spe>      = energy, kinetic
E_pot_<spe>      = energy, potential
E_int_<spe>      = energy, internal
E_tot_<spe>      = energy, total

cv_vapm         = specific heat,
cv_icem         = specific heat,
cv_liqm         = specific heat, 
cp_vapd         = specific heat, 
cp_iced         = specific heat
cp_liqd         = specific heat 
```
