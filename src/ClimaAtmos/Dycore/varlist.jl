# Following the CLIMAAtmos design document
# Based on the current list of variables (in the order that they appear in equations)
# This suggests 'reserved' variable names

Variable names in property_species format ? 

Alternates in square brackets

# Names saved for 'use and throw' debug variables
dummy
scratch

2.1  Working Fluid and Equation of State

q_dry = dry air mass fraction
q_vap = specific humidity, vapour
q_liq = specific humidity, liquid
q_ice = specific humidity, ice
q_con = specific humidity, condensate
q_tot = specific humidity, total

p_air           = pressure, air 
ρ_air           = density, air
R_moist         = gas constant, moist
R_dry           = gas constant, dry
R_vap           = gas constant, water vapour
T_air           = temperature, air [temp_air]
temp_<species>  = temperature, species (perhaps limit to 3 letter key)
T_<species>     = temperature, species
<Feedback?>
ε_vd            = ratio of molar masses (== R_airv/R_aird)

2.2 Mass Balance
dt                  = time increment
vel_u               = x-velocity [u]
vel_v               = y-velocity [v]
vel_w               = z-velocity [w]

2.3 Moisture balances 
source_qt           = local source/sink of water mass [S_qt]
diffusiveflux_vap   = diffusive flux, water vapour
diffusiveflux_liq   = diffusive flux, cloud liquid
diffusiveflux_ice   = diffusive flux, cloud ice
diffusiveflux_tot   = diffusive flux, total
<Suggestions from the microphysics side needed for section 2.3>

2.4 Momentum balances
U               = x-momentum [mom_x]
V               = y-momentum [mom_y]
W               = z-momentum [mom_z] (in 2D, this is always the vertical coordinate)
Ω_x             = x-angular momentum
Ω_y             = y-angular momentum
Ω_z             = z-angular momentum
τ_xx            = viscous and SGS turbulent stress tensor ( (1,1) component)
τ_yy            = viscous and SGS turbulent stress tensor ( (2,2) component)
τ_zz            = viscous and SGS turbulent stress tensor ( (3,3) component)
τ_<ij>          = replace ij with combination of x/y/z to recover appropriate value
gravpot         = gravitational potential energy 
λ_stokes        = Stokes parameter

2.5 Energy balance
I_dry        = internal energy, dry
I_vap        = internal energy, vapour
I_liq        = internal energy, liquid
I_ice        = internal energy, ice
I_tot        = internal energy, total
Reference values given by
I_vap0       

<Lower case e<stuff> suggests specific (per unit mass) quantities>
ekin_<spe>      = kinetic energy, 
epot_<spe>      = potential energy,
eint_<spe>      = internal energy
etot_<spe>      = total energy, 

Ekin_<spe>      = kinetic energy,
Epot_<spe>      = potential energy,
Eint_<spe>      = internal energy
Etot_<spe>      = total energy, 

cv_vapm         = specific heat
cv_icem         = specific heat
cv_liqm         = specific heat
cp_vapd         = specific heat
cp_iced         = specific heat
cp_liqd         = specific heat 
