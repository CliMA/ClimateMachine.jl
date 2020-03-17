module Soil_Physics
using Parameters

export soil_struct, van_Genuchten, Campbell, tridiagonal_solver, matric_potential, root_uptake, predictor_corrector, compute_grid_settings, compute_grid_settings_heat, physcon, soilvar, soil_thermal_properties, phase_change, soil_temperature

# --- Define soil layers for moisture computation
@with_kw mutable struct soil_struct{}
   nsoi            # Number of soil layers
   dz              = ones(nsoi)       # Soil layer thickness (cm)
   z_plus_onehalf  = zeros(nsoi) # Soil depth [cm] at i+1/2 interface between layers i & i+1 [negative distance from surface]
   z               = zeros(nsoi) # Soil depth [cm] at center of layer i [negative distance from surface]
   dz_plus_onehalf = zeros(nsoi) # Thickness between between z[i] & z[i+1]
   functions       = "aa"
   theta           = zeros(nsoi) # Soil Moisture
   psi             = zeros(nsoi) # Matric potential
   theta0          = 0.0  # Initial SM
   psi0            = 0.0  # initial matric potential
   K               = zeros(nsoi) # Hydraulic conductivity (cm H2O/s)
   cap             = zeros(nsoi) # Specific moisture capacity (/cm)
   Q0              = 0.0 # Infiltration flux (cm H2O/s)
   QN              = 0.0 # Drainage flux (cm H2O/s)
   dtheta          = 0.0 # Change in soil moisture (cm H2O)
   err             = 0.0 # Water balance error (cm H2O)

   # Some root distribution functions
   ssflag          = 1 # perform root uptake else do not
   bi              = 0.98
   fz              = zeros(nsoi)  # root distribution function
   psidry          = -20
   psiopt          = -60
   sink            = zeros(nsoi)
   beta            = zeros(nsoi)  # soil wetness factor

end
# DEfinitions for soil heat
# --- Physical constants in physcon structure
@with_kw mutable struct physcon{}
	tfrz = 273.15;                         	# Freezing point of water [K]
	cwat = 4188.0;                         	# Specific heat of water [J/kg/K]
	cice = 2117.27;                        	# Specific heat of ice [J/kg/K]
	rhowat = 1000.0;                       	# Density of water [kg/m3]
	rhoice = 917.0;                        	# Density of ice [kg/m3]
	cvwat = cwat * rhowat; 					# Heat capacity of water [J/m3/K]
	cvice = cice * rhoice; 					# Heat capacity of ice [J/m3/K]
	tkwat = 0.57;                          	# Thermal conductivity of water [W/m/K]
	tkice = 2.29;                          	# Thermal conductivity of ice [W/m/K]
	hfus = 0.3337e6;                       	# Heat of fusion for water at 0 C [J/kg]
end

# Soil texture classes [Cosby et al. 1984. Water Resources Research 20:682-690]

#  1: sand
#  2: loamy sand
#  3: sandy loam
#  4: silty loam
#  5: loam
#  6: sandy clay loam
#  7  silty clay loam
#  8: clay loam
#  9: sandy clay
# 10: silty clay
# 11: clay

@with_kw mutable struct soilvar{}
	silt = [ 5.0, 12.0, 32.0, 70.0, 39.0, 15.0, 56.0, 34.0,  6.0, 47.0, 20.0]; # Percent silt
	sand = [92.0, 82.0, 58.0, 17.0, 43.0, 58.0, 10.0, 32.0, 52.0,  6.0, 22.0]; # Percent sand
	clay = [ 3.0,  6.0, 10.0, 13.0, 18.0, 27.0, 34.0, 34.0, 42.0, 47.0, 58.0]; # Percent clay

	# Volumetric soil water content at saturation [porosity]
	# (Clapp & Hornberger. 1978. Water Resources Research 14:601-604)

	watsat = [0.395, 0.410, 0.435, 0.485, 0.451, 0.420, 0.477, 0.476, 0.426, 0.492, 0.482]

	nsoi            # Number of soil layers
	dz              = ones(nsoi)    # Soil layer thickness (cm)
	z_plus_onehalf  = zeros(nsoi) 	# Soil depth [cm] at i+1/2 interface between layers i & i+1 [negative distance from surface]
	z               = zeros(nsoi) 	# Soil depth [cm] at center of layer i [negative distance from surface]
	dz_plus_onehalf = zeros(nsoi) 	# Thickness between between z[i] & z[i+1]
	soil_texture    = 1 		  	# Initial Soil Texture Class
	method          = "excess-heat"	# Initial method for Phase Change
	tsoi            = zeros(nsoi) 	# Soil Temperature Grid
	gsoi 			= 0   			# Ground Heat Flux
	hfsoi 			= 0				# Initialize total soil heat of fusion to zero
	h2osoi_ice		= zeros(nsoi) 	# Actual water content as ice fraction
	h2osoi_liq		= zeros(nsoi) 	# Actual water content as liquid fraction
	tk 				= zeros(nsoi) 	# Thermal conductivity
	cv 				= zeros(nsoi) 	# Heat Capacity

	tsoi0 			= zeros(nsoi) 	# Saving current soil temperature for energy conservation check
	tk_plus_onehalf = zeros(nsoi) 	# Thermal conductivity at interface [W/m/K]

end

# Computational Grid
function compute_grid_settings_heat(soil::soilvar)
   # Set the Computational Grid for the Solver
   # Soil layer thickness (m)
   for i = 1:soil.nsoi
      soil.dz[i] = 0.01
   end


   # Soil depth [cm] at i+1/2 interface between layers i & i+1 [negative distance from surface]

   soil.z_plus_onehalf[1] = -soil.dz[1]
   for i = 2:soil.nsoi
      soil.z_plus_onehalf[i] = soil.z_plus_onehalf[i-1] - soil.dz[i]
   end

   # Soil depth [cm] at center of layer i [negative distance from surface]

   soil.z[1]  = 0.5 * soil.z_plus_onehalf[1]
   for i = 2:soil.nsoi
      soil.z[i] = 0.5 * (soil.z_plus_onehalf[i-1] + soil.z_plus_onehalf[i])
   end

   # Thickness between between z[i] & z[i+1]

   for i = 1:soil.nsoi-1
      soil.dz_plus_onehalf[i] = soil.z[i] - soil.z[i+1]
   end
   soil.dz_plus_onehalf[soil.nsoi] = 0.5 * soil.dz[soil.nsoi]

   return soil 
end # Function

# Function for soil thermal conductivity and specific heat
function soil_thermal_properties(physcon::physcon, soilvar::soilvar)

# Calculate soil thermal conductivity & heat capacity

# ------------------------------------------------------
# Input
#   physcon.hfus             ! Heat of fusion for water at 0 C [J/kg]
#   physcon.tfrz             ! Freezing point of water [K]
#   physcon.tkwat            ! Thermal conductivity of water [W/m/K]
#   physcon.tkice            ! Thermal conductivity of ice [W/m/K]
#   physcon.cvwat            ! Heat capacity of water [J/m3/K]
#   physcon.cvice            ! Heat capacity of ice [J/m3/K]
#   physcon.rhowat           ! Density of water [kg/m3]
#   physcon.rhoice           ! Density of ice [kg/m3]
#   soilvar.method           ! Use excess heat | apparent heat capacity for phase change
#   soilvar.soil_texture     ! Soil texture class()
#   soilvar.sand             ! Percent sand
#   soilvar.watsat           ! Volumetric soil water content at saturation [porosity]
#   soilvar.nsoi             ! Number of soil layers
#   soilvar.dz               ! Soil layer thickness [m]
#   soilvar.tsoi             ! Soil temperature [K]
#   soilvar.h2osoi_liq       ! Unfrozen water, liquid [kg H2O/m2]
#   soilvar.h2osoi_ice       ! Frozen water, ice [kg H2O/m2]
#
# Input/output
#   soilvar.tk               ! Thermal conducitivty [W/m/K]
#   soilvar.cv               ! Volumetric heat capacity [J/m3/K]
# ------------------------------------------------------

for i = 1:soilvar.nsoi

   # --- Soil texture to process

   k = soilvar.soil_texture

   # --- Volumetric soil water & ice

   watliq = soilvar.h2osoi_liq[i] / (physcon.rhowat * soilvar.dz[i])
   watice = soilvar.h2osoi_ice[i] / (physcon.rhoice * soilvar.dz[i])

   # Fraction of total volume that is liquid water

   fliq = watliq / (watliq + watice)

   # Soil water relative to saturation

   s = min((watliq + watice)/soilvar.watsat[k], 1)

   # --- Dry thermal conductivity [W/m/K] from bulk density [kg/m3]

   bd = 2700 * (1 - soilvar.watsat[k])
   tkdry = (0.135 * bd + 64.7) / (2700 - 0.947 * bd)

   # --- Soil solids thermal conducitivty [W/m/K]

   # Thermal conductivity of quartz [W/m/K]

   tk_quartz = 7.7

   # Quartz fraction

   quartz = soilvar.sand[k] / 100

   # Thermal conductivity of other minerals [W/m/K]

   if (quartz > 0.2)
      tko = 2
   else()
      tko = 3
   end

   # Thermal conductivity of soil solids [W/m/K]

   tksol = tk_quartz^quartz * tko^(1-quartz)

   # --- Saturated thermal conductivity [W/m/K] and unfrozen & frozen values

   tksat = tksol^(1-soilvar.watsat[k]) * physcon.tkwat^(fliq*soilvar.watsat[k]) * physcon.tkice^(soilvar.watsat[k]-fliq*soilvar.watsat[k])
   tksat_u = tksol^(1-soilvar.watsat[k]) * physcon.tkwat^soilvar.watsat[k]
   tksat_f = tksol^(1-soilvar.watsat[k]) * physcon.tkice^soilvar.watsat[k]

   # --- Kersten number and unfrozen & frozen values

   if (soilvar.sand[k] < 50)
      ke_u = log10(max(s,0.1)) + 1
   else()
      ke_u = 0.7 * log10(max(s,0.05)) + 1
   end
   ke_f = s

   if (soilvar.tsoi[i] >= physcon.tfrz)
      ke = ke_u
   else()
      ke = ke_f
   end
   # --- Thermal conductivity [W/m/K] and unfrozen & frozen values

   soilvar.tk[i] = (tksat - tkdry) * ke + tkdry
#=          
    if i < 100
      soilvar.tk[i] = 0 # 10000.01*[(tksat - tkdry) * ke + tkdry]
    else
      soilvar.tk[i] = 0 # [(tksat - tkdry) * ke + tkdry]
    end
=#
   tku = (tksat_u - tkdry) * ke_u + tkdry
   tkf = (tksat_f - tkdry) * ke_f + tkdry

   # --- Heat capacity of soil solids [J/m3/K]

   cvsol = 1.926e06

   # --- Heat capacity [J/m3/K] and unfrozen & frozen values

   soilvar.cv[i] = (1 - soilvar.watsat[k]) * cvsol + physcon.cvwat * watliq + physcon.cvice * watice
   cvu = (1 - soilvar.watsat[k]) * cvsol + physcon.cvwat * (watliq + watice)
   cvf = (1 - soilvar.watsat[k]) * cvsol + physcon.cvice * (watliq + watice)

   # --- Adjust heat capacity & thermal conductivity if using apparent heat capacity

   if soilvar.method == "apparent-heat-capacity"

      # Temperature range for freezing & thawing [K]

      tinc = 0.5

      # Heat of fusion [J/m3] - This is equivalent to ql = hfus * (h2osoi_liq + h2osoi_ice) / dz

      ql = physcon.hfus * (physcon.rhowat * watliq + physcon.rhoice * watice)

      # Heat capacity & thermal conductivity

      if (soilvar.tsoi[i] > physcon.tfrz+tinc)
         soilvar.cv[i] = cvu
         soilvar.tk[i] = tku
      end

      if (soilvar.tsoi[i] >= physcon.tfrz-tinc && soilvar.tsoi[i] <= physcon.tfrz+tinc)
         soilvar.cv[i] = (cvf + cvu) / 2 + ql / (2 * tinc)
         soilvar.tk[i] = tkf + (tku - tkf) * (soilvar.tsoi[i] - physcon.tfrz + tinc) / (2 * tinc)
      end

      if (soilvar.tsoi[i] < physcon.tfrz-tinc)
         soilvar.cv[i] = cvf
         soilvar.tk[i] = tkf
      end
   end

end

return soilvar
end # function

# Other Functions
function phase_change(physcon::physcon, soilvar::soilvar, dt)

# Adjust temperatures for phase change. Freeze | melt ice using
# energy excess | deficit needed to change temperature to the
# freezing point.

# ------------------------------------------------------
# Input
#   dt                      ! Time step [s]
#   physcon.hfus            ! Heat of fusion for water at 0 C [J/kg]
#   physcon.tfrz            ! Freezing point of water [K]
#   soilvar.nsoi            ! Number of soil layers
#   soilvar.dz              ! Soil layer thickness [m]
#   soilvar.cv              ! Volumetric heat capacity [J/m3/K]
#
# Input/output
#   soilvar.tsoi            ! Soil temperature [K]
#   soilvar.h2osoi_liq      ! Unfrozen water, liquid [kg H2O/m2]
#   soilvar.h2osoi_ice      ! Frozen water, ice [kg H2O/m2]
#
# Output
#   soilvar.hfsoi           ! Soil phase change energy flux [W/m2]
# ------------------------------------------------------

# --- Initialize total soil heat of fusion to zero

soilvar.hfsoi = 0

# --- Now loop over all soil layers to calculate phase change

for i = 1:soilvar.nsoi

   # --- Save variables prior to phase change

   wliq0 = soilvar.h2osoi_liq[i];     # Amount of liquid water before phase change
   wice0 = soilvar.h2osoi_ice[i];     # Amount of ice before phase change
   wmass0 = wliq0 + wice0;            # Amount of total water before phase change
   tsoi0 = soilvar.tsoi[i];           # Soil temperature before phase change

   # --- Identify melting | freezing layers & set temperature to freezing

   # Default condition is no phase change [imelt = 0]

   imelt = 0

   # Melting: if ice exists above melt point; melt some to liquid.
   # Identify melting by imelt = 1

   if (soilvar.h2osoi_ice[i] > 0 && soilvar.tsoi[i] > physcon.tfrz)
      imelt = 1
      soilvar.tsoi[i] = physcon.tfrz
   end

   # Freezing: if liquid exists below melt point; freeze some to ice.
   # Identify freezing by imelt = 2

   if (soilvar.h2osoi_liq[i] > 0 && soilvar.tsoi[i] < physcon.tfrz)
      imelt = 2
      soilvar.tsoi[i] = physcon.tfrz
   end

   # --- Calculate energy for freezing | melting

   # The energy for freezing | melting [W/m2] is assessed from the energy
   # excess | deficit needed to change temperature to the freezing point.
   # This is a potential energy flux; because cannot melt more ice than is()
   # present | freeze more liquid water than is present.
   #
   # heat_flux_pot .> 0: freezing; heat_flux_pot .< 0: melting

   if (imelt > 0)
      heat_flux_pot = (soilvar.tsoi[i] - tsoi0) * soilvar.cv[i] * soilvar.dz[i] / dt
   else()
      heat_flux_pot = 0
   end

   # Maximum energy for melting | freezing [W/m2]

   if (imelt == 1)
      heat_flux_max = -soilvar.h2osoi_ice[i] * physcon.hfus / dt
   end

   if (imelt == 2)
      heat_flux_max = soilvar.h2osoi_liq[i] * physcon.hfus / dt
   end

   # --- Now freeze | melt ice

   if (imelt > 0)

      # Change in ice [kg H2O/m2/s]: freeze [+] | melt [-]

      ice_flux = heat_flux_pot / physcon.hfus

      # Update ice [kg H2O/m2]

      soilvar.h2osoi_ice[i] = wice0 + ice_flux * dt

      # Cannot melt more ice than is present

      soilvar.h2osoi_ice[i] = max(0, soilvar.h2osoi_ice[i])

      # Ice cannot exceed total water that is present

      soilvar.h2osoi_ice[i] = min(wmass0, soilvar.h2osoi_ice[i])

      # Update liquid water [kg H2O/m2] for change in ice

      soilvar.h2osoi_liq[i] = max(0, (wmass0-soilvar.h2osoi_ice[i]))

      # Actual energy flux from phase change [W/m2]. This is equal to
      # heat_flux_pot except if tried to melt too much ice.

      heat_flux = physcon.hfus * (soilvar.h2osoi_ice[i] - wice0) / dt

      # Sum energy flux from phase change [W/m2]

      soilvar.hfsoi = soilvar.hfsoi + heat_flux

      # Residual energy not used in phase change is added to soil temperature

      residual = heat_flux_pot - heat_flux
      soilvar.tsoi[i] = soilvar.tsoi[i] - residual * dt / (soilvar.cv[i] * soilvar.dz[i])

      # Error check: make sure actual phase change does not exceed permissible phase change

      if (abs(heat_flux) > abs(heat_flux_max))
         error("Soil temperature energy conservation error: phase change")
      end

      # Freezing: make sure actual phase change does not exceed permissible phase change
      # & that the change in ice does not exceed permissible change

      if (imelt == 2)

         # Energy flux [W/m2]

         constraint = min(heat_flux_pot, heat_flux_max)
         err = heat_flux - constraint
         if (abs(err) > 1e-03)
            error("Soil temperature energy conservation error: freezing energy flux")
         end

         # Change in ice [kg H2O/m2]

         err = (soilvar.h2osoi_ice[i] - wice0) - constraint / physcon.hfus * dt
         if (abs(err) > 1e-03)
            error("Soil temperature energy conservation error: freezing ice flux")
         end
      end

      # Thawing: make sure actual phase change does not exceed permissible phase change
      # & that the change in ice does not exceed permissible change

      if (imelt == 1)

         # Energy flux [W/m2]

         constraint = max(heat_flux_pot, heat_flux_max)
         err = heat_flux - constraint
         if (abs(err) > 1e-03)
            error("Soil temperature energy conservation error: thawing energy flux")
         end

         # Change in ice [kg H2O/m2]

         err = (soilvar.h2osoi_ice[i] - wice0) - constraint / physcon.hfus * dt
         if (abs(err) > 1e-03)
            error("Soil temperature energy conservation error: thawing ice flux")
         end
      end

   end

end

return soilvar

end # function

function soil_temperature(physcon::physcon, soilvar::soilvar, tsurf, dt)

# Use an implicit formulation with the surface boundary condition specified
# as the surface temperature to solve for soil temperatures at time n+1.
#
# Calculate soil temperatures as:
#
#      dT   d     dT 
#   cv -- = -- (k --)
#      dt   dz    dz 
#
# where: T = temperature [K]
#        t = time [s]
#        z = depth [m]
#        cv = volumetric heat capacity [J/m3/K]
#        k = thermal conductivity [W/m/K]
#
# Set up a tridiagonal system of equations to solve for T at time n+1; 
# where the temperature equation for layer i is()
#
#   d_i = a_i [T_i-1] n+1 + b_i [T_i] n+1 + c_i [T_i+1] n+1
#
# For soil layers undergoing phase change, set T_i = Tf [freezing] & use
# excess energy to freeze | melt ice:
#
#   Hf_i = (Tf - [T_i] n+1) * cv_i * dz_i / dt
#
# During the phase change; the unfrozen & frozen soil water
# (h2osoi_liq, h2osoi_ice) are adjusted.
#
# Or alternatively; use the apparent heat capacity method to
# account for phase change. In this approach; h2osoi_liq
# & h2osoi_ice are not calculated.
#
# ------------------------------------------------------
# Input
#   tsurf                   ! Surface temperature [K]
#   dt                      ! Time step [s]
#   soilvar.method          ! Use excess heat | apparent heat capacity for phase change
#   soilvar.nsoi            ! Number of soil layers
#   soilvar.z               ! Soil depth [m]
#   soilvar.z_plus_onehalf  ! Soil depth [m] at i+1/2 interface between layers i & i+1
#   soilvar.dz              ! Soil layer thickness [m]
#   soilvar.dz_plus_onehalf ! Thickness [m] between between i & i+1
#   soilvar.tk              ! Thermal conductivity [W/m/K]
#   soilvar.cv              ! Heat capacity [J/m3/K]
#
# Input/output
#   soilvar.tsoi            ! Soil temperature [K]
#   soilvar.h2osoi_liq      ! Unfrozen water, liquid [kg H2O/m2]
#   soilvar.h2osoi_ice      ! Frozen water, ice [kg H2O/m2]
#
# Output
#   soilvar.gsoi            ! Energy flux into soil [W/m2]
#   soilvar.hfsoi           ! Soil phase change energy flux [W/m2]
# ------------------------------------------------------

# --- Save current soil temperature for energy conservation check

for i = 1:soilvar.nsoi
   soilvar.tsoi0[i] = soilvar.tsoi[i]
end

# --- Thermal conductivity at interface [W/m/K]

for i = 1:soilvar.nsoi-1
   soilvar.tk_plus_onehalf[i] = soilvar.tk[i] * soilvar.tk[i+1] * (soilvar.z[i]-soilvar.z[i+1]) / (soilvar.tk[i]*(soilvar.z_plus_onehalf[i]-soilvar.z[i+1]) + soilvar.tk[i+1]*(soilvar.z[i]-soilvar.z_plus_onehalf[i]))
end

# --- Set up tridiagonal matrix

# Terms for tridiagonal matrix
a = zeros(soilvar.nsoi);
b = zeros(soilvar.nsoi);
c = zeros(soilvar.nsoi);
d = zeros(soilvar.nsoi);


# Top soil layer with tsurf as boundary condition

i = 1
m = soilvar.cv[i] * soilvar.dz[i] / dt
a[i] = 0
c[i] = -soilvar.tk_plus_onehalf[i] / soilvar.dz_plus_onehalf[i]
b[i] = m - c[i] + soilvar.tk[i] / (0 - soilvar.z[i])
d[i] = m * soilvar.tsoi[i] + soilvar.tk[i] / (0 - soilvar.z[i]) * tsurf

# Layers 2 to nsoi-1

for i = 2:soilvar.nsoi-1
   m = soilvar.cv[i] * soilvar.dz[i] / dt
   a[i] = -soilvar.tk_plus_onehalf[i-1] / soilvar.dz_plus_onehalf[i-1]
   c[i] = -soilvar.tk_plus_onehalf[i] / soilvar.dz_plus_onehalf[i]
   b[i] = m - a[i] - c[i]
   d[i] = m * soilvar.tsoi[i]
end

# Bottom soil layer with zero heat flux

i = soilvar.nsoi
m = soilvar.cv[i] * soilvar.dz[i] / dt
a[i] = -soilvar.tk_plus_onehalf[i-1] / soilvar.dz_plus_onehalf[i-1]
c[i] = 0
b[i] = m - a[i]
d[i] = m * soilvar.tsoi[i]

# --- Solve for soil temperature

soilvar.tsoi = tridiagonal_solver(a, b, c, d, soilvar.nsoi)

# --- Derive energy flux into soil [W/m2]

soilvar.gsoi = soilvar.tk[1] * (tsurf - soilvar.tsoi[1]) / (0 - soilvar.z[1])

# --- Phase change for soil layers undergoing freezing of thawing

if soilvar.method == "apparent-heat-capacity"

   # No explicit phase change energy flux. This is included in the heat capacity.

   soilvar.hfsoi = 0

   elseif  soilvar.method == "excess-heat"

   # Adjust temperatures for phase change. Freeze | melt ice using energy
   # excess | deficit needed to change temperature to the freezing point.
   # The variable hfsoi is returned as the energy flux from phase change [W/m2].

   soilvar = phase_change(physcon, soilvar, dt)

end

# --- Check for energy conservation

# Sum change in energy [W/m2]

edif = 0
for i = 1:soilvar.nsoi
   edif = edif + soilvar.cv[i] * soilvar.dz[i] * (soilvar.tsoi[i] - soilvar.tsoi0[i]) / dt
end

# Error check

err = edif - soilvar.gsoi - soilvar.hfsoi
if (abs(err) > 1e-03)
   error("Soil temperature energy conservation error")
end

return soilvar

end # function


# Van Genuchten Function
function van_Genuchten(params, psi)

# ----------------------------------
# van Genuchten [1980] relationships
# ----------------------------------

# --- Soil parameters

theta_res = params[1];   # Residual water content
theta_sat = params[2];   # Volumetric water content at saturation
alpha = params[3];       # Inverse of the air entry potential
n = params[4];           # Pore-size distribution index
m = params[5];           # Exponent
Ksat = params[6];        # Hydraulic conductivity at saturation
ityp = params[7];        # Soil texture flag

# --- Effective saturation [Se] for specified matric potential [psi]

if (psi <= 0)
   Se = (1 + (alpha * abs(psi))^n)^-m
else()
   Se = 1
end

# --- Volumetric soil moisture [theta] for specified matric potential [psi]

theta = theta_res + (theta_sat - theta_res) * Se

# --- Hydraulic conductivity [K] for specified matric potential [psi]

if (Se <= 1)
   K = Ksat * sqrt(Se) * (1 - (1 - Se^(1/m))^m)^2

   # Special case for Haverkamp et al. (1977) sand [ityp = 1] & Yolo light clay [ityp = 2]

   if (ityp == 1)
      K = Ksat * 1.175e6 / (1.175e6 + abs(psi)^4.74)
   end
   if (ityp == 2)
      K = Ksat * 124.6/ (124.6 + abs(psi)^1.77)
   end

else()

   K = Ksat

end

# --- Specific moisture capacity [cap] for specified matric potential [psi]

if (psi <= 0)
   num = alpha * m * n * (theta_sat - theta_res) * (alpha * abs(psi))^(n-1)
   den =  (1 + (alpha * abs(psi))^n)^(m+1)
   cap = num / den
else()
   cap = 0
end

return theta, K, cap

end # function

function Campbell(params, psi)

# -----------------------------
# Campbell (1974) relationships
# -----------------------------

# --- Soil parameters

theta_sat = params[1];    # Volumetric water content at saturation
psi_sat = params[2];      # Matric potential at saturation
b = params[3];            # Exponent
Ksat = params[4];         # Hydraulic conductivity at saturation

# --- Volumetric soil moisture [theta] for specified matric potential [psi]

if (psi <= psi_sat)
   theta = theta_sat * (psi / psi_sat)^(-1/b)
else()
   theta = theta_sat
end

# --- Hydraulic conductivity [K] for specified matric potential [psi]

if (psi <= psi_sat)
   K = Ksat * (theta / theta_sat)^(2*b+3)
else()
   K = Ksat
end

# --- Specific moisture capacity [cap] for specified matric potential [psi]

if (psi <= psi_sat)
   cap = -theta_sat / (b * psi_sat) * (psi / psi_sat)^(-1/b-1)
else()
   cap = 0
end

return theta, K, cap
 
end # Function

# Other Functions
function tridiagonal_solver(a, b, c, d, n)

# Solve for U given the set of equations R * U = D; where U is a vector
# of length N; D is a vector of length N; & R is an N x N tridiagonal
# matrix defined by the vectors A, B, C each of length N. A[1] &
# C[N] are undefined and are not referenced.
#
#     |B[1] C[1] ...  ...  ...                     |
#     |A[2] B[2] C[2] ...  ...                     |
# R = |     A[3] B[3] C[3] ...                     |
#     |                    ... A[N-1] B[N-1] C[N-1]|
#     |                    ... ...    A[N]   B[N]  |
#
# The system of equations is written as:
#
#    A_i * U_i-1 + B_i * U_i + C_i * U_i+1 = D_i
#
# for i = 1 to N. The solution is found by rewriting the
# equations so that:
#
#    U_i = F_i - E_i * U_i+1

# --- Forward sweep [1 -> N] to get E & F
# Initialize E and F
e = copy(a)*0.0;
f = copy(a)*0.0;

e[1] = c[1] / b[1]
    

for i = 2: 1: n-1
   e[i] = c[i] / (b[i] - a[i] * e[i-1])
end
    

f[1] = d[1] / b[1]

for i = 2: 1: n
   f[i] = (d[i] - a[i] * f[i-1]) / (b[i] - a[i] * e[i-1])
end
    

# --- Backward substitution [N -> 1] to solve for U
u = zeros(n);
u[n] = f[n]

for i = n-1: -1: 1
   u[i] = f[i] - e[i] * u[i+1]
end

return u

end # function

function matric_potential(type, params, theta)

# --- Calculate psi for a given theta

if type == "van_Genuchten"

   theta_res = params[1];    # Residual water content
   theta_sat = params[2];    # Volumetric water content at saturation
   alpha = params[3];        # Inverse of the air entry potential
   n = params[4];            # Pore-size distribution index
   m = params[5];            # Exponent

   Se = (theta - theta_res) / (theta_sat - theta_res);
   psi = -((Se^(-1/m) - 1)^(1/n)) / alpha;

elseif type == "Campbell"

   theta_sat = params[1];    # Volumetric water content at saturation
   psi_sat = params[2];      # Matric potential at saturation
   b = params[3];            # Exponent

   psi = psi_sat * (theta / theta_sat)^-b;

end

return psi
end # function

function root_uptake(soil::soil_struct,ET)

   #Compute the soil sink terms

   densum = 0

   for i = 1:soil.nsoi
      if soil.psi[i] > soil.psidry
         soil.beta[i] = (soil.psi[i]-soil.psidry)/(soil.psiopt-soil.psidry)
      elseif soil.psi[i] <= soil.psidry
         soil.beta[i] = 0.0
      end

      densum = densum + soil.fz[i]*soil.beta[i]

   end

   # compute the sink terms

   for i = 1:soil.nsoi
      soil.sink[i] = ET*soil.fz[i]*soil.beta[i]/densum
   end

   return soil.sink

end # funnction


# Predictor Corrector Function
function predictor_corrector(soil::soil_struct, params, ET::Float64, dt)

# -------------------------------------------------------------
# Use predictor-corrector method to solve the Richards equation
# -------------------------------------------------------------

# Input
# dt                   ! Time step [s]
# soil.nsoi            ! Number of soil layers
# soil.functions       ! van Genuchten | Campbell relationships
# soil.dz_plus_onehalf ! Thickness between between z[i] & z[i+1] (cm)
# soil.dz              ! Soil layer thickness [cm]
# soil.psi0            ! Soil surface matric potential boundary condition [cm]
#
# Input/output
# soil.theta           ! Volumetric soil moisture
# soil.psi             ! Matric potential [cm]
#
# Output
# soil.K               ! Hydraulic conductivity [cm H2O/s]
# soil.cap             ! Specific moisture capacity [/cm]
# soil.Q0              ! Infiltration flux [cm H2O/s]
# soil.QN              ! Drainage flux [cm H2O/s]
# soil.dtheta          ! Change in soil moisture [cm H2O]
# soil.err             ! Water balance error (cm H2O)

# --- Save current soil moisture & matric potential for time n

theta_n = copy(soil.theta)
psi_n   = copy(soil.psi)

# --- Predictor step using implict solution for time n+1/2

# Hydraulic properties for current psi:
# theta - volumetric soil moisture
# K     - hydraulic conductivity
# cap   - specific moisture capacity

for i = 1:soil.nsoi
   if soil.functions == "van_Genuchten"
      soil.theta[i], soil.K[i], soil.cap[i] = van_Genuchten(params, soil.psi[i])
   elseif soil.functions == "Campbell"
      soil.theta[i], soil.K[i], soil.cap[i] = Campbell(params, soil.psi[i])
    end
end

# Hydraulic conductivity at i+1/2 interface between layers i & i+1 is the arithmetic mean()
K_plus_onehalf = soil.theta*0
for i = 1:soil.nsoi-1
   K_plus_onehalf[i] = 0.5 * (soil.K[i] + soil.K[i+1])
end

# Hydraulic conductivity at i=1/2 between surface (i=0) & first layer i=1

K_onehalf = soil.K[1]

# dz at i=1/2 between surface (i=0) & first layer i=1

dz_onehalf = 0.5 * soil.dz[1]

# Compute the sink terms
if soil.ssflag == 1

   soil.sink = root_uptake(soil,ET)
else

   soil.sink = soil.sink.*0
end



# Terms for tridiagonal matrix
a = zeros(soil.nsoi);
b = zeros(soil.nsoi);
c = zeros(soil.nsoi);
d = zeros(soil.nsoi);


i = 1
a[i] = 0
c[i] = -K_plus_onehalf[i] / soil.dz_plus_onehalf[i]
b[i] = soil.cap[i] * soil.dz[i] / (0.5 * dt) + K_onehalf / dz_onehalf - c[i]
d[i] = soil.cap[i] * soil.dz[i] / (0.5 * dt) * soil.psi[i] + K_onehalf / dz_onehalf * soil.psi0 + K_onehalf - K_plus_onehalf[i] - soil.sink[i]

for i = 2:soil.nsoi-1
   a[i] = -K_plus_onehalf[i-1] / soil.dz_plus_onehalf[i-1]
   c[i] = -K_plus_onehalf[i] / soil.dz_plus_onehalf[i]
   b[i] = soil.cap[i] * soil.dz[i] / (0.5 * dt) - a[i] - c[i]
   d[i] = soil.cap[i] * soil.dz[i] / (0.5 * dt) * soil.psi[i] + K_plus_onehalf[i-1] - K_plus_onehalf[i] - soil.sink[i]
end

i = soil.nsoi
a[i] = -K_plus_onehalf[i-1] / soil.dz_plus_onehalf[i-1]
c[i] = 0
b[i] = soil.cap[i] * soil.dz[i] / (0.5 * dt) - a[i] - c[i]
d[i] = soil.cap[i] * soil.dz[i] / (0.5 * dt) * soil.psi[i] + K_plus_onehalf[i-1] - soil.K[i] - soil.sink[i]

# Solve for psi at n+1/2

psi_pred = tridiagonal_solver(a, b, c, d, soil.nsoi)
# --- Corrector step using Crank-Nicolson solution for time n+1

# Hydraulic properties for psi_pred

for i = 1:soil.nsoi
   if soil.functions == "van_Genuchten"
      soil.theta[i], soil.K[i], soil.cap[i] = van_Genuchten(params, psi_pred[i])
   elseif  soil.functions == "Campbell"
      soil.theta[i], soil.K[i], soil.cap[i] = Campbell(params, psi_pred[i])
    end
end

# Hydraulic conductivity at i+1/2 interface between layers i & i+1

for i = 1:soil.nsoi-1
   K_plus_onehalf[i] = 0.5 * (soil.K[i] + soil.K[i+1])
end

# Hydraulic conductivity at i=1/2 between surface (i=0) & first layer i=1

K_onehalf = soil.K[1]

# dz at i=1/2 between surface (i=0) & first layer i=1

dz_onehalf = 0.5 * soil.dz[1]

# Compute the sink terms
if soil.ssflag == 1

   soil.sink = root_uptake(soil,ET)
else

   soil.sink = soil.sink.*0
end


# Terms for tridiagonal matrix

i = 1
a[i] = 0.0
c[i] = -K_plus_onehalf[i] / (2.0 * soil.dz_plus_onehalf[i])
b[i] = soil.cap[i] * soil.dz[i] / dt  + K_onehalf / (2.0 * dz_onehalf) - c[i]
d[i] = soil.cap[i] * soil.dz[i] / dt * soil.psi[i] + K_onehalf / (2.0 * dz_onehalf) * soil.psi0 + K_onehalf / (2.0 * dz_onehalf) * (soil.psi0 - soil.psi[i]) + c[i] * (soil.psi[i] - soil.psi[i+1]) + K_onehalf - K_plus_onehalf[i] - soil.sink[i]
    

for i = 2:soil.nsoi-1
   a[i] = -K_plus_onehalf[i-1] / (2.0 * soil.dz_plus_onehalf[i-1])
   c[i] = -K_plus_onehalf[i] / (2.0 * soil.dz_plus_onehalf[i])
   b[i] = soil.cap[i] * soil.dz[i] / dt - a[i] - c[i]
   d[i] = soil.cap[i] * soil.dz[i] / dt * soil.psi[i] - a[i] * (soil.psi[i-1] - soil.psi[i]) + c[i] * (soil.psi[i] - soil.psi[i+1]) + K_plus_onehalf[i-1] - K_plus_onehalf[i] - soil.sink[i]
end

i = soil.nsoi
a[i] = -K_plus_onehalf[i-1] / (2.0 * soil.dz_plus_onehalf[i-1])
c[i] = 0.0
b[i] = soil.cap[i] * soil.dz[i] / dt - a[i] - c[i]
d[i] = soil.cap[i] * soil.dz[i] / dt * soil.psi[i] - a[i] * (soil.psi[i-1] - soil.psi[i]) + K_plus_onehalf[i-1] - soil.K[i] - soil.sink[i]

# Solve for psi at n+1
    

soil.psi = tridiagonal_solver(a, b, c, d, soil.nsoi)
# --- Check water balance()

soil.Q0 = -K_onehalf / (2.0 * dz_onehalf) * ((soil.psi0 - psi_n[1]) + (soil.psi0 - soil.psi[1])) - K_onehalf
soil.QN = -soil.K[soil.nsoi]

soil.dtheta = 0.0
for i = 1:soil.nsoi
   soil.dtheta = soil.dtheta + (soil.theta[i] - theta_n[i]) * soil.dz[i]
end

soil.err = soil.dtheta - (soil.QN - soil.Q0) * dt

return soil

end # function

# Computational Grid
function compute_grid_settings(soil::soil_struct)
   # Set the Computational Grid for the Solver
   # Soil layer thickness (cm)
   for i = 1:soil.nsoi
      soil.dz[i] = 1.0
   end


   # Soil depth [cm] at i+1/2 interface between layers i & i+1 [negative distance from surface]

   soil.z_plus_onehalf[1] = -soil.dz[1]
   for i = 2:soil.nsoi
      soil.z_plus_onehalf[i] = soil.z_plus_onehalf[i-1] - soil.dz[i]
   end

   # Soil depth [cm] at center of layer i [negative distance from surface]

   soil.z[1]  = 0.5 * soil.z_plus_onehalf[1]
   soil.fz[1] = 1 - soil.bi^abs(soil.z[1])
   for i = 2:soil.nsoi
      soil.z[i] = 0.5 * (soil.z_plus_onehalf[i-1] + soil.z_plus_onehalf[i])

      # Assign the root fraction here
      soil.fz[i] = 1 - soil.bi^abs(soil.z[i]) - soil.fz[i-1] # if z is in cm
   end

   # Thickness between between z[i] & z[i+1]

   for i = 1:soil.nsoi-1
      soil.dz_plus_onehalf[i] = soil.z[i] - soil.z[i+1]
   end
   soil.dz_plus_onehalf[soil.nsoi] = 0.5 * soil.dz[soil.nsoi]



   return soil 
end # Function


end # module