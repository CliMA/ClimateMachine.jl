# thermal_properties.jl: This function calculates soil thermal conductivity
function thermal_properties(mineral_properties,theta_liq,theta_ice) 

# ------------------------------------------------------
# Input
#   mineral_properties       ! 'Sand','Clay'
#   theta_liq                ! fraction of water that is liquid
#   theta_ice                ! fraction of water that is ice
#   vom = 0.05        # Volume fraction of organic matter in soil: Global tabulated values from Dai et al. (2019a) [???]
#   porosity = 0.5    # Porosity: Global tabulated values from Dai et al. (2019a) [???]
# ------------------------------------------------------
# Output
#   κ_out                    ! Soil thermal conductivity
# ------------------------------------------------------   
    
    # Formula for thermal conductivity: κ = Ke*κ_sat  + (1-Ke)*κ_dry
    
    # κ_dry: Global tabulated values from Dai et al. (2019a)
    # [ Sand: λ = 2.42 W m-1 K-1 ; Clay: λ = 1.17 W m-1 K-1 ]
    if mineral_properties == "Sand"
        κ_dry  = 2.42
    elseif mineral_properties == "Clay"
        κ_dry  = 1.17
    else
        κ_dry  = 2.0
    end
    
    # Total water = liquid water + ice water
    theta_water =  theta_liq + theta_ice 
    
    # κ_sat_frozen and κ_sat_unfrozen: Global tabulated values from Dai et al. (2019a)
    if mineral_properties == "Sand"
        κ_sat_frozen  = 2.42
        κ_sat_unfrozen  = 2.42
    elseif mineral_properties == "Clay"
        κ_sat_frozen  = 1.17
        κ_sat_unfrozen  = 2.42
    else
        κ_sat_frozen  = 2.0
        κ_sat_unfrozen  = 2.42
    end
    
    # κ_sat = f(  κ_sat_frozen ,  κ_sat_unfrozen )
    κ_sat = ( κ_sat_frozen )^(theta_liq/theta_water) * ( κ_sat_unfrozen )^(theta_ice/theta_water)
    
    # Kersten number from Dai et al (2019a) and Balland and Arp (2005)
    vom = 0.05 # Volume fraction of organic matter in soil: Global tabulated values from Dai et al. (2019a) [???]
    porosity = 0.5 # Porosity: Global tabulated values from Dai et al. (2019a) [???]
    Sr = (theta_water) / (porosity) # Relative Saturation Sr = (theta_liquid + theta_ice) / porosity
    a = 0.24 # a = -0.24 +/- 0.04 ... adjustable parameter based on soil measurements
    b = 18.1 # b = 18.1 +/- 1.1 ... adjustable parameter based on soil measurements
    v_sand = 0.2 # Global tabulated values from Dai et al. (2019a) [???]
    v_gravel = 0.2 # Global tabulated values from Dai et al. (2019a) [???]
    
    # Use Kersten number function
    Ke = kersten(theta_ice,vom,porosity,Sr,a,b,v_sand,v_gravel)  
       
    κ = Ke*κ_sat  + (1-Ke)*κ_dry
    κ_out  = κ    
  
    return κ_out
end
