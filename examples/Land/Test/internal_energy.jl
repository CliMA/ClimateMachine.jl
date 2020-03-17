# internal_energy.jl: This function calculates soil heat capacity
function internal_energy(c_s,T,theta_ice) 

# ------------------------------------------------------
# Input
#   mineral_properties       ! 'Sand','Clay'
#   theta_liq                ! fraction of water that is liquid
#   theta_ice                ! fraction of water that is ice
# ------------------------------------------------------
# Output
#   κ_out                    ! Soil thermal conductivity
# ------------------------------------------------------   
    
    # Formula for the  Internal Energy of soil (J m-3):
    # I(T,theta;porosity) = c_s*(T-T0) - theta_ice*density_ice*Lf_0 

    # Freezing point of water
    T0 = 273.16 # K
    
    # Specific Latent heat of fusion
    Lf_0 = 333.6e3 # J kg-1
    
    # Density of ice
    density_ice = 917 # kg m-3    
        
    # Internal Energy of soil (J m-3):
    I_soil = c_s*(T-T0) - theta_ice*density_ice*Lf_0 
    
    return I_soil   
end