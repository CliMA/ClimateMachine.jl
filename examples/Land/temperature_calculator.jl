# temperature_calculator.jl: This function calculates soil temperature
function temperature_calculator(c_s,I,theta_ice) 

# ------------------------------------------------------
# Input
#   mineral_properties       ! 'Sand','Clay'
#   theta_liq                ! fraction of water that is liquid
#   theta_ice                ! fraction of water that is ice
# ------------------------------------------------------
# Output
#   κ_out                    ! Soil thermal conductivity
# ------------------------------------------------------   
    
    # Formula for the temperature soil (K):
    # T = T0 + [ ( I + theta_ice*density_ice*Lf_0 ) / c_s ]

    # Freezing point of water
    T0 = 273.16 # K
    
    # Specific Latent heat of fusion
    Lf_0 = 333.6e3 # J kg-1
    
    # Density of ice
    density_ice = 917 # kg m-3    
        
    # Temperature of soil (K):
    T_soil = T0 + ( ( I + theta_ice*density_ice*Lf_0 ) / c_s )
    
    return T_soil
end

    