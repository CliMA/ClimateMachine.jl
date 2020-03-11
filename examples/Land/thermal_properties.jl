# thermal_properties.jl: This function calculates soil thermal conductivity
function thermal_properties(mineral_properties,theta_liq,theta_ice) 

# ------------------------------------------------------
# Input
#   mineral_properties       ! 'Sand','Clay'
#   theta_liq                ! fraction of water that is liquid
#   theta_ice                ! fraction of water that is ice
# ------------------------------------------------------
# Output
#   κ_out                    ! Soil thermal conductivity
# ------------------------------------------------------   
    
    # [ Sand: λ = 2.42 W m-1 K-1 ; Clay: λ = 1.17 W m-1 K-1 ]
    if mineral_properties == "Sand"
        κ  = 2.42
    elseif mineral_properties == "Clay"
        κ  = 1.17
    else
        κ  = 2.0
    end
    
    Ke = kersten( "yes" )
    
    # Adjust for freezing
    theta_frac = theta_liq / (theta_liq + theta_ice)        
    κ_out  = Ke*theta_frac*κ    
  
    return κ_out
end
