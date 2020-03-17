# heat_capacity.jl: This function calculates soil heat capacity
function heat_capacity(mineral_properties,porosity,theta_liq,theta_ice) 

# ------------------------------------------------------
# Input
#   mineral_properties       ! 'Sand','Clay'
#   theta_liq                ! fraction of water that is liquid
#   theta_ice                ! fraction of water that is ice
# ------------------------------------------------------
# Output
#   κ_out                    ! Soil thermal conductivity
# ------------------------------------------------------   
    
    # Formula for the  Volumetric  heat  capacity  of  soil  (heat  capacity  per  unit  mass,J m-3K−1):
    # c_s = (1-q_w)*c_ds + q_l*c_l + q_i*c_i
    
    # Volumetric  heat  capacity  of  dry soil
    if mineral_properties == "Sand"
        c_ds = 2.49e6   # J m-3 K-1, Global tabulated values [???]
    elseif mineral_properties == "Clay"
        c_ds = 2.61e6   # J m-3 K-1, Global tabulated values [???]
    else
        c_ds = 2.55e6   # J m-3 K-1, Global tabulated values [???]
    end
        
    # Volumetric  heat  capacity  of  liquid water
    c_l = 4.18e6 # J kg−1K−1
    
    # Volumetric  heat  capacity  of  ice
    c_i = 1.93e6 # J kg−1K−1
    
    # Volumetric  heat  capacity  of  soil  (heat  capacity  per  unit  mass,J m-3K−1):
    c_s = (1-porosity)*c_ds + theta_liq*c_l + theta_ice*c_i
                       
    return c_s
end
