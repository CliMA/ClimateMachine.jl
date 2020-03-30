# kersten.jl: This function calculates kersten number
function kersten(theta_ice,vom,porosity,Sr,a,b,v_sand,v_gravel) 

# ------------------------------------------------------
# Input
#   flag       ! 'yes','no

# ------------------------------------------------------
# Output
#   Ke         ! Kersten number
# ------------------------------------------------------   
    
    # If frozen
    if theta_ice > 0   
        Ke = Sr^(1+vom)
    else # If not frozen
        Ke = Sr^(0.5*(1+vom- a*v_sand - v_gravel))*( (1 + exp(-b*Sr))^(-3) - ((1-Sr)/2)^3 )^(1-vom)   
    end 
    
    return Ke
end
