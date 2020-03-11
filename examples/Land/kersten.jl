# kersten.jl: This function calculates kersten number
function kersten(flag) 

# ------------------------------------------------------
# Input
#   flag       ! 'yes','no

# ------------------------------------------------------
# Output
#   Ke         ! Kersten number
# ------------------------------------------------------   
    
    if flag == "yes"
        Ke  = 0.5
    else
        Ke  = 1
    end

    return Ke
end
