using ..Atmos
using ..Atmos: MoistureModel

# Helpers to gather the dynamic variables 

# Dynamic variables
function vars_dyn(FT)
    @vars begin
        Ω₁::FT
        Ω₂::FT
        Ω₃::FT
    end
end
dyn_vars(array) = Vars{vars_dyn(eltype(array))}(array)

function vars_dyn_bl(FT)
    @vars begin
        Ω_bl₁::FT
        Ω_bl₂::FT
        Ω_bl₃::FT
    end
end
dyn_bl_vars(array) = Vars{vars_dyn_bl(eltype(array))}(array)

