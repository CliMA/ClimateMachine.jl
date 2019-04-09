module AtmosDycore

export solve!, getrhsfunction

using Requires

abstract type AbstractAtmosDiscretization end

"""
    getrhsfunction(disc::AbstractAtmosDiscretization)

The spatial discretizations are of the form ``Q̇ = f(Q)``, and this function
returns the handle to right-hand side function ``f`` of the `disc`
"""
getrhsfunction(disc::AbstractAtmosDiscretization) =
throw(MethodError(getrhsfunction, typeof(disc)))

include("VanillaAtmosDiscretizations.jl")

end # module
