"""
    Parameters

Module containing sets of parameters.
"""
module Parameters

export AbstractParameterSet
export EarthParameterSet
abstract type AbstractParameterSet end

struct EarthParameterSet <: AbstractParameterSet end

include("Planet.jl")
include("Atmos.jl")
# include("Ocean.jl")
# include("Land.jl")
# include("Ice.jl")

end
