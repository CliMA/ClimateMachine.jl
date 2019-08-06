#### Turbulence Convection
abstract type TurbulenceConvection{N} end


"""
    horizontal_windspeed(state::Vars)

Computes the horizontal windspeed
"""
@inline horizontal_windspeed(state::Vars) = sqrt(state.ρu[1]^2 + state.ρu[2]^2)

include(joinpath("EDMF","EDMF.jl"))

