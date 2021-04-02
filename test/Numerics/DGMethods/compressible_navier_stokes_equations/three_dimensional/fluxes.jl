
Base.@kwdef struct RoesanovFlux{S,T} <: NumericalFluxFirstOrder
    ω_roe::S = 1.0
    ω_rusanov::T = 1.0
end