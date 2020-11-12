##### Sum wrapper

export Σfluxes, Σsources

"""
    flux

An individual flux.
See [`BalanceLaw`](@ref) for more info.
"""
function flux end

"""
    source

An individual source.
See [`BalanceLaw`](@ref) for more info.
"""
function source end

"""
    ntuple_sum(nt::NTuple{N,T}) where {N, T}

sum of `NTuple`, which requires more strict
type input than `sum`. This is added to better
synchronize the success/failure between CPU/GPU
runs to help improve debugging.
"""
ntuple_sum(nt::NTuple{N, T}) where {N, T} = sum(nt)

"""
    Σfluxes(fluxes::NTuple, args...)

Sum of the fluxes where
 - `fluxes` is an `NTuple{N, TendencyDef{Flux{O}, PV}} where {N, PV, O}`
"""
function Σfluxes(
    fluxes::NTuple{N, TendencyDef{Flux{O}, PV}},
    args...,
) where {N, PV, O}
    return ntuple_sum(ntuple(Val(N)) do i
        flux(fluxes[i], args...)
    end)
end
Σfluxes(fluxes::Tuple{}, args...) = 0

"""
    Σsources(sources::NTuple, args...)

Sum of the sources where
 - `sources` is an `NTuple{N, TendencyDef{Source, PV}} where {N, PV}`
"""
function Σsources(
    sources::NTuple{N, TendencyDef{Source, PV}},
    args...,
) where {N, PV}
    return ntuple_sum(ntuple(Val(N)) do i
        source(sources[i], args...)
    end)
end
Σsources(sources::Tuple{}, args...) = 0
