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
    Σfluxes(fluxes::NTuple, bl, args)

Sum of the fluxes where
 - `fluxes` is an `NTuple{N, TendencyDef{Flux{O}, PV}} where {N, PV, O}`
 - `bl` is the balance law
 - `args` are the arguments passed to the individual `flux` functions
"""
function Σfluxes(
    pv::PV,
    fluxes::NTuple{N, TendencyDef{Flux{O}, PV}},
    bl,
    args,
) where {N, O, PV}
    return ntuple_sum(
        ntuple(Val(N)) do i
            projection(bl, fluxes[i], args, flux(fluxes[i], bl, args))
        end,
    )
end

# Emptry scalar case:
function Σfluxes(
    pv::PV,
    fluxes::NTuple{0, TendencyDef{Flux{O}, PV}},
    args...,
) where {O, PV}
    return SVector(0, 0, 0)
end

# Emptry vector case:
function Σfluxes(
    pv::PV,
    fluxes::NTuple{0, TendencyDef{Flux{O}, PV}},
    args...,
) where {O, PV <: AbstractMomentum}
    return SArray{Tuple{3, 3}}(ntuple(i -> 0, 9))
end

# Emptry tracer case:
function Σfluxes(
    pv::PV,
    fluxes::NTuple{0, TendencyDef{Flux{O}, PV}},
    args...,
) where {O, N, PV <: AbstractTracers{N}}
    return SArray{Tuple{3, N}}(ntuple(i -> 0, 3 * N))
end

"""
    Σsources(sources::NTuple, bl, args)

Sum of the sources where
 - `sources` is an `NTuple{N, TendencyDef{Source, PV}} where {N, PV}`
 - `bl` is the balance law
 - `args` are the arguments passed to the individual `source` functions
"""
function Σsources(
    pv::PV,
    sources::NTuple{N, TendencyDef{Source, PV}},
    bl,
    args,
) where {N, PV}
    return ntuple_sum(
        ntuple(Val(N)) do i
            projection(bl, sources[i], args, source(sources[i], bl, args))
        end,
    )
end

# Emptry scalar case:
function Σsources(
    pv::PV,
    sources::NTuple{0, TendencyDef{Source, PV}},
    args...,
) where {PV}
    return 0
end

# Emptry vector case:
function Σsources(
    pv::PV,
    sources::NTuple{0, TendencyDef{Source, PV}},
    args...,
) where {PV <: AbstractMomentum}
    return SVector(0, 0, 0)
end

# Emptry tracer case:
function Σsources(
    pv::PV,
    sources::NTuple{0, TendencyDef{Source, PV}},
    args...,
) where {N, PV <: AbstractTracers{N}}
    return SArray{Tuple{N}}(ntuple(i -> 0, N))
end
