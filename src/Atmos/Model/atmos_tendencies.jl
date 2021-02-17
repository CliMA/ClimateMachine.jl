#####
##### Tendency specification
#####

#####
##### Sources
#####

# Diagonalize sources: only add sources that correspond to correct equation
diag_source(pv::PV, m::AtmosModel, s::TendencyDef{Source, PV}) where {PV} = s
diag_source(pv, m, s) = nothing

# Filter sources / empty elements
filter_sources(t::Tuple) = filter(x -> !(x == nothing), t)
filter_sources(pv::PrognosticVariable, m, srcs) =
    filter_sources(map(s -> diag_source(pv, m, s), srcs))

# Entry point
eq_tends(pv::PrognosticVariable, m::AtmosModel, tt::Source) =
    (filter_sources(pv, m, m.source)..., eq_tends(pv, m.turbconv, tt)...)
# ---------

#####
##### First order fluxes
#####

# Mass
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Mass} =
    (Advect{PV}(),)

# Momentum
eq_tends(pv::PV, m::AtmosModel, ::Flux{FirstOrder}) where {PV <: Momentum} =
    (Advect{PV}(), PressureGradient{PV}())

# Energy
eq_tends(pv::PV, m::EnergyModel, tt::Flux{FirstOrder}) where {PV <: Energy} =
    (Advect{PV}(), Pressure{PV}())

eq_tends(pv::PV, m::θModel, tt::Flux{FirstOrder}) where {PV <: ρθ_liq_ice} =
    (Advect{PV}(),)

# TODO: make radiation aware of which energy formulation is used:
# eq_tends(pv::PV, m::AtmosModel, tt::Flux{FirstOrder}) where {PV <: AbstractEnergy} =
#     (eq_tends(pv, m.energy, tt)..., eq_tends(pv, m.energy, m.radiation, tt)...)
eq_tends(
    pv::PV,
    m::AtmosModel,
    tt::Flux{FirstOrder},
) where {PV <: AbstractEnergy} =
    (eq_tends(pv, m.energy, tt)..., eq_tends(pv, m.radiation, tt)...)

# Moisture
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Moisture} =
    (Advect{PV}(),)

# Precipitation
eq_tends(
    pv::PV,
    m::AtmosModel,
    tt::Flux{FirstOrder},
) where {PV <: Precipitation} = (eq_tends(pv, m.precipitation, tt)...,)

# Tracers
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {N, PV <: Tracers{N}} =
    (Advect{PV}(),)

#####
##### Second order fluxes
#####

eq_tends(
    pv::PV,
    ::DryModel,
    ::Flux{SecondOrder},
) where {PV <: Union{Mass, Momentum, Moisture}} = ()
eq_tends(
    pv::PV,
    ::MoistureModel,
    ::Flux{SecondOrder},
) where {PV <: Union{Mass, Momentum, Moisture}} = (MoistureDiffusion{PV}(),)

# Mass
eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Mass} =
    (eq_tends(pv, m.moisture, tt)...,)

# Momentum
eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Momentum} =
    (
        ViscousStress{PV}(),
        eq_tends(pv, m.moisture, tt)...,
        eq_tends(pv, m.turbconv, tt)...,
        eq_tends(pv, m.hyperdiffusion, tt)...,
    )

# Energy
eq_tends(pv::PV, m::EnergyModel, tt::Flux{SecondOrder}) where {PV <: Energy} =
    (ViscousFlux{PV}(), DiffEnthalpyFlux{PV}())

eq_tends(pv::PV, m::θModel, tt::Flux{SecondOrder}) where {PV <: ρθ_liq_ice} =
    (ViscousFlux{PV}(),)

eq_tends(
    pv::PV,
    m::AtmosModel,
    tt::Flux{SecondOrder},
) where {PV <: AbstractEnergy} = (
    eq_tends(pv, m.energy, tt)...,
    eq_tends(pv, m.turbconv, tt)...,
    eq_tends(pv, m.hyperdiffusion, tt)...,
)

# Moisture
eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Moisture} =
    (
        eq_tends(pv, m.moisture, tt)...,
        eq_tends(pv, m.turbconv, tt)...,
        eq_tends(pv, m.hyperdiffusion, tt)...,
    )

# Precipitation
eq_tends(
    pv::PV,
    m::AtmosModel,
    tt::Flux{SecondOrder},
) where {PV <: Precipitation} = (eq_tends(pv, m.precipitation, tt)...,)

# Tracers
eq_tends(
    pv::PV,
    m::AtmosModel,
    tt::Flux{SecondOrder},
) where {N, PV <: Tracers{N}} = (eq_tends(pv, m.tracers, tt)...,)
