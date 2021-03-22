#####
##### Tendency specification
#####

#####
##### Sources
#####

eq_tends(pv::AbstractPrognosticVariable, m::AtmosModel, tt::Source) =
    (m.source[pv]..., eq_tends(pv, m.turbconv, tt)...)

#####
##### First order fluxes
#####

eq_tends(::Mass, ::Anelastic1D, ::Flux{FirstOrder}) = ()

eq_tends(::Mass, ::Compressible, ::Flux{FirstOrder}) = (Advect(),)

# Mass
eq_tends(pv::Mass, atmos::AtmosModel, tt::Flux{FirstOrder}) =
    (eq_tends(pv, atmos.compressibility, tt))

# Momentum
eq_tends(pv::Momentum, ::Compressible, ::Flux{FirstOrder}) =
    (PressureGradient(),)

eq_tends(pv::Momentum, ::Anelastic1D, ::Flux{FirstOrder}) = ()

eq_tends(pv::Momentum, m::AtmosModel, tt::Flux{FirstOrder}) =
    (Advect(), eq_tends(pv, m.compressibility, tt)...)

# Energy
eq_tends(::Energy, m::TotalEnergyModel, tt::Flux{FirstOrder}) =
    (Advect(), Pressure())

eq_tends(::ρθ_liq_ice, m::θModel, tt::Flux{FirstOrder}) = (Advect(),)

# TODO: make radiation aware of which energy formulation is used:
# eq_tends(pv::PV, m::AtmosModel, tt::Flux{FirstOrder}) where {PV <: AbstractEnergyVariable} =
#     (eq_tends(pv, m.energy, tt)..., eq_tends(pv, m.energy, m.radiation, tt)...)
eq_tends(pv::AbstractEnergyVariable, m::AtmosModel, tt::Flux{FirstOrder}) =
    (eq_tends(pv, m.energy, tt)..., eq_tends(pv, m.radiation, tt)...)

# AbstractMoistureVariable
eq_tends(::AbstractMoistureVariable, ::AtmosModel, ::Flux{FirstOrder}) =
    (Advect(),)

# AbstractPrecipitationVariable
eq_tends(
    pv::AbstractPrecipitationVariable,
    m::AtmosModel,
    tt::Flux{FirstOrder},
) = (eq_tends(pv, m.precipitation, tt)...,)

# Tracers
eq_tends(pv::Tracers{N}, ::AtmosModel, ::Flux{FirstOrder}) where {N} =
    (Advect(),)

#####
##### Second order fluxes
#####

eq_tends(
    ::Union{Mass, Momentum, AbstractMoistureVariable},
    ::DryModel,
    ::Flux{SecondOrder},
) = ()
eq_tends(
    ::Union{Mass, Momentum, AbstractMoistureVariable},
    ::AbstractMoistureModel,
    ::Flux{SecondOrder},
) = (MoistureDiffusion(),)

# Mass
eq_tends(pv::Mass, m::AtmosModel, tt::Flux{SecondOrder}) =
    (eq_tends(pv, m.moisture, tt)...,)

# Momentum
eq_tends(pv::Momentum, m::AtmosModel, tt::Flux{SecondOrder}) = (
    ViscousStress(),
    eq_tends(pv, m.moisture, tt)...,
    eq_tends(pv, m.turbconv, tt)...,
    eq_tends(pv, m.hyperdiffusion, tt)...,
)

# Energy
eq_tends(::Energy, m::TotalEnergyModel, tt::Flux{SecondOrder}) =
    (ViscousFlux(), DiffEnthalpyFlux())

eq_tends(::ρθ_liq_ice, m::θModel, tt::Flux{SecondOrder}) = (ViscousFlux(),)

eq_tends(pv::AbstractEnergyVariable, m::AtmosModel, tt::Flux{SecondOrder}) = (
    eq_tends(pv, m.energy, tt)...,
    eq_tends(pv, m.turbconv, tt)...,
    eq_tends(pv, m.hyperdiffusion, tt)...,
)

# AbstractMoistureVariable
eq_tends(pv::AbstractMoistureVariable, m::AtmosModel, tt::Flux{SecondOrder}) = (
    eq_tends(pv, m.moisture, tt)...,
    eq_tends(pv, m.turbconv, tt)...,
    eq_tends(pv, m.hyperdiffusion, tt)...,
)

# AbstractPrecipitationVariable
eq_tends(
    pv::AbstractPrecipitationVariable,
    m::AtmosModel,
    tt::Flux{SecondOrder},
) = (eq_tends(pv, m.precipitation, tt)...,)

# Tracers
eq_tends(pv::Tracers{N}, m::AtmosModel, tt::Flux{SecondOrder}) where {N} =
    (eq_tends(pv, m.tracers, tt)...,)
