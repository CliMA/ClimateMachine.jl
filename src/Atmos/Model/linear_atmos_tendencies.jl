
#####
##### First order fluxes
#####

# Mass
eq_tends(pv::PV, ::AtmosLinearModel, ::Flux{FirstOrder}) where {PV <: Mass} =
    (Advect{PV}(),)

# Momentum
eq_tends(
    pv::PV,
    ::AtmosLinearModel,
    ::Flux{FirstOrder},
) where {PV <: Momentum} = (LinearPressureGradient{PV}(),)

# Energy
eq_tends(
    pv::PV,
    m::AtmosLinearModel,
    tt::Flux{FirstOrder},
) where {PV <: Energy} = (LinearEnergyFlux{PV}(),)

# Moisture
# TODO: Is this right?
eq_tends(
    pv::PV,
    ::AtmosLinearModel,
    ::Flux{FirstOrder},
) where {PV <: Moisture} = ()

# Tracers
eq_tends(
    pv::PV,
    ::AtmosLinearModel,
    ::Flux{FirstOrder},
) where {N, PV <: Tracers{N}} = ()

#####
##### Second order fluxes
#####

eq_tends(pv::PV, ::AtmosLinearModel, ::Flux{SecondOrder}) where {PV} = ()

#####
##### Sources
#####
eq_tends(pv::PV, ::AtmosLinearModel, ::Source) where {PV} = ()

eq_tends(
    pv::PV,
    ::AtmosAcousticGravityLinearModel,
    ::Source,
) where {PV <: Momentum} = (Gravity{PV}(),)
