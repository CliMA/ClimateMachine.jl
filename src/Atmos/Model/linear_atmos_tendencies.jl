
#####
##### First order fluxes
#####

# Mass
eq_tends(::Mass, ::AtmosLinearModel, ::Flux{FirstOrder}) = (Advect(),)

# Momentum
eq_tends(::Momentum, ::AtmosLinearModel, ::Flux{FirstOrder}) =
    (LinearPressureGradient(),)

# Energy
eq_tends(::Energy, m::AtmosLinearModel, tt::Flux{FirstOrder}) =
    (LinearEnergyFlux(),)

# AbstractMoisture
# TODO: Is this right?
eq_tends(::AbstractMoisture, ::AtmosLinearModel, ::Flux{FirstOrder}) = ()

# Tracers
eq_tends(::Tracers{N}, ::AtmosLinearModel, ::Flux{FirstOrder}) where {N} = ()

#####
##### Second order fluxes
#####

eq_tends(pv::PV, ::AtmosLinearModel, ::Flux{SecondOrder}) where {PV} = ()

#####
##### Sources
#####
eq_tends(pv::PV, ::AtmosLinearModel, ::Source) where {PV} = ()

eq_tends(pv::Momentum, ::AtmosAcousticGravityLinearModel, ::Source) =
    (Gravity(),)
