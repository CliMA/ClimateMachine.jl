#####
##### Tendency specification
#####

#####
##### Sources
#####

# --------- Some of these methods are generic or
#           temporary during transition to new specification:
filter_source(pv, m, s) = nothing
# Sources that have been added to new specification:
filter_source(pv::PV, m, s::Subsidence{PV}) where {PV <: PrognosticVariable} = s
filter_source(pv::PV, m, s::Gravity{PV}) where {PV <: Momentum} = s
filter_source(pv::PV, m, s::GeostrophicForcing{PV}) where {PV <: Momentum} = s
filter_source(pv::PV, m, s::Coriolis{PV}) where {PV <: Momentum} = s
filter_source(pv::PV, m, s::RayleighSponge{PV}) where {PV <: Momentum} = s

filter_source(pv::PV, m, s::CreateClouds{PV}) where {PV <: LiquidMoisture} = s
filter_source(pv::PV, m, s::CreateClouds{PV}) where {PV <: IceMoisture} = s

filter_source(
    pv::PV,
    m,
    s::RemovePrecipitation{PV},
) where {PV <: Union{Mass, Energy, TotalMoisture}} = s

filter_source(
    pv::PV,
    ::NonEquilMoist,
    s::WarmRain_1M{PV},
) where {PV <: LiquidMoisture} = s
filter_source(
    pv::PV,
    ::MoistureModel,
    s::WarmRain_1M{PV},
) where {PV <: LiquidMoisture} = nothing
filter_source(pv::PV, m::MoistureModel, s::WarmRain_1M{PV}) where {PV} = s
filter_source(pv::PV, m::AtmosModel, s::WarmRain_1M{PV}) where {PV} =
    filter_source(pv, m.moisture, s)

filter_source(
    pv::PV,
    ::NonEquilMoist,
    s::RainSnow_1M{PV},
) where {PV <: LiquidMoisture} = s
filter_source(
    pv::PV,
    ::MoistureModel,
    s::RainSnow_1M{PV},
) where {PV <: LiquidMoisture} = nothing
filter_source(
    pv::PV,
    ::NonEquilMoist,
    s::RainSnow_1M{PV},
) where {PV <: IceMoisture} = s
filter_source(
    pv::PV,
    ::MoistureModel,
    s::RainSnow_1M{PV},
) where {PV <: IceMoisture} = nothing
filter_source(pv::PV, m::MoistureModel, s::RainSnow_1M{PV}) where {PV} = s
filter_source(pv::PV, m::AtmosModel, s::RainSnow_1M{PV}) where {PV} =
    filter_source(pv, m.moisture, s)

# Filter sources / empty elements
filter_sources(t::Tuple) = filter(x -> !(x == nothing), t)
filter_sources(pv::PrognosticVariable, m, srcs) =
    filter_sources(map(s -> filter_source(pv, m, s), srcs))

# Entry point
eq_tends(pv::PrognosticVariable, m::AtmosModel, ::Source) =
    filter_sources(pv, m, m.source)
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
eq_tends(pv::PV, m::AtmosModel, tt::Flux{FirstOrder}) where {PV <: Energy} =
    (Advect{PV}(), Pressure{PV}(), eq_tends(pv, m.radiation, tt)...)

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
        hyperdiff_momentum_flux(pv, m.hyperdiffusion, tt)...,
    )

# Energy
eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Energy} = (
    ViscousFlux{PV}(),
    DiffEnthalpyFlux{PV}(),
    hyperdiff_enthalpy_and_momentum_flux(pv, m.hyperdiffusion, tt)...,
)

# Moisture
eq_tends(pv::PV, m::AtmosModel, tt::Flux{SecondOrder}) where {PV <: Moisture} =
    (
        eq_tends(pv, m.moisture, tt)...,
        hyperdiff_momentum_flux(pv, m.hyperdiffusion, tt)...,
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
