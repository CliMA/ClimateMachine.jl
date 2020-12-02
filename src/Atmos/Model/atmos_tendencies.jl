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
    s::Rain_1M{PV},
) where {PV <: LiquidMoisture} = s
filter_source(
    pv::PV,
    ::MoistureModel,
    s::Rain_1M{PV},
) where {PV <: LiquidMoisture} = nothing
filter_source(pv::PV, m::MoistureModel, s::Rain_1M{PV}) where {PV} = s

filter_source(pv::PV, m::AtmosModel, s::Rain_1M{PV}) where {PV} =
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
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Precipitation} =
    ()

#####
##### Second order fluxes
#####

# Mass
moist_diffusion(pv::PV, ::DryModel) where {PV <: Mass} = ()
moist_diffusion(pv::PV, ::MoistureModel) where {PV <: Mass} =
    (MoistureDiffusion{PV}(),)
eq_tends(pv::PV, m::AtmosModel, ::Flux{SecondOrder}) where {PV <: Mass} =
    (moist_diffusion(pv, m.moisture)...,)

# Momentum
eq_tends(pv::PV, ::AtmosModel, ::Flux{SecondOrder}) where {PV <: Momentum} =
    (ViscousStress{PV}(),)

# Energy
eq_tends(pv::PV, ::AtmosModel, ::Flux{SecondOrder}) where {PV <: Energy} =
    (ViscousProduction{PV}(), EnthalpyProduction{PV}())

# Moisture
eq_tends(pv::PV, ::AtmosModel, ::Flux{SecondOrder}) where {PV <: Moisture} = ()

# Precipitation
eq_tends(
    pv::PV,
    ::AtmosModel,
    ::Flux{SecondOrder},
) where {PV <: Precipitation} = ()
