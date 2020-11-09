#####
##### Tendency specification
#####

import ..BalanceLaws: eq_tends

# --------- Some of these methods are generic or
#           temporary during transition to new specification:
filter_source(pv::PrognosticVariable, s) = nothing
# Sources that have been added to new specification:
filter_source(pv::PV, s::Subsidence{PV}) where {PV <: PrognosticVariable} = s
filter_source(pv::PV, s::Gravity{PV}) where {PV <: Momentum} = s

# Filter sources / empty elements
filter_sources(t::Tuple) = filter(x -> !(x == nothing), t)
filter_sources(pv::PrognosticVariable, srcs) =
    filter_sources(map(s -> filter_source(pv, s), srcs))

# Entry point
eq_tends(pv::PrognosticVariable, m::AtmosModel, ::Source) =
    filter_sources(pv, m.source)
# ---------

# Mass
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Mass} =
    (Advect{PV}(),)

# Momentum
eq_tends(pv::PV, m::AtmosModel, ::Flux{FirstOrder}) where {PV <: Momentum} =
    (Advect{PV}(), PressureGradient{PV}())

# Energy
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: Energy} =
    (Advect{PV}(), Pressure{PV}())

# TotalMoisture
eq_tends(pv::PV, ::AtmosModel, ::Flux{FirstOrder}) where {PV <: TotalMoisture} =
    ()
