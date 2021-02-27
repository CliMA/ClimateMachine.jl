using CLIMAParameters.Planet: cv_d, T_0
using ..BalanceLaws: BCDef
import ..BalanceLaws: bc_val, default_bcs
export InitStateBC

const NF1 = NumericalFluxFirstOrder
const NF2 = NumericalFluxSecondOrder
const NF∇ = NumericalFluxGradient

export AtmosBC,
    ImpenetrableFreeSlip,
    ImpenetrableNoSlip,
    ImpenetrableDragLaw,
    Insulating,
    PrescribedTemperature,
    PrescribedEnergyFlux,
    Adiabaticθ,
    BulkFormulaEnergy,
    Impermeable,
    OutflowPrecipitation,
    ImpermeableTracer,
    PrescribedMoistureFlux,
    BulkFormulaMoisture,
    PrescribedTracerFlux,
    NishizawaEnergyFlux

export average_density_sfc_int

"""
    AtmosBC(;
            tup      = (ImpenetrableFreeSlip{Momentum}(),)
            momentum = ImpenetrableFreeSlip{Momentum}()
            energy   = Insulating()
            moisture = Impermeable()
            precipitation = OutflowPrecipitation{Rain}()
            tracer  = ImpermeableTracer())

The standard boundary condition for [`AtmosModel`](@ref). The default options imply a "no flux" boundary condition.
"""
struct AtmosBC{M, E, Q, P, TR, TC, T}
    momentum::M
    energy::E
    moisture::Q
    precipitation::P
    tracer::TR
    turbconv::TC
    tup::T
end

function AtmosBC(;
    momentum = ImpenetrableFreeSlip{Momentum}(),
    energy = Insulating(),
    moisture = Impermeable{TotalMoisture}(),
    precipitation = OutflowPrecipitation{Rain}(),
    tracer = ImpermeableTracer(),
    turbconv = NoTurbConvBC(),
    tup = (),
)
    args = (
        momentum,
        energy,
        moisture,
        precipitation,
        tracer,
        turbconv,
        dispatched_tuple(tup),
    )
    return AtmosBC{typeof.(args)...}(args...)
end

default_bcs(atmos::AtmosModel) = (
    DefaultBC{Mass}(),
    ImpenetrableFreeSlip{Momentum}(),
    default_bcs(atmos.energy)...,
    default_bcs(atmos.moisture)...,
    default_bcs(atmos.precipitation)...,
    default_bcs(atmos.turbconv)...,
    default_bcs(atmos.tracers)...,
)

default_bcs(::DryModel) = ()
default_bcs(::EquilMoist) = (Impermeable{TotalMoisture}(),)
default_bcs(::NonEquilMoist) = (
    Impermeable{TotalMoisture}(),
    Impermeable{LiquidMoisture}(),
    Impermeable{IceMoisture}(),
)
default_bcs(::EnergyModel) = (Insulating(), ImpenetrableFreeSlip{Energy}())
default_bcs(::NoPrecipitation) = ()
default_bcs(::RainModel) = (OutflowPrecipitation{Rain}(),)
default_bcs(::RainSnowModel) =
    (OutflowPrecipitation{Rain}(), OutflowPrecipitation{Snow}())
default_bcs(::NoTracers) = ()
default_bcs(::NTracers{N}) where {N} = (ImpermeableTracer{Tracers{N}}(),)
default_bcs(::NoTurbConv) = ()


boundary_conditions(atmos::AtmosModel) = atmos.problem.boundaryconditions

function boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxGradient},
    bc::AtmosBC,
    atmos::AtmosModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    t,
    args...,
)

    ntargs = (; n, state = state⁻, aux = aux⁻, t, args)

    set_bcs!(state⁺, atmos, nf, bc, ntargs)

    atmos_boundary_state!(
        nf,
        bc,
        atmos,
        state⁺,
        aux⁺,
        n,
        state⁻,
        aux⁻,
        t,
        args...,
    )
    # update moisture auxiliary variables (perform saturation adjustment, if necessary)
    # to make thermodynamic quantities consistent with the boundary state
    atmos_nodal_update_auxiliary_state!(atmos.moisture, atmos, state⁺, aux⁺, t)
end

function boundary_state!(
    nf::Union{CentralNumericalFluxHigherOrder, CentralNumericalFluxDivergence},
    bc::AtmosBC,
    m::AtmosModel,
    x...,
)
    nothing
end

function atmos_boundary_state!(nf, bc::AtmosBC, atmos, args...)
    turbconv_boundary_state!(nf, bc.turbconv, atmos, args...)
end


function normal_boundary_flux_second_order!(
    nf,
    bc::AtmosBC,
    atmos::AtmosModel,
    fluxᵀn::Vars{S},
    n⁻,
    state⁻,
    diff⁻,
    hyperdiff⁻,
    aux⁻,
    state⁺,
    diff⁺,
    hyperdiff⁺,
    aux⁺,
    t,
    args...,
) where {S}
    atmos_normal_boundary_flux_second_order!(
        nf,
        bc,
        atmos,
        fluxᵀn,
        n⁻,
        state⁻,
        diff⁻,
        hyperdiff⁻,
        aux⁻,
        state⁺,
        diff⁺,
        hyperdiff⁺,
        aux⁺,
        t,
        args...,
    )
end

function atmos_normal_boundary_flux_second_order!(
    nf,
    bc::AtmosBC,
    atmos::AtmosModel,
    args...,
)
    atmos_momentum_normal_boundary_flux_second_order!(
        nf,
        bc.momentum,
        atmos,
        args...,
    )
    atmos_energy_normal_boundary_flux_second_order!(
        nf,
        bc.energy,
        atmos,
        args...,
    )
    atmos_moisture_normal_boundary_flux_second_order!(
        nf,
        bc.moisture,
        atmos,
        args...,
    )
    atmos_precipitation_normal_boundary_flux_second_order!(
        nf,
        bc.precipitation,
        atmos,
        args...,
    )
    atmos_tracer_normal_boundary_flux_second_order!(
        nf,
        bc.tracer,
        atmos,
        args...,
    )
    turbconv_normal_boundary_flux_second_order!(nf, bc.turbconv, atmos, args...)
end

"""
    average_density(ρ_sfc, ρ_int)

Average density between the surface and the interior point, given
 - `ρ_sfc` density at the surface
 - `ρ_int` density at the interior point
"""
function average_density(ρ_sfc::FT, ρ_int::FT) where {FT <: Real}
    return FT(0.5) * (ρ_sfc + ρ_int)
end

include("bc_momentum.jl")
include("bc_energy.jl")
include("bc_moisture.jl")
include("bc_precipitation.jl")
include("bc_initstate.jl")
include("bc_tracer.jl")
