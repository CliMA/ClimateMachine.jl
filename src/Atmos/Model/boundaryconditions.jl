using CLIMAParameters.Planet: cv_d, T_0

export InitStateBC

export AtmosBC,
    Impenetrable,
    FreeSlip,
    NoSlip,
    DragLaw,
    Insulating,
    PrescribedTemperature,
    PrescribedEnergyFlux,
    BulkFormulaEnergy,
    Impermeable,
    OutflowPrecipitation,
    ImpermeableTracer,
    PrescribedMoistureFlux,
    BulkFormulaMoisture,
    PrescribedTracerFlux

export average_density_sfc_int

"""
    AtmosBC(momentum = Impenetrable(FreeSlip())
            energy   = Insulating()
            moisture = Impermeable()
            precipitation = OutflowPrecipitation()
            tracer  = ImpermeableTracer())

The standard boundary condition for [`AtmosModel`](@ref). The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct AtmosBC{M, E, Q, P, TR, TC}
    momentum::M = Impenetrable(FreeSlip())
    energy::E = Insulating()
    moisture::Q = Impermeable()
    precipitation::P = OutflowPrecipitation()
    tracer::TR = ImpermeableTracer()
    turbconv::TC = NoTurbConvBC()
end

boundary_conditions(atmos::AtmosModel) = atmos.problem.boundaryconditions


function boundary_state!(
    nf,
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
    atmos_momentum_boundary_state!(nf, bc.momentum, atmos, args...)
    atmos_energy_boundary_state!(nf, bc.energy, atmos, args...)
    atmos_moisture_boundary_state!(nf, bc.moisture, atmos, args...)
    atmos_precipitation_boundary_state!(nf, bc.precipitation, atmos, args...)
    atmos_tracer_boundary_state!(nf, bc.tracer, atmos, args...)
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
