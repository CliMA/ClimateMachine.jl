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
    ImpermeableTracer,
    PrescribedMoistureFlux,
    BulkFormulaMoisture,
    PrescribedTracerFlux

export average_density_sfc_int


boundary_condition(m::AtmosModel) = m.problem.boundarycondition

"""
    AtmosBC(momentum = Impenetrable(FreeSlip())
            energy   = Insulating()
            moisture = Impermeable()
            tracer  = ImpermeableTracer())

The standard boundary condition for [`AtmosModel`](@ref). The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct AtmosBC{M, E, Q, TR, TC} <: BoundaryCondition
    momentum::M = Impenetrable(FreeSlip())
    energy::E = Insulating()
    moisture::Q = Impermeable()
    tracer::TR = ImpermeableTracer()
    turbconv::TC = NoTurbConvBC()
end

function boundary_state!(
    nf::Union{CentralNumericalFluxHigherOrder, CentralNumericalFluxDivergence},
    bc,
    m::AtmosModel,
    x...,
)
    nothing
end


function boundary_state!(nf, bc::AtmosBC, atmos::AtmosModel, state⁺,
    aux⁺, args...)
    # update moisture auxiliary variables (perform saturation adjustment, if necessary)
    # to make thermodynamic quantities consistent with the boundary state
    atmos_nodal_update_auxiliary_state!(atmos.moisture, atmos, state⁺, aux⁺, t)
    boundary_state!(nf, bc.momentum, atmos, state⁺, aux⁺, args...)
    boundary_state!(nf, bc.energy,   atmos, state⁺, aux⁺, args...)
    boundary_state!(nf, bc.moisture, atmos, state⁺, aux⁺, args...)
    boundary_state!(nf, bc.tracer,   atmos, state⁺, aux⁺, args...)
    boundary_state!(nf, bc.turbconv, atmos, state⁺, aux⁺, args...)
end

function numerical_boundary_flux_second_order!(nf, bc::AtmosBC, atmos::AtmosModel, fluxᵀn::Vars, args...)
    numerical_boundary_flux_second_order!(
        nf,
        bc.momentum,
        atmos,
        fluxᵀn, 
        args...,
    )
    numerical_boundary_flux_second_order!(
        nf,
        bc.energy,
        atmos,
        fluxᵀn, 
        args...,
    )
    numerical_boundary_flux_second_order!(
        nf,
        bc.moisture,
        atmos,
        fluxᵀn, 
        args...,
    )
    numerical_boundary_flux_second_order!(
        nf,
        bc.tracer,
        atmos,
        fluxᵀn, 
        args...,
    )
    numerical_boundary_flux_second_order!(nf, bc.turbconv, atmos, fluxᵀn, args...)
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
include("bc_initstate.jl")
include("bc_tracer.jl")
