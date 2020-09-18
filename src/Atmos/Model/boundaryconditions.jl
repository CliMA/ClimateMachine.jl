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
    nf::Union{DivNumericalPenalty, GradNumericalFlux},
    bc::AtmosBC,
    atmos::AtmosModel,
    args...,
)
    nothing
end

function boundary_state!(
    nf::Union{NumericalFluxFirstOrder, NumericalFluxSecondOrder, NumericalFluxGradient},
    bc::AtmosBC,
    atmos::AtmosModel,
    args...,
)
    boundary_state!(nf,bc.momentum, atmos, args...)
    boundary_state!(nf,bc.energy, atmos, args...)
    boundary_state!(nf,bc.moisture, atmos, args...)
    boundary_state!(nf,bc.tracer, atmos, args...)
    boundary_state!(nf,bc.turbconv, atmos, args...)
end


function boundary_flux_second_order!(
    nf::NumericalFluxSecondOrder,
    bc::AtmosBC,
    atmos::AtmosModel,
    args...,
)
    boundary_flux_second_order!(nf,bc.momentum, atmos, args...)
    boundary_flux_second_order!(nf,bc.energy, atmos, args...)
    boundary_flux_second_order!(nf,bc.moisture, atmos, args...)
    boundary_flux_second_order!(nf,bc.tracer, atmos, args...)
    boundary_flux_second_order!(nf,bc.turbconv, atmos, args...)
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
