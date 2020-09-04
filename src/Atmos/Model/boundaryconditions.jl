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


boundary_conditions(m::AtmosModel) = m.boundarycondition

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

<<<<<<< HEAD
=======
function boundary_state!(nf, atmos::AtmosModel, args...)
    atmos_boundary_state!(nf, atmos.problem.boundarycondition, atmos, args...)
end

>>>>>>> master
function boundary_state!(
    nf::Union{CentralNumericalFluxHigherOrder, CentralNumericalFluxDivergence},
    bc,
    m::AtmosModel,
    x...,
)
    nothing
end


function boundary_state!(nf, bc::AtmosBC, atmos::AtmosModel, args...)
    boundary_state!(nf, bc.momentum, atmos, args...)
    boundary_state!(nf, bc.energy,   atmos, args...)
    boundary_state!(nf, bc.moisture, atmos, args...)
    boundary_state!(nf, bc.tracer,   atmos, args...)
    boundary_state!(nf, bc.turbconv, atmos, args...)
end

<<<<<<< HEAD
function numerical_boundary_flux_second_order!(nf, bc::AtmosBC, atmos::AtmosModel, fluxᵀn::Vars, args...)
    numerical_boundary_flux_second_order!(
=======

function normal_boundary_flux_second_order!(
    nf,
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
    bctype::Integer,
    t,
    args...,
) where {S}
    atmos_normal_boundary_flux_second_order!(
        nf,
        atmos.problem.boundarycondition,
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
        bctype,
        t,
        args...,
    )
end
@generated function atmos_normal_boundary_flux_second_order!(
    nf,
    tup::Tuple,
    atmos::AtmosModel,
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
    bctype,
    t,
    args...,
)
    N = fieldcount(tup)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> atmos_normal_boundary_flux_second_order!(
                nf,
                tup[i],
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
                bctype,
                t,
                args...,
            ), #expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end
function atmos_normal_boundary_flux_second_order!(
    nf,
    bc::AtmosBC,
    atmos::AtmosModel,
    args...,
)
    atmos_momentum_normal_boundary_flux_second_order!(
>>>>>>> master
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
