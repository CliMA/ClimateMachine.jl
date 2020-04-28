using CLIMAParameters.Planet: cv_d, T_0

export BoundaryCondition, InitStateBC

export AtmosBC,
    Impenetrable,
    FreeSlip,
    NoSlip,
    DragLaw,
    Insulating,
    PrescribedTemperature,
    PrescribedEnergyFlux,
    Impermeable,
    ImpermeableTracer,
    PrescribedMoistureFlux,
    PrescribedTracerFlux

"""
    AtmosBC(momentum = Impenetrable(FreeSlip())
            energy   = Insulating()
            moisture = Impermeable()
            tracer  = ImpermeableTracer())

The standard boundary condition for [`AtmosModel`](@ref). The default options imply a "no flux" boundary condition.
"""
Base.@kwdef struct AtmosBC{M, E, Q, TR}
    momentum::M = Impenetrable(FreeSlip())
    energy::E = Insulating()
    moisture::Q = Impermeable()
    tracer::TR = ImpermeableTracer()
end

function boundary_state!(nf, atmos::AtmosModel, args...)
    atmos_boundary_state!(nf, atmos.boundarycondition, atmos, args...)
end

function boundary_state!(
    nf::Union{CentralHyperDiffusiveFlux, CentralDivPenalty},
    m::AtmosModel,
    x...,
)
    nothing
end

@generated function atmos_boundary_state!(
    nf,
    tup::Tuple,
    atmos,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
)
    N = fieldcount(tup)
    return quote
        Base.Cartesian.@nif(
            $(N + 1),
            i -> bctype == i, # conditionexpr
            i -> atmos_boundary_state!(
                nf,
                tup[i],
                atmos,
                state⁺,
                aux⁺,
                n,
                state⁻,
                aux⁻,
                bctype,
                t,
                args...,
            ), # expr
            i -> error("Invalid boundary tag")
        ) # elseexpr
        return nothing
    end
end

function atmos_boundary_state!(nf, bc::AtmosBC, atmos, args...)
    atmos_momentum_boundary_state!(nf, bc.momentum, atmos, args...)
    atmos_energy_boundary_state!(nf, bc.energy, atmos, args...)
    atmos_moisture_boundary_state!(nf, bc.moisture, atmos, args...)
    atmos_tracer_boundary_state!(nf, bc.tracer, atmos, args...)
end


function normal_boundary_flux_diffusive!(
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
    atmos_normal_boundary_flux_diffusive!(
        nf,
        atmos.boundarycondition,
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
@generated function atmos_normal_boundary_flux_diffusive!(
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
            i -> atmos_normal_boundary_flux_diffusive!(
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
function atmos_normal_boundary_flux_diffusive!(
    nf,
    bc::AtmosBC,
    atmos::AtmosModel,
    args...,
)
    atmos_momentum_normal_boundary_flux_diffusive!(
        nf,
        bc.momentum,
        atmos,
        args...,
    )
    atmos_energy_normal_boundary_flux_diffusive!(nf, bc.energy, atmos, args...)
    atmos_moisture_normal_boundary_flux_diffusive!(
        nf,
        bc.moisture,
        atmos,
        args...,
    )
    atmos_tracer_normal_boundary_flux_diffusive!(nf, bc.tracer, atmos, args...)
end

include("bc_momentum.jl")
include("bc_energy.jl")
include("bc_moisture.jl")
include("bc_initstate.jl")
include("bc_tracer.jl")
