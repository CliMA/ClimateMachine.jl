using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxSecondOrder
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.Mesh.Grids: Direction

import ClimateMachine.BalanceLaws:
    # declaration
    vars_state,
    # initialization
    nodal_init_state_auxiliary!,
    init_state_prognostic!,
    init_state_auxiliary!,
    # rhs computation
    compute_gradient_argument!,
    compute_gradient_flux!,
    flux_first_order!,
    flux_second_order!,
    source!,
    # boundary conditions
    boundary_conditions,
    boundary_state!

abstract type DryAtmosLinearModel <: BalanceLaw end

Base.@kwdef struct DryAtmosAcousticGravityLinearModel{P,B} <: DryAtmosLinearModel
    physics::P
    boundary_conditions::B
end

"""
    LHS computations
"""
function flux_first_order!(
    lm::DryAtmosAcousticGravityLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # ref = aux.ref_state
    
    # flux.ρ = state.ρu
    # pL = calc_pressure(lm.physics.eos, state, aux)
    # flux.ρu += pL * I
    # flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu

    # lm.physics.lhs = (DivergencePressure(eos), LinearAdvection())
    lhs = lm.physics.lhs
    ntuple(Val(length(lhs))) do s
        Base.@_inline_meta
        calc_flux!(flux, lhs[s], state, aux, t)
    end

    return nothing
end

# flux_second_order!(
#     lm::DryAtmosLinearModel,
#     flux::Grad,
#     state::Vars,
#     diffusive::Vars,
#     aux::Vars,
#     t::Real,
# ) = nothing

# function flux_second_order!(
#     lm::DryAtmosLinearModel,
#     flux::Grad,
#     state::Vars,
#     diffusive::Vars,
#     hyperdiffusive::Vars,
#     aux::Vars,
#     t::Real,
# )
#     nothing
# end

"""
    RHS computations
"""
function source!(
    lm::DryAtmosAcousticGravityLinearModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    ::NTuple{1, Dir},
) where {Dir <: Direction}
    sources = lm.physics.sources
    if Dir === VerticalDirection || Dir === EveryDirection
        ntuple(Val(length(sources))) do s
            Base.@_inline_meta
            calc_force!(source, sources[s], state, aux)
        end
    end
    nothing
end