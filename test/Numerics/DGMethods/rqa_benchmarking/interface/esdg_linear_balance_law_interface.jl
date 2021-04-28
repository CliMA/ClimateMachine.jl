using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxSecondOrder
using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.Mesh.Grids: Direction

import ClimateMachine.BalanceLaws:
    flux_second_order!,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

abstract type DryAtmosLinearModel <: BalanceLaw end

struct DryAtmosAcousticGravityLinearModel{M} <: DryAtmosLinearModel
    atmos::M
    function DryAtmosAcousticGravityLinearModel(atmos::M) where {M}
        if atmos.physics.ref_state === NoReferenceState()
            error("DryAtmosAcousticGravityLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end

"""
    Declaration of state variables

    vars_state returns a NamedTuple of data types.
"""
function vars_state(lm::DryAtmosLinearModel, ::Prognostic, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end
vars_state(lm::DryAtmosLinearModel, st::Auxiliary, FT) =
    vars_state(lm.atmos, st, FT)

"""
    Initialization of state variables

    init_state_xyz! sets up the initial fields within our state variables
    (e.g., prognostic, auxiliary, etc.), however it seems to not initialized
    the gradient flux variables by default.
"""
init_state_auxiliary!(lm::DryAtmosLinearModel, aux::Vars, geom::LocalGeometry) =
    nothing

init_state_prognostic!(
    lm::DryAtmosLinearModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
) = nothing

function update_auxiliary_state!(
    dg::DGModel,
    lm::DryAtmosLinearModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return false
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
    FT = eltype(state)
    ref = aux.ref_state

    flux.ρ = state.ρu
    pL = linearized_pressure(state.ρ, state.ρe, aux.Φ)
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu

    return nothing
end

flux_second_order!(
    lm::DryAtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) = nothing

function flux_second_order!(
    lm::DryAtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end

integral_load_auxiliary_state!(
    lm::DryAtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
integral_set_auxiliary_state!(lm::DryAtmosLinearModel, aux::Vars, integ::Vars) =
    nothing
reverse_integral_load_auxiliary_state!(
    lm::DryAtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
reverse_integral_set_auxiliary_state!(
    lm::DryAtmosLinearModel,
    aux::Vars,
    integ::Vars,
) = nothing

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
    if Dir === VerticalDirection || Dir === EveryDirection
        ∇Φ = aux.∇Φ
        source.ρu -= state.ρ * ∇Φ
        if !total_energy
            source.ρe -= state.ρu' * ∇Φ
        end
    end
    nothing
end

"""
    Boundary conditions
"""
boundary_conditions(lm::DryAtmosLinearModel) = (5, 6)
function boundary_state!(
    nf::NumericalFluxFirstOrder,
    bc,
    lm::DryAtmosLinearModel,
    args...,
)
    boundary_state!(nf, bc, lm.atmos, args...)
end
function boundary_state!(
    nf::NumericalFluxSecondOrder,
    bc,
    lm::DryAtmosLinearModel,
    args...,
)
    nothing
end