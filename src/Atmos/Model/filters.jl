export AtmosFilterPerturbations
export AtmosCovariantVelocityFilter

import ..Mesh.Filters:
    AbstractFilterTarget,
    vars_state_filtered,
    compute_filter_argument!,
    compute_filter_result!

struct AtmosFilterPerturbations{M} <: AbstractFilterTarget
    atmos::M
end

vars_state_filtered(target::AtmosFilterPerturbations, FT) =
    vars_state(target.atmos, Prognostic(), FT)

function compute_filter_argument!(
    target::AtmosFilterPerturbations,
    filter_state::Vars,
    state::Vars,
    aux::Vars,
    geo::LocalGeometry,
)
    # copy the whole state
    parent(filter_state) .= parent(state)
    # remove reference state
    filter_state.ρ -= aux.ref_state.ρ
    filter_state.energy.ρe -= aux.ref_state.ρe
    if !(target.atmos.moisture isa DryModel)
        filter_state.moisture.ρq_tot -= aux.ref_state.ρq_tot
    end
    if (target.atmos.moisture isa NonEquilMoist)
        filter_state.moisture.ρq_liq -= aux.ref_state.ρq_liq
        filter_state.moisture.ρq_ice -= aux.ref_state.ρq_ice
    end
end
function compute_filter_result!(
    target::AtmosFilterPerturbations,
    state::Vars,
    filter_state::Vars,
    aux::Vars,
    geo::LocalGeometry,
)
    # copy the whole filter state
    parent(state) .= parent(filter_state)
    # add reference state
    state.ρ += aux.ref_state.ρ
    state.energy.ρe += aux.ref_state.ρe
    if !(target.atmos.moisture isa DryModel)
        state.moisture.ρq_tot += aux.ref_state.ρq_tot
    end
    if (target.atmos.moisture isa NonEquilMoist)
        filter_state.moisture.ρq_liq += aux.ref_state.ρq_liq
        filter_state.moisture.ρq_ice += aux.ref_state.ρq_ice
    end
end

struct AtmosCovariantVelocityFilter{M} <: AbstractFilterTarget
    atmos::M
end

vars_state_filtered(target::AtmosCovariantVelocityFilter, FT) =
    vars_state(target.atmos, Prognostic(), FT)

function compute_filter_argument!(
    target::AtmosCovariantVelocityFilter,
    filter_state::Vars,
    state::Vars,
    aux::Vars,
    geo::LocalGeometry,
)
    parent(filter_state) .= parent(state)

    # Rotate to covariant basis
    ∇ξ = geo.invJ
    filter_state.ρu = ∇ξ * state.ρu
end

function compute_filter_result!(
    target::AtmosCovariantVelocityFilter,
    state::Vars,
    filter_state::Vars,
    aux::Vars,
    geo::LocalGeometry,
)
    parent(state) .= parent(filter_state)

    # Rotate to Cartesian basis
    ∇ξ = geo.invJ
    state.ρu = ∇ξ \ filter_state.ρu
end
