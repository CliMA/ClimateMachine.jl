export AtmosFilterPerturbations

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
)
    # copy the whole state
    parent(filter_state) .= parent(state)
    # remove reference state
    filter_state.ρ -= aux.ref_state.ρ
    filter_state.ρe -= aux.ref_state.ρe
    if !(target.atmos.moisture isa DryModel)
        filter_state.moisture.ρq_tot -= aux.ref_state.ρq_tot
    end
end
function compute_filter_result!(
    target::AtmosFilterPerturbations,
    state::Vars,
    filter_state::Vars,
    aux::Vars,
)
    # copy the whole filter state
    parent(state) .= parent(filter_state)
    # add reference state
    state.ρ += aux.ref_state.ρ
    state.ρe += aux.ref_state.ρe
    if !(target.atmos.moisture isa DryModel)
        state.moisture.ρq_tot += aux.ref_state.ρq_tot
    end
end
