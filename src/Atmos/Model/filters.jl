export AtmosFilterPerturbations
export AtmosSpecificFilterPerturbations

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
    filter_state.energy.ρe -= aux.ref_state.ρe
    if !(moisture_model(target.atmos) isa DryModel)
        filter_state.moisture.ρq_tot -= aux.ref_state.ρq_tot
    end
    if (moisture_model(target.atmos) isa NonEquilMoist)
        filter_state.moisture.ρq_liq -= aux.ref_state.ρq_liq
        filter_state.moisture.ρq_ice -= aux.ref_state.ρq_ice
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
    state.energy.ρe += aux.ref_state.ρe
    if !(moisture_model(target.atmos) isa DryModel)
        state.moisture.ρq_tot += aux.ref_state.ρq_tot
    end
    if (moisture_model(target.atmos) isa NonEquilMoist)
        filter_state.moisture.ρq_liq += aux.ref_state.ρq_liq
        filter_state.moisture.ρq_ice += aux.ref_state.ρq_ice
    end
end

struct AtmosSpecificFilterPerturbations{M} <: AbstractFilterTarget
    atmos::M
end

vars_state_filtered(target::AtmosSpecificFilterPerturbations, FT) =
    vars_state_filtered(target.atmos, FT)

function compute_filter_argument!(
    target::AtmosSpecificFilterPerturbations,
    filter_state::Vars,
    state::Vars,
    aux::Vars,
)

    ρ_inv = 1 / state.ρ
    ρ_ref_inv = 1 / aux.ref_state.ρ

    # copy the whole state
    parent(filter_state) .= parent(state) * ρ_inv

    # remove reference state
    filter_state.energy.e -= aux.ref_state.ρe * ρ_ref_inv
    if !(moisture_model(target.atmos) isa DryModel)
        filter_state.moisture.q_tot -= aux.ref_state.ρq_tot * ρ_ref_inv
    end
    if (moisture_model(target.atmos) isa NonEquilMoist)
        filter_state.moisture.q_liq -= aux.ref_state.ρq_liq * ρ_ref_inv
        filter_state.moisture.q_ice -= aux.ref_state.ρq_ice * ρ_ref_inv
    end

end

function compute_filter_result!(
    target::AtmosSpecificFilterPerturbations,
    state::Vars,
    filter_state::Vars,
    aux::Vars,
)

    ρ = state.ρ
    ρ_ρ_ref_ratio = ρ / aux.ref_state.ρ

    # copy the whole filter state
    parent(state) .= parent(filter_state) * ρ

    # add reference state
    state.energy.ρe += aux.ref_state.ρe * ρ_ρ_ref_ratio
    if !(moisture_model(target.atmos) isa DryModel)
        state.moisture.ρq_tot += aux.ref_state.ρq_tot * ρ_ρ_ref_ratio
    end
    if (moisture_model(target.atmos) isa NonEquilMoist)
        state.moisture.ρq_liq += aux.ref_state.ρq_liq * ρ_ρ_ref_ratio
        state.moisture.ρq_ice += aux.ref_state.ρq_ice * ρ_ρ_ref_ratio
    end
end
