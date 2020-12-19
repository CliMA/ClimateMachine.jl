export AtmosFVConstantBalanced

using ..DGMethods.FVReconstructions: AbstractReconstruction, FVConstant
using StaticArrays: SUnitRange
import ..DGMethods.FVReconstructions: width

struct AtmosFVConstantBalanced{M} <: AbstractReconstruction
    atmos::M
end
width(::AtmosFVConstantBalanced) = 1

function (recon::AtmosFVConstantBalanced)(
    state_bot,
    state_top,
    cell_states::SVector{3},
    cell_aux::SVector{3},
    cell_weights,
)
    FT = eltype(state_bot)
    cell_state = cell_states[2]

    FVConstant()(
        state_bot,
        state_top,
        cell_states[SUnitRange(2, 2)],
        cell_aux,
        cell_weights,
    )

    m = recon.atmos
    hydrostatic_reconstruction!(
        Vars{vars_state(m, Primitive(), FT)}(state_bot),
        Vars{vars_state(m, Primitive(), FT)}(state_top),
        Vars{vars_state(m, Prognostic(), FT)}(cell_state),
        Vars{vars_state(m, Auxiliary(), FT)}.(cell_aux),
    )
end
function hydrostatic_reconstruction!(
    state_bot::Vars,
    state_top::Vars,
    cell_state::Vars,
    cell_aux::SVector{3},
)
    ΔΦ_bot = cell_aux[2].orientation.Φ - cell_aux[1].orientation.Φ
    ΔΦ_top = cell_aux[3].orientation.Φ - cell_aux[2].orientation.Φ
    state_bot.p += cell_state.ρ * ΔΦ_bot / 2
    state_top.p -= cell_state.ρ * ΔΦ_top / 2
end

# boundary reconstruction
function (recon::AtmosFVConstantBalanced)(
    state_bot,
    state_top,
    cell_states::SVector{1},
    cell_aux::SVector{2},
    cell_weights,
)
    FT = eltype(state_bot)
    cell_state = cell_states[1]

    FVConstant()(state_bot, state_top, cell_states, cell_aux, cell_weights)

    m = recon.atmos
    hydrostatic_reconstruction!(
        Vars{vars_state(m, Primitive(), FT)}(state_bot),
        Vars{vars_state(m, Primitive(), FT)}(state_top),
        Vars{vars_state(m, Prognostic(), FT)}(cell_state),
        Vars{vars_state(m, Auxiliary(), FT)}.(cell_aux),
    )
end
function hydrostatic_reconstruction!(
    state_bot::Vars,
    state_top::Vars,
    cell_state::Vars,
    cell_aux::SVector{2},
)
    # one-sided approximation near the boundary
    ΔΦ_bot = cell_aux[2].orientation.Φ - cell_aux[1].orientation.Φ
    ΔΦ_top = cell_aux[2].orientation.Φ - cell_aux[1].orientation.Φ
    state_bot.p += cell_state.ρ * ΔΦ_bot / 2
    state_top.p -= cell_state.ρ * ΔΦ_top / 2
end
