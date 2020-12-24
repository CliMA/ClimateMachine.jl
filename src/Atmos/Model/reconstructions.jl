using KernelAbstractions.Extras: @unroll

export HBFVReconstruction

using ..DGMethods.FVReconstructions: AbstractReconstruction
import ..DGMethods.FVReconstructions: width

struct HBFVReconstruction{M} <: AbstractReconstruction
    _atmo::M
    _recon::AbstractReconstruction
end

width(hb_recon::HBFVReconstruction) = width(hb_recon._recon)

function (hb_recon::HBFVReconstruction)(
    state_bot,
    state_top,
    cell_states::SVector{D},
    cell_weights,
) where {D}


    FT = eltype(state_bot)
    vars_prim = Vars{vars_state(hb_recon._atmo, Primitive(), FT)}
    _grav = FT(grav(hb_recon._atmo.param_set))

    # stencil info
    stencil_diameter = D
    stencil_width = div(D - 1, 2)
    stencil_center = stencil_width + 1
    # save the pressure states
    ps = similar(state_bot, stencil_diameter)
    @unroll for i in 1:stencil_diameter
        ps[i] = vars_prim(cell_states[i]).p
    end

    # construct reference pressure steates and update pressure states
    p_ref = ps[stencil_center]
    p_bot_ref =
        p_ref +
        vars_prim(cell_states[stencil_center]).ρ *
        cell_weights[stencil_center] *
        _grav / 2
    p_top_ref =
        p_ref -
        vars_prim(cell_states[stencil_center]).ρ *
        cell_weights[stencil_center] *
        _grav / 2

    vars_prim(cell_states[stencil_center]).p -= p_ref

    p⁺_ref, p⁻_ref = p_ref, p_ref
    @unroll for i in 1:stencil_width
        # stencil_center - i , stencil_center - i + 1
        p⁻_ref +=
            vars_prim(cell_states[stencil_center - i + 1]).ρ *
            cell_weights[stencil_center - i + 1] *
            _grav / 2 +
            vars_prim(cell_states[stencil_center - i]).ρ *
            cell_weights[stencil_center - i] *
            _grav / 2
        vars_prim(cell_states[stencil_center - i]).p -= p⁻_ref

        # stencil_center + i - 1 , stencil_center + i
        p⁺_ref -=
            vars_prim(cell_states[stencil_center + i - 1]).ρ *
            cell_weights[stencil_center + i - 1] *
            _grav / 2 +
            vars_prim(cell_states[stencil_center + i]).ρ *
            cell_weights[stencil_center + i] *
            _grav / 2
        vars_prim(cell_states[stencil_center + i]).p -= p⁺_ref
    end




    hb_recon._recon(state_bot, state_top, cell_states, cell_weights)

    vars_prim(state_bot).p += p_bot_ref
    vars_prim(state_top).p += p_top_ref



    # reverse the pressure states back
    @unroll for i in 1:stencil_diameter
        vars_prim(cell_states[i]).p = ps[i]
    end
end
