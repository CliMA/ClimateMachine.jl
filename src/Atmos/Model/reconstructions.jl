using KernelAbstractions.Extras: @unroll

export HBFVReconstruction

using ..DGMethods.FVReconstructions: AbstractReconstruction
import ..DGMethods.FVReconstructions: width

struct HBFVReconstruction{M, R} <: AbstractReconstruction
    _atmo::M
    _recon::R
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
    param_set = parameter_set(hb_recon._atmo)
    _grav = FT(grav(param_set))

    # stencil info
    stencil_diameter = D
    stencil_width = div(D - 1, 2)
    stencil_center = stencil_width + 1
    # save the pressure Δpressure_ref states
    ps = similar(state_bot, Size(stencil_diameter))
    ρgΔz_half = similar(state_bot, Size(stencil_diameter))

    @inbounds begin
        @unroll for i in 1:stencil_diameter
            ps[i] = vars_prim(cell_states[i]).p
            ρgΔz_half[i] =
                vars_prim(cell_states[i]).ρ * _grav * cell_weights[i] / 2
        end

        # construct reference pressure states and update pressure states
        p_ref = ps[stencil_center]
        p_bot_ref = p_ref + ρgΔz_half[stencil_center]
        p_top_ref = p_ref - ρgΔz_half[stencil_center]

        vars_prim(cell_states[stencil_center]).p -= p_ref

        p⁺_ref, p⁻_ref = p_ref, p_ref
        @unroll for i in 1:stencil_width
            # stencil_center - i , stencil_center - i + 1
            p⁻_ref +=
                ρgΔz_half[stencil_center - i + 1] +
                ρgΔz_half[stencil_center - i]
            vars_prim(cell_states[stencil_center - i]).p -= p⁻_ref

            # stencil_center + i - 1 , stencil_center + i
            p⁺_ref -=
                ρgΔz_half[stencil_center + i - 1] +
                ρgΔz_half[stencil_center + i]
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
end
