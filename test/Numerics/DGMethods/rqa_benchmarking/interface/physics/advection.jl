abstract type AbstractAdvection <: AbstractPhysicsComponent end

struct NonLinearAdvection <: AbstractAdvection end

@inline calc_advective_flux!(flux, ::Nothing, _...) = nothing

@inline function calc_advective_flux!(flux, ::NonLinearAdvection, state, aux, t)
    ρ = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ

    flux.ρu += ρu ⊗ ρu / ρ
    flux.ρθ += ρu * ρθ / ρ

    return nothing
end