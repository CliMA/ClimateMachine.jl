abstract type AbstractDiffusion  <: AbstractPhysicsComponent end

Base.@kwdef struct ConstantViscosity{FT} <: AbstractDiffusion
    μ::FT
    ν::FT
    κ::FT
end

@inline function calc_diffusive_flux_argument!(grad, ::Nothing, _...) 
    grad.∇ρ = 0
    grad.∇u = @SVector [0, 0, 0]
    grad.∇θ = 0

    return nothing
end

@inline function calc_diffusive_flux_argument!(grad, diff::ConstantViscosity, state::Vars, aux::Vars, t::Real)  
    ρ = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ

    u = ρu / ρ
    θ = ρθ / ρ

    grad.∇ρ = ρ
    grad.∇u = u
    grad.∇θ = θ

    return nothing
end

@inline function calc_diffusive_flux!(gradflux, ::Nothing, _...)
    gradflux.μ∇ρ = @SVector [0, 0, 0]
    gradflux.ν∇u = @SMatrix zeros(3,3)
    gradflux.κ∇θ = @SVector [0, 0, 0]

    return nothing
end

@inline function calc_diffusive_flux!(gradflux, diff::ConstantViscosity, grad::Grad, state::Vars, aux::Vars, t::Real)  
    μ = diff.μ * I
    ν = diff.ν * I
    κ = diff.κ * I

    gradflux.μ∇ρ = -μ * grad.∇ρ
    gradflux.ν∇u = -ν * grad.∇u
    gradflux.κ∇θ = -κ * grad.∇θ

    return nothing
end