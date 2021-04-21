abstract type AbstractDiffusion  <: AbstractPhysicsComponent end

struct ConstantViscosity{𝒯} <: AbstractTerm{𝒯} end

@inline function calc_diffusive_flux_argument!(grad, ::Nothing, _...) 
    grad.∇ρ = 0
    grad.∇u = @SVector [0, 0, 0]
    grad.∇θ = 0

    return nothing
end

@inline function calc_diffusive_flux_argument!(grad, diff::ConstantViscosity, state, aux, physics)  
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

@inline function calc_diffusive_flux!(gradflux, ::ConstantViscosity, grad, state, aux, physics)
    μ = physics.params.μ * I
    ν = physics.params.ν * I
    κ = physics.params.κ * I

    gradflux.μ∇ρ = -μ * grad.∇ρ
    gradflux.ν∇u = -ν * grad.∇u
    gradflux.κ∇θ = -κ * grad.∇θ

    return nothing
end