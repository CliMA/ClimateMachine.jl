module StabilizingDissipations

using StaticArrays
using LinearAlgebra

using ClimateMachine.Mesh.Grids: min_node_distance, HorizontalDirection

import ClimateMachine.Ocean.HydrostaticBoussinesq: diffusivity_tensor, viscosity_tensor

using ClimateMachine.Ocean.HydrostaticBoussinesq: HydrostaticBoussinesqModel

struct StabilizingDissipation{T}
    κʰ_min :: T 
    κʰ_max :: T
    νʰ_min :: T
    νʰ_max :: T
    smoothness_exponent :: T
    minimum_node_spacing :: T
    Δu :: T
    Δθ :: T
end

function StabilizingDissipation(;
                                time_step,
                                Δu,
                                Δθ,
                                minimum_node_spacing,
                                diffusive_cfl = 0.1,
                                κʰ_min = 0,
                                νʰ_min = 0,
                                smoothness_exponent = 2)

    FT = typeof(time_step)

    κʰ_max = diffusive_cfl * minimum_node_spacing^2 / time_step
    νʰ_max = diffusive_cfl * minimum_node_spacing^2 / time_step

    return StabilizingDissipation{FT}(FT(κʰ_min),
                                      FT(κʰ_max),
                                      FT(νʰ_min),
                                      FT(νʰ_max),
                                      FT(smoothness_exponent),
                                      FT(minimum_node_spacing),
                                      FT(Δu),
                                      FT(Δθ))
end

@inline function viscosity_tensor(m::HydrostaticBoussinesqModel{<:StabilizingDissipation}, ∇u)
    ϰ = m.stabilizing_dissipation
    Δh = ϰ.minimum_node_spacing

    # ∇u is a 2 × 3 tensor (eg ∇uʰ, where ∇ is 3D)
    @inbounds begin
        ux = ∇u[1, 1]
        uy = ∇u[1, 2]
        vx = ∇u[2, 1]
        vy = ∇u[2, 2]
    end

    ∇u² = ux^2 + uy^2 + vx^2 + vy^2

    arg = (sqrt(∇u²) * Δh / ϰ.Δu)^ϰ.smoothness_exponent

    νʰ = m.νʰ + ϰ.νʰ_min + (ϰ.νʰ_max - ϰ.νʰ_min) * tanh(arg)

    return Diagonal(@SVector [νʰ, νʰ, m.νᶻ])
end

@inline function diffusivity_tensor(m::HydrostaticBoussinesqModel{<:StabilizingDissipation}, ∇θ)

    @inbounds begin
        θx = ∇θ[1]
        θy = ∇θ[2]
        θz = ∇θ[3]
    end

    # Compute convective adjustement diffusivity
    θz < 0 ? κᶻ = m.κᶜ : κᶻ = m.κᶻ

    # Compute stabilizing dissipation
    ϰ = m.stabilizing_dissipation
    Δh = ϰ.minimum_node_spacing

    ∇θ² = θx^2 + θy^2 + θz^2

    arg = (sqrt(∇θ²) * Δh / ϰ.Δθ)^ϰ.smoothness_exponent

    κʰ = m.κʰ + ϰ.κʰ_min + (ϰ.κʰ_max - ϰ.κʰ_min) * tanh(arg)

    return Diagonal(@SVector [κʰ, κʰ, κᶻ])
end

end # module
