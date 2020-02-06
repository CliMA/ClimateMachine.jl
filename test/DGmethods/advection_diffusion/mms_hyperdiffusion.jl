# This file generates the solution used in method of manufactured solutions
using LinearAlgebra, SymPy, Printf

@syms φ θ r t real=true
ν = 1 // 10000

output = open("mms_solution_generated.jl", "w")

@printf output "const ν_exact = %s\n" ν

ρ = cos(t) * sin(φ) * sin(θ) * cos(π * r) + 3

Δρ = (diff(sin(θ) * diff(ρ, θ), θ) + diff(ρ, φ, φ) / sin(θ)) / (r ^ 2 * sin(θ))
Δ²ρ = (diff(sin(θ) * diff(Δρ, θ), θ) + diff(Δρ, φ, φ) / sin(θ)) / (r ^ 2 * sin(θ))

dρdt = simplify(diff(ρ, t) + ν * Δ²ρ)

@printf output "ρ_g(t, φ, θ, r) = %s\n" ρ
@printf output "Sρ_g(t, φ, θ, r) = %s\n" dρdt

close(output)
