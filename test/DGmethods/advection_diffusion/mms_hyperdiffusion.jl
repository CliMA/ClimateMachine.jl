# This file generates the solution used in method of manufactured solutions
using LinearAlgebra, SymPy, Printf

@syms φ θ r t real=true
ν = 1 // 100

output = open("mms_solution_generated.jl", "w")

a = 10 ^ 6
b = 10 ^ 4

@printf output "const ν_exact = %s\n" ν
@printf output "const a_exact = %s\n" a
@printf output "const b_exact = %s\n" b

ρ = cos(t) * sin(φ) ^ 2 * sin(θ) ^ 4 * sin(pi * (r - a) / b) + 3

Δρ = (diff(sin(θ) * diff(ρ, θ), θ) + diff(ρ, φ, φ) / sin(θ)) / (r ^ 2 * sin(θ))
Δ²ρ = (diff(sin(θ) * diff(Δρ, θ), θ) + diff(Δρ, φ, φ) / sin(θ)) / (r ^ 2 * sin(θ))

dρdt = simplify(diff(ρ, t) - ν * Δρ)

@printf output "ρ_g(t, φ, θ, r) = %s\n" ρ
@printf output "Sρ_g(t, φ, θ, r) = %s\n" dρdt

close(output)
