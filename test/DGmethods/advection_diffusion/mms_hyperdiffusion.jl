# This file generates the solution used in method of manufactured solutions
using LinearAlgebra, SymPy, Printf

@syms φ θ r t real=true
ν = 1 // 10000

output = open("mms_solution_generated.jl", "w")

a = 1
b = 1

@printf output "const ν_exact = %s\n" ν
@printf output "const a_exact = %s\n" a
@printf output "const b_exact = %s\n" b

ρ = cos(t) * sin(φ) ^ 2 * sin(θ) ^ 4 * sin(pi * (r - a) / b) + 3
u = sin(t) * cos(φ) ^ 2 * sin(θ) ^ 3 * sin(pi * (r - a) / b) ^ 3
v = sin(t) * sin(φ) ^ 2 * sin(θ) ^ 2 * sin(pi * (r - a) / b) ^ 2
w = sin(t) * cos(φ) ^ 3 * sin(θ) ^ 1 * sin(pi * (r - a) / b) ^ 1


∇ρ = (diff(sin(θ) * v * ρ, θ) + diff(u * ρ, φ)) / (r * sin(θ))

dρdt = simplify(diff(ρ, t) + ∇ρ)

@printf output "ρ_g(t, φ, θ, r) = %s\n" ρ
@printf output "Sρ_u(t, φ, θ, r) = %s\n" u
@printf output "Sρ_v(t, φ, θ, r) = %s\n" v
@printf output "Sρ_w(t, φ, θ, r) = %s\n" w
@printf output "Sρ_g(t, φ, θ, r) = %s\n" dρdt

close(output)
