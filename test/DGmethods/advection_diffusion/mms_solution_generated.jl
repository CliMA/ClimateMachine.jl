const ν_exact = 1//100
const a_exact = 1
const b_exact = 1
ρ_g(t, φ, θ, r) = sin(θ)^4*sin(φ)^2*sin(pi*(r - 1))*cos(t) + 3
Sρ_g(t, φ, θ, r) = (25*r^4*sin(t)*sin(θ)^4*sin(φ)^2 - 100*sin(θ)^4*sin(φ)^2*cos(t) + 78*sin(θ)^2*sin(φ)^2*cos(t) + 13*sin(θ)^2*cos(t) - 8*cos(t))*sin(pi*r)/(25*r^4)
