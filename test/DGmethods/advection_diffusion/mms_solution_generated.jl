const ν_exact = 1//10000
const a_exact = 1
const b_exact = 1
ρ_g(t, φ, θ, r) = sin(θ)^4*sin(φ)^2*sin(pi*(r - 1))*cos(t) + 3
Sρ_g(t, φ, θ, r) = (r*sin(θ)^3*sin(φ) - 2*sin(θ)^5*sin(φ)^2*sin(pi*r)^3*cos(t)*cos(φ) + 2*sin(θ)^5*sin(pi*r)^3*cos(t)*cos(φ)^3 - 7*sin(θ)^4*sin(φ)^3*sin(pi*r)^2*cos(t)*cos(θ) + 6*sin(θ)*sin(pi*r)^2*cos(φ) + 9*sin(φ)*sin(pi*r)*cos(θ))*sin(t)*sin(θ)*sin(φ)*sin(pi*r)/r
