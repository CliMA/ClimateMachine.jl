const ν_exact = 1//100
ρ_g(t, φ, θ, r) = sin(θ)*sin(φ)*cos(t)*cos(pi*r) + 3
Sρ_g(t, φ, θ, r) = (-50*r^2*sin(t) + cos(t))*sin(θ)*sin(φ)*cos(pi*r)/(50*r^2)
