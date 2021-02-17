"""
    initial condition is given by ρ0 = Y{m,l}(θ, λ)
    test: ∇^4_horz ρ0 = l^2(l+1)^2/r^4 ρ0 where r=a+z
"""

function initial_condition!(
    problem::ConstantHyperDiffusion{dim, dir},
    state,
    aux,
    x,
    t,
) where {dim, dir}
    @inbounds begin
        FT = eltype(state)
        # import planet paraset
        _a::FT = planet_radius(param_set)

        φ = latitude(SphericalOrientation(), aux)
        λ = longitude(SphericalOrientation(), aux)
        r = norm(aux.coord)
        z = r - _a

        l = Int64(problem.l)
        m = Int64(problem.m)

        c = get_c(l, r)
        # state.ρ = calc_Ylm(φ, λ, l, m) * exp(-problem.D*c*t)
        # state.ρ = calc_Ylm(φ, λ, l, m) * exp(-problem.D*c*t) * exp(-z/30.0e3)
        state.ρ = cos(z/30.0e3)
    end
end