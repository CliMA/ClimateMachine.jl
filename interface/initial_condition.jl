# """
#     initial condition is given by ρ0 = Y{m,l}(θ, λ)
#     test: ∇^4_horz ρ0 = l^2(l+1)^2/r^4 ρ0 where r=a+z
# """

# function initial_condition!(
#     problem::HyperDiffusionCubedSphereProblem{FT},
#     state,
#     aux,
#     x,
#     t,
# ) where {FT}
#     @inbounds begin
#         # import planet paraset
#         _a::FT = planet_radius(param_set)

#         φ = latitude(SphericalOrientation(), aux)
#         λ = longitude(SphericalOrientation(), aux)
#         r = norm(aux.coord)
#         z = r - _a

#         l = Int64(problem.l)
#         m = Int64(problem.m)

#         c = get_c(l, r)
#         state.ρ = calc_Ylm(φ, λ, l, m) * exp(-problem.D*c*t)
#     end
# end