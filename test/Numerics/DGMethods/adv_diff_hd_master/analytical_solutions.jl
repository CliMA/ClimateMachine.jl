function compute_analytical(::SphericalHarm, dg, Q0; D = nothing) 
    # rhs analytical solution for spherical harmonics Y_{l,m}
    ∂Q∂t_anal = .- dg.state_auxiliary.c * D .* Q0
end