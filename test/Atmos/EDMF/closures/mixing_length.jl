#### Mixing length model kernels

function mixing_length(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    δ::AbstractArray{FT},
    εt::AbstractArray{FT},
    ) where {FT, N}

    # need to code / use the functions: obukhov_length, ustar, ϕ_m
    ss.edmf.surface
    # Alias convention:
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft
    gm_a = aux
    en_d = diffusive.edmf.environment

    z = gm_a.z
    _grav = FT(grav(ss.param_set))
    ρinv = 1/gm.ρ

    fill!(m.L, 0)

    # precompute
    en_area  = 1 - sum([up[i].ρa for i in 1:N])*ρinv
    w_env    = (gm.ρu[3]  - sum([up[i].ρau[3] for i in 1:N]))*ρinv
    en_ρe = (gm.ρe-sum([up[j].ρae for j in 1:N]))/a_en
    en_ρu = (gm.ρu-sum([up[j].ρae for j in 1:N]))/a_en
    e_pot = gravitational_potential(orientation, aux)# ask about this 
    en_e_int = internal_energy(gm.ρ, ρe, ρu, e_pot)
    en_q_tot = (gm.ρq_tot - sum([up[i].ρaq_tot for i in 1:N]))*ρinv
    ∂e_int∂z = en_d.∇e_int[3]
    ∂q_tot∂z = en_d.∇q_tot[3]

    # TODO: check rank of `en_d.∇u`
    Shear = en_d.∇u[1].^2 + en_d.∇u[2].^2 + en_d.∇u[3].^2 # consider scalar product of two vectors
    tke = en.ρatke*ρinv/en_area
    
    # bflux     = Nishizawa2018.compute_buoyancy_flux(ss.param_set, m.shf, m.lhf, m.T_b, q, ρinv)
    bflux = FT(1)
    θ_surf    = ss.edmf.surface.T_surf
    # ustar = Nishizawa2018.compute_friction_velocity(ss.param_set ,u_ave ,θ_suft ,flux ,Δz ,z_0 ,a ,Ψ_m_tol ,tol_abs ,iter_max)
    ustar = FT(0.28)
    # obukhov_length = Nishizawa2018.monin_obukhov_len(ss.param_set, u, θ_surf, flux)
    obukhov_length = FT(-100)

    # buoyancy related functions
    ∂b∂z, Nˢ_eff = compute_buoyancy_gradients(ss, m, state, diffusive, aux, t)
    Grad_Ri = gradient_Richardson_number(∂b∂z, Shear, FT(0.25)) # this parameter should be exposed in the model
    Pr_z = turbulent_Prandtl_number(FT(1), Grad_Ri, obukhov_length)

    # compute L1 - stability - YAIR missing Nˢ
    # @show(Nˢ_eff)
    if Nˢ_eff>eps(FT)
      m.L[1] = sqrt(m.c_w*tke)/Nˢ_eff
    else
      m.L[1] = eps(FT)
    end

    # compute L2 - law of the wall  - YAIR define tke_surf
    tke_surf = FT(1)
    if obukhov_length < eps(FT)
      m.L[2] = (m.κ * z/(sqrt(tke_surf)/ss.edmf.surface.ustar/ss.edmf.surface.ustar)* m.c_k) * min(
         (FT(1) - FT(100) * z/obukhov_length)^FT(0.2), 1/m.κ)
    else
      m.L[2] = m.κ * z/(sqrt(tke_surf)/ss.edmf.surface.ustar/ss.edmf.surface.ustar)*m.c_k
    end

    # compute L3 - entrainment detrainment sources
    # Production/destruction terms

    a = m.c_m*(Shear - ∂e_int∂z/Pr_z - ∂q_tot∂z/Pr_z)* sqrt(abs(tke))
    # Dissipation term
    b = FT(0)
    # detrainment and turb_entr should of the i'th updraft
    for i in 1:N
      a_up = up[i].ρa/gm.ρ
      w_up = up[i].ρau[3]/up[i].ρa
      b += a_up*w_up*δ[i]/en_area*((w_up-w_env)*(w_up-w_env)/2-tke) - a_up*w_up*(w_up-w_env)*εt[i]*w_env/en_area
    end

    c_neg = m.c_m*tke*sqrt(abs(tke))
    if abs(a) > eps(FT) && 4*a*c_neg > - b^2
              l_entdet = max( -b/FT(2)/a + sqrt(b^2 + 4*a*c_neg)/2/a, FT(0))
    elseif abs(a) < eps(FT) && abs(b) > eps(FT)
              l_entdet = c_neg/b
    else
      l_entdet = FT(0)
    end
    m.L[3] = l_entdet

    frac_upper_bound = FT(0.1) # expose these in the model
    lower_bound = FT(1.5) # expose these in the model
    l = lamb_smooth_minimum(m.L, lower_bound,frac_upper_bound)
    return l
end;