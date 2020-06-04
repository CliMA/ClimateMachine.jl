#### Mixing length model kernels

function mixing_length(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    δ::{FT, N},
    εt::{FT, N},
    ) where {FT, N}

    # need to code / use the functions: obukhov_length, ustar, ϕ_m
    ss.edmf.surface
    # Alias convention:
    gm = state
    en = state
    up = state.edmf.updraft
    gm_s = source
    en_s = source
    up_s = source.edmf.updraft
    en_d = state.edmf.environment.diffusive

    z = gm_a.z
    _grav = FT(grav(param_set))
    ρinv = 1/gm.ρ


    fill!(m.L, 0)

    # precompute
    en_area  = 1 - sum([up[i].ρa for i in 1:N])*ρinv
    w_env    = (gm.ρu[3]  - sum([up[i].ρau[3] for i in 1:N]))*ρinv
    en_e_int = (gm.ρe_int - sum([up[i].ρae_int for i in 1:N]))*ρinv
    en_q_tot = (gm.ρq_tot - sum([up[i].ρaq_tot for i in 1:N]))*ρinv
    ∂e_int∂z = en_d.e_int # check if there is a missing ∇
    ∂q_tot∂z = en_d.q_tot # check if there is a missing ∇

    # TODO: check rank of `en_d.∇u`
    Shear = en_d.∇u[1].^2 + en_d.∇u[2].^2 + en_d.∇u[3].^2 # consider scalar product of two vectors

    # Thermodynamic local variables for mixing length
    ts::FT    = PhaseEquil(param_set ,en_e_int, gm.ρ, en_q_tot)
    Π::FT     = exner(ts)
    lv::FT    = latent_heat_vapor(ts)
    cp_m_::FT = cp_m(ts)
    tke::FT   = sqrt(en.ρatke, FT(0))*ρinv/en_area
    θ_v::FT   = virtual_pottemp(ts)
    T::FT     = air_temperature(ts)
    q         = PhasePartition(ts)
    ϵ_v::FT   = 1 / molmass_ratio(param_set)
    bflux     = Nishizawa2018.compute_buoyancy_flux(param_set, m.shf, m.lhf, m.T_b, q, ρinv)
    θ_surf    = ss.SurfaceModel.T_surf
    ustar = Nishizawa2018.compute_friction_velocity(param_set ,u_ave ,θ_suft ,flux ,Δz ,z_0 ,a ,Ψ_m_tol ,tol_abs ,iter_max)
    obukhov_length = Nishizawa2018.monin_obukhov_len(param_set, u, θ_surf, flux)

    # buoyancy related functions
    ∂b∂z_e_int, ∂b∂z_q_tot, buoyancy_freq = compute_buoyancy_gradients(ss, m,source,state, diffusive, aux, t, direction)
    Grad_Ri = gradient_Richardson_number(∂b∂z_e_int, Shear, ∂b∂z_q_tot, FT(0.25)) # this parameter should be exposed in the model
    Pr_z = turbulent_Prandtl_number(m.Pr_n, Grad_Ri, obukhov_length)

    # compute L1 - stability - YAIR missing buoyancy_freq
    buoyancy_freq = g*en_d.∇θ_v/θ_v
    if buoyancy_freq>FT(0)
      m.L[1] = sqrt(m.c_w*tke)/buoyancy_freq
    else
      m.L[1] = FT(1e-6)
    end

    # compute L2 - law of the wall  - YAIR define tke_surf
    if obukhov_length < FT(0)
      m.L[2] = (m.κ * z/(sqrt(tke_surf)/m.ustar/m.ustar)* m.c_k) * min(
         (FT(1) - FT(100) * z/obukhov_length)^FT(0.2), 1/m.κ)
    else
      m.L[2] = m.κ * z/(sqrt(tke_surf)/m.ustar/m.ustar)*m.c_k
    end

    # compute L3 - entrainment detrainment sources
    # Production/destruction terms
    a = m.c_m*(Shear - ∂b∂z_e_int/Pr_z - ∂b∂z_q_tot/Pr_z)* sqrt(tke)
    # Dissipation term
    b = FT(0)
    # detrainment and turb_entr should of the i'th updraft
    for i in 1:N
      a_up = up[i].ρa/gm.ρ
      w_up = up[i].ρau[3]/up[i].ρa
      b += a_up*w_up*δ[i]/en_area*((w_up-w_env)*(w_up-w_env)/2-tke) - a_up*w_up*(w_up-w_env)*εt[i]*w_env/en_area
    end

    c_neg = m.c_m*tke*sqrt(tke)
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