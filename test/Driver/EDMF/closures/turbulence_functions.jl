function compute_buoyancy_gradients(
    ss::SingleStack{FT, N},
    m::MixingLengthModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
    δ::FT,
    εt::FT,
) where {FT, N}
    # think how to call subdomain statistics here to get cloudy and dry values of T if you nee them 
    # buoyancy gradients via chain-role
    ∂b∂ρ = - g/gm.ρ
    #                  <-------- ∂ρ∂T -------->*<----- ∂T∂e_int ---------->
    ∂ρ∂e_int_dry    = - R_d*gm_a.p_0/(R_m*T*T)/((1-en_q_tot)*cv_d+q_vap *cv_v)
    #                  <-------- ∂ρ∂T --------->*<----- ∂T∂e_int ---------->
    ∂ρ∂e_int_cloudy = - (R_d*gm_a.p_0/(R_m*T*T)/((1-en_q_tot)*cv_d+q_vap *cv_v+q_liq*cv_l+ q_ice*cv_i)
                       + gm_a.p_0/(R_m*R_m*T)*ϵ_v*R_d/(cv_v*(T-T0)+e_int_i0) )  
    #                    <----- ∂ρ∂Rm ------->*<------- ∂Rm∂e_int ---------->
    
    ∂ρ∂e_int = (en_a.cld_frac * ∂ρ∂e_int_cloudy + (1-en_a.cld_frac) * ∂ρ∂e_int_dry)
    ∂ρ∂q_tot = R_d*gm_a.p_0/(R_m*R_m*T)
    # apply chain role
    ∂b∂z_e_int = ∂b∂ρ * ∂ρ∂e_int * ∂e_int∂z
    ∂b∂z_q_tot = ∂b∂ρ * ∂ρ∂q_tot * ∂q_tot∂z
    return ∂b∂z_e_int, ∂b∂z_q_tot
end;

function gradient_Richardson_number(∂b∂z_e_int, TKE_Shear, ∂b∂z_q_tot, minval)
    Grad_Ri = min(∂b∂z_e_int/max(TKE_Shear, eps(FT)) + ∂b∂z_q_tot/max(TKE_Shear, eps(FT)) , minval)
    return Grad_Ri
end;

function turbulent_Prandtl_number(Pr_n, Grad_Ri, obukhov_length, a_empirical, b_empirical, c_empirical)
    if unstable(obukhov_length)
      Pr_z = Pr_n
    else
      Pr_z = Pr_n*(2*Grad_Ri/
                        (1+(a_empirical/b_empirical)*Grad_Ri -sqrt( (1+(a_empirical/c_empirical)*Grad_Ri)^2 - 4*Grad_Ri ) ) )
    end
    return Pr_z
end;

