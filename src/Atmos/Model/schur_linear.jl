import CLIMA.DGmethods: SchurComplement,
                        schur_vars_state_auxiliary,
                        schur_vars_gradient_auxiliary,
                        schur_init_aux!,
                        schur_init_state!,
                        schur_lhs_conservative!,
                        schur_lhs_nonconservative!,
                        schur_rhs_conservative!,
                        schur_rhs_nonconservative!,
                        schur_update_conservative!,
                        schur_update_nonconservative!,
                        schur_lhs_boundary_state!,
                        schur_gradient_boundary_state!,
                        schur_update_boundary_state!,
                        schur_rhs_boundary_state!

using CLIMAParameters.Planet: kappa_d

export AtmosAcousticLinearSchurComplement 

struct AtmosAcousticLinearSchurComplement <: SchurComplement end

function schur_vars_state_auxiliary(::AtmosAcousticLinearSchurComplement, FT)
  @vars begin
    h0::FT
    Φ::FT # FIXME: get this from aux directly
    ∇h0::SVector{3, FT}
  end
end
schur_vars_gradient_auxiliary(::AtmosAcousticLinearSchurComplement, FT) = @vars(h0::FT)

function schur_init_aux!(::AtmosAcousticLinearSchurComplement,
                         lm, schur_aux, aux, geom)
  Φ = gravitational_potential(lm.atmos.orientation, aux)
  schur_aux.h0 = (aux.ref_state.ρe + aux.ref_state.p) / aux.ref_state.ρ - Φ
  schur_aux.Φ = Φ
end

function schur_init_state!(::AtmosAcousticLinearSchurComplement,
                           lm,
                           schur_state, schur_aux, state_rhs, aux)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ = 1 / (1 - kappa_d(param_set))
  schur_state.p = (γ - 1) * (state_rhs.ρe - state_rhs.ρ * (Φ - R_d(param_set) * T_0(param_set) / (γ - 1)))
end

function schur_lhs_conservative!(::AtmosAcousticLinearSchurComplement, 
                                 lm, lhs_flux, schur_state, schur_grad, schur_aux, α)
  lhs_flux.p = -α * schur_grad.∇p
end
function schur_lhs_nonconservative!(::AtmosAcousticLinearSchurComplement,
                                    lm, lhs_state, schur_state, schur_grad, schur_aux, α)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ = 1 / (1 - kappa_d(param_set))
  p = schur_state.p
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  ∇p = schur_grad.∇p
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  lhs_state.p = p / ((γ - 1) * α * (h0 - Δp - Φ)) - α * ∇h0' * ∇p / (h0 - Δp - Φ)
end
function schur_gradient_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   schur_state⁺, n,
                                   schur_state⁻, bctype
                                  )
  #schur_state⁺.p = -schur_state⁻.p
end
function schur_lhs_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   schur_state⁺, schur_grad⁺, schur_aux⁺, n,
                                   schur_state⁻, schur_grad⁻, schur_aux⁻, bctype
                                  )
  schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p, n) * n
end

function schur_rhs_conservative!(::AtmosAcousticLinearSchurComplement, 
                                 balance_law,
                                 rhs_flux, state, schur_aux, α)
  rhs_flux.p = -state.ρu
end
function schur_rhs_nonconservative!(::AtmosAcousticLinearSchurComplement, 
                                    lm,
                                    rhs_state, state, schur_aux, α)
  Φ = schur_aux.Φ
  param_set = lm.atmos.param_set
  γ = 1 / (1 - kappa_d(param_set))
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  h0 = schur_aux.h0
  ∇h0 = schur_aux.∇h0
  Δp = -R_d(param_set) * T_0(param_set) / (γ - 1)
  
  rhs_state.p = (ρe - (Φ + Δp) * ρ) / (α * (h0 - Δp - Φ)) - ∇h0' * ρu / (h0 - Δp - Φ)
end
function schur_rhs_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   state⁺, schur_aux⁺, n,
                                   state⁻, schur_aux⁻, bctype
                                  )
  state⁺.ρu = state⁻.ρu - 2 * dot(state⁻.ρu, n) * n
end

function schur_update_conservative!(::AtmosAcousticLinearSchurComplement, 
                                    balance_law,
                                    lhs_flux, schur_state, schur_grad, schur_aux, state_rhs, α)
  p = schur_state.p
  h0 = schur_aux.h0
  ∇p = schur_grad.∇p
  
  lhs_flux.ρ = -α * (state_rhs.ρu - α * ∇p)
  lhs_flux.ρu -= α * p * I
  lhs_flux.ρe = -α * (h0 * (state_rhs.ρu - α * ∇p))
end
function schur_update_nonconservative!(::AtmosAcousticLinearSchurComplement, 
                                       balance_law,
                                       lhs_state,
                                       schur_state, schur_grad, schur_aux, state_rhs, α)
  lhs_state.ρ = state_rhs.ρ
  lhs_state.ρu = state_rhs.ρu
  lhs_state.ρe = state_rhs.ρe
end

function schur_update_boundary_state!(::AtmosAcousticLinearSchurComplement, 
                                   balance_law,
                                   schur_state⁺, schur_grad⁺, schur_aux⁺, state_rhs⁺, n,
                                   schur_state⁻, schur_grad⁻, schur_aux⁻, state_rhs⁻, bctype
                                  )
  schur_grad⁺.∇p = schur_grad⁻.∇p - 2 * dot(schur_grad⁻.∇p, n) * n
  state_rhs⁺.ρu = state_rhs⁻.ρu - 2 * dot(state_rhs⁻.ρu, n) * n
end
