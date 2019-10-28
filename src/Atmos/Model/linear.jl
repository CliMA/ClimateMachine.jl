abstract type AtmosLinearModel <: BalanceLaw
end


vars_state(lm::AtmosLinearModel, FT) = vars_state(lm.atmos,FT)
vars_gradient(lm::AtmosLinearModel, FT) = @vars()
vars_diffusive(lm::AtmosLinearModel, FT) = @vars()
vars_aux(lm::AtmosLinearModel, FT) = vars_aux(lm.atmos,FT)
vars_integrals(lm::AtmosLinearModel, FT) = @vars()


update_aux!(dg::DGModel, lm::AtmosLinearModel, Q::MPIStateArray, auxstate::MPIStateArray, t::Real) = nothing
integrate_aux!(lm::AtmosLinearModel, integ::Vars, state::Vars, aux::Vars) = nothing
flux_diffusive!(lm::AtmosLinearModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real) = nothing
function wavespeed(lm::AtmosLinearModel, nM, state::Vars, aux::Vars, t::Real)
  ref = aux.ref_state
  return soundspeed_air(ref.T)
end

function boundary_state!(nf::NumericalFluxNonDiffusive, lm::AtmosLinearModel,
                         x...)
  atmos_boundary_state!(nf, NoFluxBC(), lm.atmos, x...)
end
function boundary_state!(nf::CentralNumericalFluxDiffusive, lm::AtmosLinearModel, x...)
  nothing
end
init_aux!(lm::AtmosLinearModel, aux::Vars, geom::LocalGeometry) = nothing
init_state!(lm::AtmosLinearModel, state::Vars, aux::Vars, coords, t) = nothing


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
  atmos::M
end
function flux_nondiffusive!(lm::AtmosAcousticLinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  ref = aux.ref_state
  e_pot = gravitational_potential(lm.atmos.orientation, aux)

  flux.ρ = state.ρu
  # TODO: use MoistThermodynamics.linearized_air_pressure 
  # need to avoid dividing then multiplying by ρ
  pL = state.ρ * FT(R_d) * FT(T_0) + FT(R_d) / FT(cv_d) * (state.ρe - state.ρ * e_pot)
  flux.ρu += pL*I
  flux.ρe = ((ref.ρe + ref.p)/ref.ρ - e_pot)*state.ρu
  nothing
end
function source!(lm::AtmosAcousticLinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  nothing
end

struct AtmosAcousticGravityLinearModel{M} <: AtmosLinearModel
  atmos::M
end
function flux_nondiffusive!(lm::AtmosAcousticGravityLinearModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  ref = aux.ref_state
  e_pot = gravitational_potential(lm.atmos.orientation, aux)

  flux.ρ = state.ρu
  pL = state.ρ * FT(R_d) * FT(T_0) + FT(R_d) / FT(cv_d) * (state.ρe - state.ρ * e_pot)
  flux.ρu += pL*I
  flux.ρe = ((ref.ρe + ref.p)/ref.ρ)*state.ρu
  nothing
end
function source!(lm::AtmosAcousticGravityLinearModel, source::Vars, state::Vars, aux::Vars, t::Real)
  ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
  source.ρu = state.ρ * ∇Φ
  nothing
end

function NumericalFluxes.numerical_flux_nondiffusive!(nf::Upwind,
                                                      bl::AtmosAcousticLinearModel,
                                                      F::MArray, nM::SVector,
                                                      QM::MArray, auxM::MArray,
                                                      QP::MArray, auxP::MArray,
                                                      t)

  FT = eltype(QM)
  NumericalFluxes.numerical_flux_nondiffusive!(nf, bl,
                                               Vars{vars_state(bl, FT)}(F),
                                               nM,
                                               Vars{vars_state(bl, FT)}(QM),
                                               Vars{vars_aux(bl, FT)}(auxM),
                                               Vars{vars_state(bl, FT)}(QP),
                                               Vars{vars_aux(bl, FT)}(auxP),
                                               t)
end

# Note: this assumes that the reference state is continuous across element
# boundaries
function NumericalFluxes.numerical_flux_nondiffusive!(::Upwind,
                                                      lm::AtmosAcousticLinearModel,
                                                      F::Vars, nM::SVector,
                                                      QM::Vars, auxM::Vars,
                                                      QP::Vars, auxP::Vars,
                                                      t)

  FT = eltype(QM)

  e_pot = gravitational_potential(lm.atmos.orientation, auxM)
  ref = auxM.ref_state

  # Coefficients for the flux matrix
  α = FT(R_d) * FT(T_0) - e_pot * FT(R_d) / FT(cv_d)
  β = FT(R_d) / FT(cv_d)
  γ = (ref.ρe + ref.p)/ref.ρ - e_pot

  # wave speed
  λ = sqrt(β * γ + α)

  #=
  # Matrix for the flux is:
  Tn = [1 0    0     0   0;
        0 n[1] n[2] n[3] 0;
        0 0    0    0    1]

  B = [0 1 0;
       α 0 β;
       0 γ 0]

  An = Tn' * B * Tn

  # The upwinding is based on the following eigenvalue decomposition of B
  V = [-1  β 1;
        λ  0 λ;
       -γ -α γ]

  W = [-α / (2α + 2γ * β)   1 / 2λ  -β / (2α + 2γ * β);
       2γ / (2α + 2γ * β)   0       -2 / (2α + 2γ * β);
        α / (2α + 2γ * β)   1 / 2λ   β / (2α + 2γ * β)]

  @assert B ≈ V * Diagonal([-λ, 0, λ]) * W
  =#

  # rotated state vector based on outward normal
  ρM, ρuM, ρeM = QM.ρ, nM' * QM.ρu, QM.ρe
  ρP, ρuP, ρeP = QP.ρ, nM' * QP.ρu, QP.ρe

  # Left eigenvector entries
  δ1 = -α / (2α + 2γ * β)
  δ2 =  1 / 2λ
  δ3 = -β / (2α + 2γ * β)

  # incoming wave
  ωP = -λ * ( δ1 * ρP + δ2 * ρuP + δ3 * ρeP)

  # outgoing wave
  ωM =  λ * (-δ1 * ρM + δ2 * ρuM - δ3 * ρeM)

  # compute the upwind flux using the right eigenvectors and rotate back based
  # on the outward normal
  F.ρ = ωM - ωP
  F.ρu = λ * (ωP + ωM) * nM
  F.ρe = γ * (ωM - ωP)
end
