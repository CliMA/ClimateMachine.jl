using CLIMA.PlanetParameters
export PeriodicBC, NoFluxBC, InitStateBC, DYCOMS_BC, RayleighBenardBC

function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive, bc,
                                        atmos::AtmosModel, F⁺, state⁺, diff⁺,
                                        aux⁺, n⁻, F⁻, state⁻, diff⁻, aux⁻,
                                        bctype, t, state1⁻, diff1⁻, aux1⁻)
  FT = eltype(F⁺)
  atmos_boundary_state!(nf, bc, atmos, state⁺, diff⁺, aux⁺, n⁻,
                        state⁻, diff⁻, aux⁻, bctype, t,
                        state1⁻, diff1⁻, aux1⁻)
  fill!(parent(F⁺), -zero(FT))
  flux_diffusive!(atmos, F⁺, state⁺, diff⁺, aux⁺, t)
end

#TODO: figure out a better interface for this.
# at the moment we can just pass a function, but we should do something better
# need to figure out how subcomponents will interact.
function atmos_boundary_state!(::Rusanov, f::Function, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, _...)
  f(stateP, auxP, nM, stateM, auxM, bctype, t)
end

function atmos_boundary_state!(::CentralNumericalFluxDiffusive, f::Function,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  f(stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

# lookup boundary condition by face
function atmos_boundary_state!(nf::Rusanov, bctup::Tuple, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, _...)
  atmos_boundary_state!(nf, bctup[bctype], m, stateP, auxP, nM, stateM, auxM,
                        bctype, t)
end

function atmos_boundary_state!(nf::CentralNumericalFluxDiffusive,
                               bctup::Tuple, m::AtmosModel, stateP::Vars,
                               diffP::Vars, auxP::Vars, nM, stateM::Vars,
                               diffM::Vars, auxM::Vars, bctype, t, _...)
  atmos_boundary_state!(nf, bctup[bctype], m, stateP, diffP, auxP, nM, stateM,
                        diffM, auxM, bctype, t)
end


abstract type BoundaryCondition
end

"""
    PeriodicBC <: BoundaryCondition

Assume that the topology is periodic and hence nothing special needs to be done at the boundaries.
"""
struct PeriodicBC <: BoundaryCondition end

# TODO: assert somewhere that the topology is actually periodic when using those
atmos_boundary_state!(_, ::PeriodicBC, _...) = nothing

"""
    NoFluxBC <: BoundaryCondition

Set the momentum at the boundary to be zero.

# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)

"""
struct NoFluxBC <: BoundaryCondition
end

function atmos_boundary_state!(::Rusanov, bc::NoFluxBC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, _...)
  FT = eltype(stateM)
  stateP.ρ = stateM.ρ
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
end

function atmos_boundary_state!(::CentralNumericalFluxDiffusive, bc::NoFluxBC,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  FT = eltype(stateM)
  stateP.ρ = stateM.ρ
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
  
  fill!(getfield(diffP, :array), FT(0))
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function. This is
mainly useful for cases where the problem has an explicit solution.

# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
"""
struct InitStateBC <: BoundaryCondition
end
function atmos_boundary_state!(::Rusanov, bc::InitStateBC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, _...)
  init_state!(m, stateP, auxP, auxP.coord, t)
end
function atmos_boundary_state!(::CentralNumericalFluxDiffusive, bc::InitStateBC,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  init_state!(m, stateP, auxP, auxP.coord, t)
end


"""
  DYCOMS_BC <: BoundaryCondition
  Prescribes boundary conditions for Dynamics of Marine Stratocumulus Case
"""
struct DYCOMS_BC{FT} <: BoundaryCondition
  C_drag::FT
  LHF::FT
  SHF::FT
end
function atmos_boundary_state!(::Rusanov, bc::DYCOMS_BC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, state1::Vars, aux1::Vars)
  # stateM is the 𝐘⁻ state while stateP is the 𝐘⁺ state at an interface.
  # at the boundaries the ⁻, minus side states are the interior values
  # state1 is 𝐘 at the first interior nodes relative to the bottom wall
  FT = eltype(stateP)
  # Get values from minus-side state
  ρM = stateM.ρ
  UM, VM, WM = stateM.ρu
  EM = stateM.ρe
  QTM = stateM.moisture.ρq_tot
  uM, vM, wM  = UM/ρM, VM/ρM, WM/ρM
  q_totM = QTM/ρM
  UnM = nM[1] * UM + nM[2] * VM + nM[3] * WM

  # Assign reflection wall boundaries (top wall)
  stateP.ρu = SVector(UM - 2 * nM[1] * UnM,
                      VM - 2 * nM[2] * UnM,
                      WM - 2 * nM[3] * UnM)

  # Assign scalar values at the boundaries
  stateP.ρ = ρM
  stateP.moisture.ρq_tot = QTM
end
function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                        bc::DYCOMS_BC, atmos::AtmosModel,
                                        F⁺, state⁺, diff⁺, aux⁺, n⁻,
                                        F⁻, state⁻, diff⁻, aux⁻,
                                        bctype, t,
                                        state1⁻, diff1⁻, aux1⁻)
  FT = eltype(state⁺)

  # state⁻ is the 𝐘⁻ state while state⁺ is the 𝐘⁺ state at an interface.
  # at the boundaries the ⁻, minus side states are the interior values
  # state1⁻ is 𝐘 at the first interior nodes relative to the bottom wall
  # Get values from minus-side state
  ρ⁻ = state⁻.ρ
  U⁻, V⁻, W⁻ = state⁻.ρu
  E⁻ = state⁻.ρe
  QT⁻ = state⁻.moisture.ρq_tot
  u⁻, v⁻, w⁻  = U⁻/ρ⁻, V⁻/ρ⁻, W⁻/ρ⁻
  q_tot⁻ = QT⁻/ρ⁻
  Un⁻ = n⁻[1] * U⁻ + n⁻[2] * V⁻ + n⁻[3] * W⁻

  # Assign reflection wall boundaries (top wall)
  state⁺.ρu = SVector(U⁻ - 2 * n⁻[1] * Un⁻,
                      V⁻ - 2 * n⁻[2] * Un⁻,
                      W⁻ - 2 * n⁻[3] * Un⁻)

  # Assign scalar values at the boundaries
  state⁺.ρ = ρ⁻
  state⁺.moisture.ρq_tot = QT⁻
  # Assign diffusive fluxes at boundaries
  diff⁺ = diff⁻
  if bctype != 1
    flux_diffusive!(atmos, F⁺, state⁺, diff⁺, aux⁺, t)
  else
    # ------------------------------------------------------------------------
    # (<var>_FN) First node values (First interior node from bottom wall)
    # ------------------------------------------------------------------------
    z_FN             = aux1⁻.coord[3]
    ρ_FN             = state1⁻.ρ
    U_FN, V_FN, W_FN = state1⁻.ρu
    E_FN             = state1⁻.ρe
    u_FN, v_FN, w_FN = U_FN/ρ_FN, V_FN/ρ_FN, W_FN/ρ_FN
    windspeed_FN     = sqrt(u_FN^2 + v_FN^2 + w_FN^2)
    q_tot_FN         = state1⁻.moisture.ρq_tot / ρ_FN
    e_int_FN         = E_FN/ρ_FN - windspeed_FN^2/2 - grav*z_FN
    TS_FN            = PhaseEquil(e_int_FN, ρ_FN, q_tot_FN)
    T_FN             = air_temperature(TS_FN)
    q_vap_FN         = q_tot_FN - PhasePartition(TS_FN).liq
    # --------------------------
    # Bottom boundary quantities
    # --------------------------
    z⁻          = aux⁻.coord[3]
    q_tot⁻      = QT⁻/ρ⁻
    windspeed   = sqrt(u⁻^2 + v⁻^2 + w⁻^2)
    e_int⁻      = E⁻/ρ⁻ - windspeed^2/2 - grav*z⁻
    TS⁻         = PhaseEquil(e_int⁻, ρ⁻, q_tot⁻)
    q_vap⁻      = q_tot⁻ - PhasePartition(TS⁻).liq
    T⁻          = air_temperature(TS⁻)
    # ----------------------------------------------------------
    # Extract components of diffusive momentum flux (minus-side)
    # ----------------------------------------------------------
    _, τ⁻ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)

    # ----------------------------------------------------------
    # Boundary momentum fluxes
    # ----------------------------------------------------------
    # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
    # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is replaced by the appropriate flux terms
    C_drag = bc.C_drag
    τ13⁺  = - C_drag * windspeed_FN * u_FN
    τ23⁺  = - C_drag * windspeed_FN * v_FN
    # Assign diffusive momentum and moisture fluxes
    # (i.e. ρ𝛕 terms)
    τ⁺ = SHermitianCompact{3, FT, 6}(SVector(FT(0), τ⁻[2,1], τ13⁺, FT(0), τ23⁺,
                                             FT(0)))

    # ----------------------------------------------------------
    # Boundary moisture fluxes
    # ----------------------------------------------------------
    # really ∇q_tot is being used to store d_q_tot
    d_q_tot⁺  = SVector(FT(0), FT(0), bc.LHF/(LH_v0))

    # ----------------------------------------------------------
    # Boundary energy fluxes
    # ----------------------------------------------------------
    # Assign diffusive enthalpy flux (i.e. ρ(J+D) terms)
    d_h_tot⁺ = SVector(FT(0), FT(0), bc.LHF + bc.SHF)

    flux_diffusive!(atmos, F⁺, state⁺, τ⁺, d_h_tot⁺)
    flux_diffusive!(atmos.moisture, F⁺, state⁺, d_q_tot⁺)
  end
end

"""
  RayleighBenardBC <: BoundaryCondition

# Fields
$(DocStringExtensions.FIELDS)
"""
struct RayleighBenardBC{FT} <: BoundaryCondition
  "Prescribed bottom wall temperature [K]"
  T_bot::FT
  "Prescribed top wall temperature [K]"
  T_top::FT
end
# Rayleigh-Benard problem with two fixed walls (prescribed temperatures)
function atmos_boundary_state!(::Rusanov, bc::RayleighBenardBC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t,_...)
  # Dry Rayleigh Benard Convection
  @inbounds begin
    FT = eltype(stateP)
    ρP = stateM.ρ
    stateP.ρ = ρP
    stateP.ρu = SVector{3,FT}(0,0,0)
    if bctype == 1
      E_intP = ρP * cv_d * (bc.T_bot - T_0)
    else
      E_intP = ρP * cv_d * (bc.T_top - T_0)
    end
    stateP.ρe = (E_intP + ρP * auxP.coord[3] * grav)
    nothing
  end
end
function atmos_boundary_state!(::CentralNumericalFluxDiffusive, bc::RayleighBenardBC,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  # Dry Rayleigh Benard Convection
  @inbounds begin
    FT = eltype(stateM)
    ρP = stateM.ρ
    stateP.ρ = ρP
    stateP.ρu = SVector{3,FT}(0,0,0)
    if bctype == 1
      E_intP = ρP * cv_d * (bc.T_bot - T_0)
    else
      E_intP = ρP * cv_d * (bc.T_top - T_0)
    end
    stateP.ρe = (E_intP + ρP * auxP.coord[3] * grav)
    diffP.∇h_tot = SVector(diffP.∇h_tot[1], diffP.∇h_tot[2], FT(0))
    nothing
  end
end
