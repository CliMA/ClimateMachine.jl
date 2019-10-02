using CLIMA.PlanetParameters
export PeriodicBC, NoFluxBC, InitStateBC, DYCOMS_BC

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
"""
# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
struct NoFluxBC <: BoundaryCondition
end

function atmos_boundary_state!(::Rusanov, bc::NoFluxBC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, _...)
  DF = eltype(stateM)
  stateP.ρ = stateM.ρ
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
end

function atmos_boundary_state!(::CentralNumericalFluxDiffusive, bc::NoFluxBC,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  DF = eltype(stateM)
  stateP.ρ = stateM.ρ
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
  diffP.ρτ = SVector(DF(0), DF(0), DF(0), DF(0), DF(0), DF(0))
  diffP.ρd_h_tot = SVector(DF(0), DF(0), DF(0))
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function. This is
mainly useful for cases where the problem has an explicit solution.
"""
# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
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
struct DYCOMS_BC{DT} <: BoundaryCondition
  C_drag::DT
  LHF::DT
  SHF::DT
end
function atmos_boundary_state!(::Rusanov, bc::DYCOMS_BC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t, state1::Vars, aux1::Vars)
  # stateM is the 𝐘⁻ state while stateP is the 𝐘⁺ state at an interface. 
  # at the boundaries the ⁻, minus side states are the interior values
  # state1 is 𝐘 at the first interior nodes relative to the bottom wall 
  DT = eltype(stateP)
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
  
  if bctype == 1 # bctype identifies bottom wall 
    stateP.ρu = SVector(0,0,0)
  end
end
function atmos_boundary_state!(::CentralNumericalFluxDiffusive, bc::DYCOMS_BC,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, state1::Vars, diff1::Vars,
                               aux1::Vars)
  # stateM is the 𝐘⁻ state while stateP is the 𝐘⁺ state at an interface. 
  # at the boundaries the ⁻, minus side states are the interior values
  # state1 is 𝐘 at the first interior nodes relative to the bottom wall 
  DT = eltype(stateP)
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
  # Assign diffusive fluxes at boundaries
  diffP = diffM
  xvert = auxM.coord[3]

  if bctype == 1 # bctype identifies bottom wall 
    # ------------------------------------------------------------------------
    # (<var>_FN) First node values (First interior node from bottom wall)
    # ------------------------------------------------------------------------
    z_FN             = aux1.coord[3]
    ρ_FN             = state1.ρ
    U_FN, V_FN, W_FN = state1.ρu
    E_FN             = state1.ρe
    u_FN, v_FN, w_FN = U_FN/ρ_FN, V_FN/ρ_FN, W_FN/ρ_FN
    windspeed_FN     = sqrt(u_FN^2 + v_FN^2 + w_FN^2)
    q_tot_FN         = state1.moisture.ρq_tot / ρ_FN
    e_int_FN         = E_FN/ρ_FN - windspeed_FN^2/2 - grav*z_FN
    TS_FN            = PhaseEquil(e_int_FN, q_tot_FN, ρ_FN) 
    T_FN             = air_temperature(TS_FN)
    q_vap_FN         = q_tot_FN - PhasePartition(TS_FN).liq
    # --------------------------
    # Bottom boundary quantities 
    # --------------------------
    zM          = auxM.coord[3] 
    q_totM      = QTM/ρM
    windspeed   = sqrt(uM^2 + vM^2 + wM^2)
    e_intM      = EM/ρM - windspeed^2/2 - grav*zM
    TSM         = PhaseEquil(e_intM, q_totM, ρM) 
    q_vapM      = q_totM - PhasePartition(TSM).liq
    TM          = air_temperature(TSM)
    # ----------------------------------------------------------
    # Extract components of diffusive momentum flux (minus-side)
    # ----------------------------------------------------------
    ρτM = diffM.ρτ

    # ----------------------------------------------------------
    # Boundary momentum fluxes
    # ----------------------------------------------------------
    # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
    # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is replaced by the appropriate flux terms
    C_drag = bc.C_drag
    ρτ13P  = -ρM * C_drag * windspeed_FN * u_FN 
    ρτ23P  = -ρM * C_drag * windspeed_FN * v_FN 
    # Assign diffusive momentum and moisture fluxes
    # (i.e. ρ𝛕 terms)  
    stateP.ρu = SVector(0,0,0)
    diffP.ρτ = SHermitianCompact{3,DT,6}(SVector(DT(0),ρτM[2,1],ρτ13P, DT(0), ρτ23P,DT(0)))

    # ----------------------------------------------------------
    # Boundary moisture fluxes
    # ----------------------------------------------------------
    diffP.moisture.ρd_q_tot  = SVector(DT(0),
                                       DT(0),
                                       bc.LHF/(LH_v0))
    # ----------------------------------------------------------
    # Boundary energy fluxes
    # ----------------------------------------------------------
    # Assign diffusive enthalpy flux (i.e. ρ(J+D) terms) 
    diffP.ρd_h_tot  = SVector(DT(0),
                              DT(0),
                              bc.LHF + bc.SHF)
  end
end

