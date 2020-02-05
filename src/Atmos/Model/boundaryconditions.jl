using CLIMA.PlanetParameters
export PeriodicBC, NoFluxBC, InitStateBC, DYCOMS_BC, RayleighBenardBC

function atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                        bc,
                                        atmos::AtmosModel,
                                        F,
                                        state⁺, diff⁺, aux⁺, n⁻,
                                        state⁻, diff⁻, aux⁻,
                                        bctype, t, state1⁻, diff1⁻, aux1⁻)
  atmos_boundary_state!(nf, bc, atmos,
                        state⁺, diff⁺, aux⁺, n⁻,
                        state⁻, diff⁻, aux⁻,
                        bctype, t,
                        state1⁻, diff1⁻, aux1⁻)
  flux_diffusive!(atmos, F, state⁺, diff⁺, aux⁺, t)
end

#TODO: figure out a better interface for this.
# at the moment we can just pass a function, but we should do something better
# need to figure out how subcomponents will interact.
function atmos_boundary_state!(::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               f::Function, m::AtmosModel, stateP::Vars,
                               auxP::Vars, nM, stateM::Vars, auxM::Vars, bctype,
                               t, _...)
  f(stateP, auxP, nM, stateM, auxM, bctype, t)
end

function atmos_boundary_state!(::NumericalFluxDiffusive, f::Function,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  f(stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

# lookup boundary condition by face
function atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               bctup::Tuple, m::AtmosModel, stateP::Vars,
                               auxP::Vars, nM, stateM::Vars, auxM::Vars, bctype,
                               t, _...)
  atmos_boundary_state!(nf, bctup[bctype], m, stateP, auxP, nM, stateM, auxM,
                        bctype, t)
end

function atmos_boundary_state!(nf::NumericalFluxDiffusive,
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

function atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               bc::NoFluxBC, m::AtmosModel, stateP::Vars,
                               auxP::Vars, nM, stateM::Vars, auxM::Vars, bctype,
                               t, _...)
  FT = eltype(stateM)
  stateP.ρ = stateM.ρ
  if typeof(nf) <: NumericalFluxNonDiffusive
    stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
  else
    stateP.ρu -=  dot(stateM.ρu, nM) * SVector(nM)
  end
end

function atmos_boundary_state!(::NumericalFluxDiffusive, bc::NoFluxBC,
                               m::AtmosModel, stateP::Vars, diffP::Vars,
                               auxP::Vars, nM, stateM::Vars, diffM::Vars,
                               auxM::Vars, bctype, t, _...)
  FT = eltype(stateM)
  stateP.ρ = stateM.ρ
  stateP.ρu -= dot(stateM.ρu, nM) * SVector(nM)
  
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
function atmos_boundary_state!(::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               bc::InitStateBC, m::AtmosModel, stateP::Vars,
                               auxP::Vars, nM, stateM::Vars, auxM::Vars, bctype,
                               t, _...)
  init_state!(m, stateP, auxP, auxP.coord, t)
end
function atmos_boundary_state!(::NumericalFluxDiffusive, bc::InitStateBC,
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

"""
    atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                          bc::DYCOMS_BC, args...)

For the non-diffussive and gradient terms we just use the `NoFluxBC`
"""
atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                      bc::DYCOMS_BC, args...) = atmos_boundary_state!(nf, NoFluxBC(), args...)

"""
    atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                   bc::DYCOMS_BC, atmos::AtmosModel,
                                   F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)

When `bctype == 1` the `NoFluxBC` otherwise the specialized DYCOMS BC is used
"""
function atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                        bc::DYCOMS_BC, atmos::AtmosModel,
                                        F,
                                        state⁺, diff⁺, aux⁺, n⁻,
                                        state⁻, diff⁻, aux⁻,
                                        bctype, t,
                                        state1⁻, diff1⁻, aux1⁻)
  if bctype != 1
    atmos_boundary_flux_diffusive!(nf, NoFluxBC(), atmos, F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)
  else
    # Start with the noflux BC and then build custom flux from there
    atmos_boundary_state!(nf, NoFluxBC(), atmos,
                          state⁺, diff⁺, aux⁺, n⁻,
                          state⁻, diff⁻, aux⁻,
                          bctype, t)

    # ------------------------------------------------------------------------
    # (<var>_FN) First node values (First interior node from bottom wall)
    # ------------------------------------------------------------------------
    u_FN = state1⁻.ρu / state1⁻.ρ
    windspeed_FN = norm(u_FN)

    # ----------------------------------------------------------
    # Extract components of diffusive momentum flux (minus-side)
    # ----------------------------------------------------------
    _, τ⁻ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)

    # ----------------------------------------------------------
    # Boundary momentum fluxes
    # ----------------------------------------------------------
    # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
    # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is
    # replaced by the appropriate flux terms
    C_drag = bc.C_drag
    @inbounds begin
      τ13⁺ = - C_drag * windspeed_FN * u_FN[1]
      τ23⁺ = - C_drag * windspeed_FN * u_FN[2]
      τ21⁺ = τ⁻[2,1]
    end

    # Assign diffusive momentum and moisture fluxes
    # (i.e. ρ𝛕 terms)
    FT = eltype(state⁺)
    τ⁺ = SHermitianCompact{3, FT, 6}(SVector(0   ,
                                             τ21⁺, τ13⁺,
                                             0   , τ23⁺, 0))

    # ----------------------------------------------------------
    # Boundary moisture fluxes
    # ----------------------------------------------------------
    # really ∇q_tot is being used to store d_q_tot
    d_q_tot⁺  = SVector(0, 0, bc.LHF/(LH_v0))

    # ----------------------------------------------------------
    # Boundary energy fluxes
    # ----------------------------------------------------------
    # Assign diffusive enthalpy flux (i.e. ρ(J+D) terms)
    d_h_tot⁺ = SVector(0, 0, bc.LHF + bc.SHF)

    # Set the flux using the now defined plus-side data
    flux_diffusive!(atmos, F, state⁺, τ⁺, d_h_tot⁺)
    flux_diffusive!(atmos.moisture, F, state⁺, d_q_tot⁺)
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
function atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               bc::RayleighBenardBC, m::AtmosModel,
                               stateP::Vars, auxP::Vars, nM, stateM::Vars,
                               auxM::Vars, bctype, t,_...)
  # Dry Rayleigh Benard Convection
  @inbounds begin
    FT = eltype(stateP)
    stateP.ρ = ρP = stateM.ρ
    if typeof(nf) <: NumericalFluxNonDiffusive
      stateP.ρu = -stateM.ρu
    else
      stateP.ρu = SVector{3,FT}(0,0,0)
    end
    if bctype == 1
      E_intP = ρP * cv_d * (bc.T_bot - T_0)
    else
      E_intP = ρP * cv_d * (bc.T_top - T_0)
    end
    stateP.ρe = (E_intP + ρP * auxP.coord[3] * grav)
    nothing
  end
end
function atmos_boundary_state!(::NumericalFluxDiffusive, bc::RayleighBenardBC,
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
