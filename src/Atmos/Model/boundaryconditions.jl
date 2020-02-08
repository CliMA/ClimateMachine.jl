using CLIMA.PlanetParameters
export PeriodicBC, NoFluxBC, InitStateBC, DYCOMS_BC

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
                               f::Function, m::AtmosModel, state⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, aux⁻::Vars, bctype,
                               t, _...)
  f(state⁺, aux⁺, n⁻, state⁻, aux⁻, bctype, t)
end

function atmos_boundary_state!(::NumericalFluxDiffusive, f::Function,
                               m::AtmosModel, state⁺::Vars, diff⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, diff⁻::Vars,
                               aux⁻::Vars, bctype, t, _...)
  f(state⁺, diff⁺, aux⁺, n⁻, state⁻, diff⁻, aux⁻, bctype, t)
end

# lookup boundary condition by face
function atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               bctup::Tuple, m::AtmosModel, state⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, aux⁻::Vars, bctype,
                               t, _...)
  atmos_boundary_state!(nf, bctup[bctype], m, state⁺, aux⁺, n⁻, state⁻, aux⁻,
                        bctype, t)
end

function atmos_boundary_state!(nf::NumericalFluxDiffusive,
                               bctup::Tuple, m::AtmosModel, state⁺::Vars,
                               diff⁺::Vars, aux⁺::Vars, n⁻, state⁻::Vars,
                               diff⁻::Vars, aux⁻::Vars, bctype, t, _...)
  atmos_boundary_state!(nf, bctup[bctype], m, state⁺, diff⁺, aux⁺, n⁻, state⁻,
                        diff⁻, aux⁻, bctype, t)
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
                               bc::NoFluxBC, m::AtmosModel, state⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, aux⁻::Vars, bctype,
                               t, _...)
  FT = eltype(state⁻)
  state⁺.ρ = state⁻.ρ
  if typeof(nf) <: NumericalFluxNonDiffusive
    state⁺.ρu -= 2 * dot(state⁻.ρu, n⁻) * SVector(n⁻)
  else
    state⁺.ρu -=  dot(state⁻.ρu, n⁻) * SVector(n⁻)
  end
end

function atmos_boundary_state!(::NumericalFluxDiffusive, bc::NoFluxBC,
                               m::AtmosModel, state⁺::Vars, diff⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, diff⁻::Vars,
                               aux⁻::Vars, bctype, t, _...)
  FT = eltype(state⁻)
  state⁺.ρ = state⁻.ρ
  state⁺.ρu -= dot(state⁻.ρu, n⁻) * SVector(n⁻)
  
  fill!(getfield(diff⁺, :array), FT(0))
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
                               bc::InitStateBC, m::AtmosModel, state⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, aux⁻::Vars, bctype,
                               t, _...)
  init_state!(m, state⁺, aux⁺, aux⁺.coord, t)
end
function atmos_boundary_state!(::NumericalFluxDiffusive, bc::InitStateBC,
                               m::AtmosModel, state⁺::Vars, diff⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, diff⁻::Vars,
                               aux⁻::Vars, bctype, t, _...)
  init_state!(m, state⁺, aux⁺, aux⁺.coord, t)
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
