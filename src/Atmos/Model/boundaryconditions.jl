using CLIMA.PlanetParameters
export BoundaryCondition, PeriodicBC, NoFluxBC, InitStateBC

function atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                        bc,
                                        atmos::AtmosModel,
                                        F,
                                        state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                                        state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                        bctype, t, state1⁻, diff1⁻, aux1⁻)
  atmos_boundary_state!(nf, bc, atmos,
                        state⁺, diff⁺, aux⁺, n⁻,
                        state⁻, diff⁻, aux⁻,
                        bctype, t,
                        state1⁻, diff1⁻, aux1⁻)
  flux_diffusive!(atmos, F, state⁺, diff⁺, hyperdiff⁺, aux⁺, t)
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
