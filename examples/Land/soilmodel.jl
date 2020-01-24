using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry
import CLIMA.DGmethods.NumericalFluxes: NumericalFluxNonDiffusive,
                                       NumericalFluxDiffusive,
                                       GradNumericalPenalty,
                                       numerical_boundary_flux_diffusive!

"""
    SoilModel

Computes diffusive flux `F` in:

∂y / ∂t = ∇ ⋅ Flux + Source

```
∂(ρcT)   ∂      ∂T
------ = --(λ * --)
  ∂t     ∂z     ∂z
```
where

 - `ρ` is the density of the soil (kg/m³)
 - `c` is the soil heat capacity (J/(kg K))
 - `λ` is the thermal conductivity (W/(m K))

To write this in the form
```
∂Y
-- + ∇⋅F(Y,t) = 0
∂t
```
we write `Y = ρcT` and `F(Y, t) = -λ ∇T`.

"""
struct SoilModel <: BalanceLaw
  ρ::Float64
  c::Float64
  λ::Float64
  surfaceT::Float64
  initialT::Float64
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_aux(::SoilModel, FT) = @vars(z::FT, ρ::FT, c::FT, λ::FT) # stored dg.auxstate
vars_state(::SoilModel, FT) = @vars(ρcT::FT) # stored in Q
vars_gradient(::SoilModel, FT) = @vars(T::FT) # not stored
vars_diffusive(::SoilModel, FT) = @vars(∇T::SVector{3,FT}) # stored in dg.diffstate

function gradvariables!(m::SoilModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.T = state.ρcT / (m.ρ * m.c)
end
function diffusive!(m::SoilModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.∇T = ∇transform.T
end
function flux_nondiffusive!(m::SoilModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
function flux_diffusive!(m::SoilModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  flux.ρcT -= m.λ * diffusive.∇T
end

function source!(m::SoilModel, state::Vars, _...)
  # state.ρcT += d(ρcT)/dt
end

#=
function wavespeed(m::SoilModel, nM, state::Vars, aux::Vars, t::Real)
  zero(eltype(state))
end
=#

function init_aux!(m::SoilModel, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[3]
end

function init_state!(m::SoilModel, state::Vars, aux::Vars, coords, t::Real)
  state.ρcT = 10.0 #+ sin(aux.z)
end

# Neumann boundary conditions

function numerical_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
    bl::SoilModel, fluxᵀn::Vars{S}, n::SVector,
    state⁻::Vars{S}, diff⁻::Vars{D}, aux⁻::Vars{A},
    state⁺::Vars{S}, diff⁺::Vars{D}, aux⁺::Vars{A},
    bctype, t,
    state1⁻::Vars{S}, diff1⁻::Vars{D}, aux1⁻::Vars{A}) where {S,D,A}

  if bctype == 1 # top
    fluxᵀn.ρcT = 2 #state⁻.ρcT/(60*60*24)*sinpi(2t/(60*60*24))
  else # bottom
    fluxᵀn.ρcT = 0
  end
end

#=
# set up domain
topl = StackedBrickTopology(mpicomm, (0:1,0:1,0:-1:-10); periodicity = (true,true,false),boundary=((0,0),(0,0),(1,2)))
grid = DiscontinuousSpectralElementGrid(topl, FloatType = Float64, DeviceArray = Array, polynomialorder = 5)

m = SoilModel(1.0,1.0,1.0,20.0,10.0)

# Set up DG scheme
dg = DGModel( #
  m, # "PDE part"
  grid,
  CentralNumericalFluxNonDiffusive(), # penalty terms for discretizations
  CentralNumericalFluxDiffusive(),
  CentralGradPenalty())


Δ = min_node_distance(grid)
CFL_bound = (Δ^2 / (2m.λ/(m.ρ*m.c)))
dt = CFL_bound*0.001 # TODO: provide a "default" timestep based on  Δx,Δy,Δz

# state variable
Q = init_ode_state(dg, Float64(0))

# initialize ODE solver
lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

function plotstate(grid, Q)
    # TODO:
    # this currently uses some internals: provide a better way to do this
    Xg = reshape(grid.vgeo[(1:6^2:6^3),CLIMA.Mesh.Grids.vgeoid.x3id,:],:)
    Yg = reshape(Q.data[(1:6^2:6^3),1,:],:)
    plot(Xg, Yg, xlabel="depth", ylabel="ρcT", ylimit=(0,40))
end

#plotstate(grid, Q)
=#

function boundary_state!(nf, m::SoilModel, stateP::Vars, auxP::Vars,
                         nM, stateM::Vars, auxM::Vars, bctype, t, _...)
  nothing
end
#=
function boundary_state!(nf, m::SoilModel, stateP::Vars, diffP::Vars,
                         auxP::Vars, nM, stateM::Vars, diffM::Vars, auxM::Vars,
                         bctype, t, _...)
  if bctype == 1
    # top
    #stateP.ρcT = 15
    diffP.∇T = SVector(0,0,1.0)
  elseif bctype == 2
    #diffP.∂T∂z = -diffM.∂T∂z
  end
end
=#
