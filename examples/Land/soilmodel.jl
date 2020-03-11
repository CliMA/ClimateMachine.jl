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

# --------------------------------- CLIMA SOIL MODEL -----------------------
# soilmodel.jl: This model simulates soil dynamics for the CliMA model


# --------------------------------- 1) Import/Export Needed Functions -----------------------

# Add necessary CliMA functions and sub-routines
using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!, update_aux!, nodal_update_aux!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry


# --------------------------------- 2) Define Structs ---------------------------------------

# Introduce needed variables into SoilModel struct
Base.@kwdef struct SoilModel{Fρc, Fκ, FiT, Fst} <: BalanceLaw
  
  # Define heat capacity. This is an input to the model now.
  ρc::Fρc       = (state, aux, t) -> 2.49e6   # [ Sand: ρc = 2.49e6 J m-3 K-1 ; Clay: ρc = 2.61e6 J m-3 K-1 ]
  # Replace this with a function that calculates heat capacity (based on liquid+ice)    
  # OR Replace this with tabulated values of heat capacity (based on liquid+ice)
 
  # Define kappa (thermal conductivity). This is an input to the model now.
  κ::Fκ         = (state, aux, t) -> 2.42     # [ Sand: λ = 2.42 W m-1 K-1 ; Clay: λ = 1.17 W m-1 K-1 ]
    
  # Define initial and boundary condition parameters 
  initialT::FiT = (aux, t) -> 273.15 + 2.0 # Initial Temperature. This is an input to the model now.
  surfaceT::Fst = (state, aux, t) -> (273.15 + 2.0) # Surface boundary condition. This is an input to the model now.
end


# --------------------------------- 3) Define CliMA vars ---------------------------------------

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_aux(::SoilModel, FT) = @vars(z::FT, T::FT) # stored dg.auxstate
vars_state(::SoilModel, FT) = @vars(ρcT::FT) # stored in Q
vars_gradient(::SoilModel, FT) = @vars(T::FT) # not stored
vars_diffusive(::SoilModel, FT) = @vars(∇T::SVector{3,FT}) # stored in dg.diffstate


# --------------------------------- 4) CliMA functions needed for simulation -------------------

# ---------------- 4a) Update states

# Update all auxiliary variables
function update_aux!(dg::DGModel, m::SoilModel, Q::MPIStateArray, t::Real)
  nodal_update_aux!(soil_nodal_update_aux!, dg, m, Q, t)
  return true
end
# Update all auxiliary nodes
function soil_nodal_update_aux!(m::SoilModel, state::Vars, aux::Vars, t::Real)        
  aux.T = state.ρcT / m.ρc(state, aux, t) # TODO: figure out why can't use aux.T here
end

# ---------------- 4b) Calculate state and derivative of T

# Calculate T based on internal energy state variable
function gradvariables!(m::SoilModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.T = state.ρcT / m.ρc(state, aux, t)
end
# Gradient of T calculation
function diffusive!(m::SoilModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.∇T = ∇transform.T
end
# Calculate thermal flux (non-diffusive (?))
function flux_nondiffusive!(m::SoilModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
# Calculate thermal flux (diffusive (?))
function flux_diffusive!(m::SoilModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)     
   flux.ρcT -= m.κ(state, aux, t) * diffusive.∇T
   if aux.z == 0
    #@show   aux.T flux.ρcT
    end
end

# ---------------- 4c) Extra Sources

# Introduce sources of energy (e.g. Metabolic heat from microbes) 
function source!(m::SoilModel, state::Vars, _...)
end


# ---------------- 4d) Initialization

# Initialize z-Profile
function init_aux!(m::SoilModel, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[3]
  aux.T = m.initialT(aux, t)
end
# Initialize State variables from T to internal energy
function init_state!(m::SoilModel, state::Vars, aux::Vars, coords, t::Real)
  state.ρcT = m.ρc(state, aux, t) * aux.T
end

# ---------------- 4e) Boundary Conditions

# Boundary condition function
function boundary_state!(nf, m::SoilModel, state⁺::Vars, aux⁺::Vars,
                         nM, state⁻::Vars, aux⁻::Vars, bctype, t, _...)
  if bctype == 1
    # surface
    state⁺.ρcT = m.ρc(state⁻, aux⁻, t) * m.surfaceT(state⁻, aux⁻, t)
  elseif bctype == 2
    # bottom
    nothing
  end
end
# Boundary condition function - repeated?
function boundary_state!(nf, m::SoilModel, state⁺::Vars, diff⁺::Vars,
                         aux⁺::Vars, nM, state⁻::Vars, diff⁻::Vars, aux⁻::Vars,
                         bctype, t, _...)
  if bctype == 1
    # surface
    state⁺.ρcT = m.ρc(state⁻, aux⁻, t) * m.surfaceT(state⁻, aux⁻, t)
  elseif bctype == 2
    # bottom
    diff⁺.∇T = -diff⁻.∇T
  end
end

