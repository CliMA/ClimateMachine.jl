# Add necessary CliMA functions and sub-routines
using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!, update_aux!, nodal_update_aux!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry

export lamba_calculator

"""
SoilModelMoisture

Computes diffusive flux `F` in:

∂y / ∂t = ∇ ⋅ Flux + Source

```
 ∂(ρ)    ∂      ∂h
------ = --(k * --)
  ∂t     ∂z     ∂z
```
where
 - `ρ` is the volumetric water content of soil (m³/m³), this is state var.
 - `k` is the hydraulic conductivity (m/s)
 - `h` is the hydraulic head or water potential (m), it is a function of ρ 
 - `z` is the depth (m)
 - `p` is matric potential (m), p=-(ρ/v)^(-M) with Brooks and Corey Formulation
 - `v` is porosity, typical value :
 - `M` is paramter, typical value for sand: 1/0.378, from  BRAUN 2004, p. 1118

To write this in the form
```
∂Y
-- + ∇⋅F(Y,t) = 0
∂t
```
we write `Y = ρ` and `F(Y, t) =-k ∇h`.

"""

# Introduce needed variables into SoilModel struct
Base.@kwdef struct SoilModelMoisture{Fκ, Fiρ, Fsρ} <: BalanceLaw 
  # Define kappa (hydraulic conductivity)
  κ::Fκ         = (state, aux, t) -> (0.001/(60*60*24)) # [m/s] typical value taken from Land Surface Model CLiMA, table 2.2, =0.1cm/day  
  # Define initial and boundary condition parameters
  initialρ::Fiρ = (state, aux, t) -> 0.1 # [m3/m3] constant water content in soil, from Bonan, Ch.8, fig 8.8 as in Haverkamp et al. 1977, p.287
  surfaceρ::Fsρ = (state, aux, t) -> 0.6 #267 # [m3/m3] constant flux at surface, from Bonan, Ch.8, fig 8.8 as in Haverkamp et al. 1977, p.287
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_aux(::SoilModelMoisture, Fρ) = @vars(z::Fρ, h::Fρ) # p::Fρ stored in dg.auxstate
vars_state(::SoilModelMoisture, Fρ) = @vars(ρ::Fρ) # stored in Q
vars_gradient(::SoilModelMoisture, Fρ) = @vars(h::Fρ) # not stored
vars_diffusive(::SoilModelMoisture, Fρ) = @vars(∇h::SVector{3,Fρ}) # stored in dg.diffstate

# Update all auxiliary variables
function update_aux!(dg::DGModel, m::SoilModelMoisture, Q::MPIStateArray, t::Real)
  nodal_update_aux!(soil_nodal_update_aux!, dg, m, Q, t)
  return true
end
# Update all auxiliary nodes
function soil_nodal_update_aux!(m::SoilModelMoisture, state::Vars, aux::Vars, t::Real)     
  aux.h = aux.z+state.ρ #^(-1/0.378))*(-0.3020)
end

# Calculate h based on state variable
function gradvariables!(m::SoilModelMoisture, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.h = aux.z+state.ρ #^(-1/0.378))*(-0.3020)
end

# Gradient of h calculation
function diffusive!(m::SoilModelMoisture, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.∇h = ∇transform.h
end

# Calculate thermal flux (non-diffusive)
function flux_nondiffusive!(m::SoilModelMoisture, flux::Grad, state::Vars, aux::Vars, t::Real)
end

# Calculate water flux (diffusive)
function flux_diffusive!(m::SoilModelMoisture, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)     
   flux.ρ -= m.κ(state, aux, t) * diffusive.∇h
   if aux.z == 0
    #@show   aux.T flux.ρcT
    end
end

# Introduce sources of energy (e.g. Metabolic heat from microbes) 
function source!(m::SoilModelMoisture, state::Vars, _...)
end

# Initialize z-Profile
function init_aux!(m::SoilModelMoisture, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[3]
  aux.h = 0.5 #aux.z+m.initialρ(state, aux, t) #^(-1/0.378))*(-0.3020)
end

# Initialize State variables from T to internal energy
function init_state!(m::SoilModelMoisture, state::Vars, aux::Vars, coords, t::Real)
  state.ρ = m.initialρ(state, aux, t)
end

# Boundary condition function
function boundary_state!(nf, m::SoilModelMoisture, state⁺::Vars, aux⁺::Vars,
                         nM, state⁻::Vars, aux⁻::Vars, bctype, t, _...)
  if bctype == 1
    # surface
    state⁺.ρ= m.surfaceρ(state⁻, aux⁻, t)
  elseif bctype == 2
    # bottom
    nothing
  end
end

# Boundary condition function 
function boundary_state!(nf, m::SoilModelMoisture, state⁺::Vars, diff⁺::Vars,
                         aux⁺::Vars, nM, state⁻::Vars, diff⁻::Vars, aux⁻::Vars,
                         bctype, t, _...)
  if bctype == 1
    # surface
    state⁺.ρ = m.surfaceρ(state⁻, aux⁻, t)
  elseif bctype == 2
    # bottom
    diff⁺.∇h = -diff⁻.∇h
  end
end
