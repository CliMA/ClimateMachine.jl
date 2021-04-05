#!/usr/bin/env julia --project

include("../boilerplate.jl")
include("ThreeDimensionalCompressibleNavierStokesEquations.jl")

using CLIMAParameters
using CLIMAParameters.Planet: grav, day, cp_d, cv_d, R_d, grav, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()


ClimateMachine.init()

########
# Setup physical and numerical domains
########
Ωˣ = IntervalDomain(0, 27e6, periodic = true)
Ωʸ = IntervalDomain(0, 6e6, periodic = false)
Ωᶻ = IntervalDomain(0, 30e3, periodic = false)

grid = DiscretizedDomain(
    Ωˣ × Ωʸ × Ωᶻ;
    elements = 10,
    polynomial_order = 1,
    overintegration_order = 1,
)

########
# Define timestepping parameters
########
start_time = 0
end_time = 86400
Δt = 0.05
method = SSPRK22Heuns

timestepper = TimeStepper(method = method, timestep = Δt)

callbacks = (Info(), StateCheck(10))

########
# Define physical parameters and parameterizations
########
parameters = (
    ρₒ = 1, # reference density
    cₛ = sqrt(10), # sound speed
    b = 2,
    uₚ = 1,
    Lₚ = 6e5,
    x₀ = 2e6,
    y₀ = 2.5e6,
    γ_lapse = 5e-3,
    β₀ = -0,
    u₀ = 35,
    T₀ = 288,
    gravity = 9.81,
)

"""
    baroclinic_instability_cube(...)
Initialisation helper for baroclinic-wave (channel flow) test case for iterative
determination of η = p/pₛ coordinate for given z-altitude. 
"""
function baroclinic_instability_cube!(eta, 
                                      temp, 
                                      tolerance, 
                                      (x,y,z),
                                      f0, 
                                      beta0,
                                      u0,
                                      T0,
                                      gamma_lapse,
                                      gravity, 
                                      R_gas, 
                                      cp
)
  ## Cap pressure solve at 200 iterations
  for niter = 1:200
    FT    = eltype(y)
    b     = FT(2)
    Ly    = FT(6e6)  
    y0    = FT(Ly/2)
    b2    = b*b
    #Get Mean Temperature
    exp1  = R_gas*gamma_lapse/gravity
    Tmean = T0*eta^exp1
    phimean = T0*gravity/gamma_lapse * (FT(1) - eta^exp1)
    logeta = log(eta)
    fac1   = (f0-beta0*y0)*(y - FT(1/2)*Ly - Ly/2π * sin(2π*y/Ly))  
    fac2   = FT(1/2)*beta0*(y^2 - Ly*y/π*sin(2π*y/Ly) - 
                          FT(1/2)*(Ly/π)^2*cos(2π*y/Ly) - 
                          Ly^2/3 - FT(1/2)*(Ly/π)^2)
    fac3 = exp(-logeta*logeta/b2)
    fac4 = exp(-logeta/b) 
    ## fac4 applies a correction based on the required temp profile from Ullrich's paper
    phi_prime = FT(1/2)*u0*(fac1 + fac2)
    geo_phi = phimean + phi_prime*fac3*logeta
    temp = Tmean + phi_prime/R_gas*fac3*(2/b2*logeta*logeta - 1)
    num  = -gravity*z + geo_phi
    den  = -R_gas/(eta)*temp
    deta = num/den
    eta  = eta - deta
    if (abs(deta) <= FT(tolerance))
      break
    elseif (abs(deta) > FT(tolerance)) && niter==200
      #@error "Initialisation: η convergence failure."
      break
    end
  end
  return (eta, temp)
end 


function cnse_init_state!(model::CNSE3D, state, aux,localgeo,t)

  (x,y,z) = localgeo.coord
  ### Problem float-type
  FT = eltype(state)
  ### Unpack CLIMAParameters
  _planet_radius = FT(6.371e6)
  gravity        = FT(9.81)
  cp             = FT(1000)
  R_gas          = FT(287.1)
  ### Global Variables
  up    = FT(1)                ## See paper: Perturbation peak value
  Lp    = FT(6e5)              ## Perturbation parameter (radius)
  Lp2   = Lp*Lp              
  xc    = FT(2e6)              ## Streamwise center of perturbation
  yc    = FT(2.5e6)            ## Spanwise center of perturbation
  gamma_lapse = FT(5/1000)     ## Γ Lapse Rate
  Ω     = FT(7.292e-5)         ## Rotation rate [rad/s]
  f0    = 2Ω/sqrt(2)           ## 
  beta0 = f0/_planet_radius    ##  
  beta0 = -zero(FT)
  b     = FT(2)
  b2    = b*b
  u0    = FT(35)
  Ly    = FT(6e6)
  T0    = FT(288)
  T_ref = T0                   
  x0    = FT(2e7)
  p00   = FT(1e5)              ## Surface pressure

  ## Step 1: Get current coordinate value by unpacking nodal coordinates from aux state
  eta = FT(1e-7)
  temp = FT(300)
  ## Step 2: Define functions for initial condition temperature and geopotential distributions
  ## These are written in terms of the pressure coordinate η = p/pₛ

  ### Unpack initial conditions (solved by iterating for η)
  tolerance = FT(1e-10)
  eta, temp = baroclinic_instability_cube!(eta, 
                                           temp, 
                                           tolerance, 
                                           (x,y,z),
                                           f0, 
                                           beta0, 
                                           u0, 
                                           T0, 
                                           gamma_lapse, 
                                           gravity, 
                                           R_gas, 
                                           cp)
  eta = min(eta,FT(1))
  eta = max(eta,FT(0))
  ### η = p/p_s
  logeta = log(eta)
  T=FT(temp)
  press = p00*eta
  theta = T *(p00/press)^(R_gas/cp)
  rho = press/(R_gas*T)
  thetaref = T_ref * (1 - gamma_lapse*z/T0)^(-gravity/(cp*gamma_lapse))
  rhoref = p00/(T0*R_gas) * (1 - gamma_lapse*z/T0)^(gravity/(R_gas*gamma_lapse) - 1)

  ### Balanced Flow
  u = -u0*(sinpi(y/Ly))^2  * logeta * exp(-logeta*logeta/b2)

  ### Perturbation of the balanced flow
  rc2 = (x-xc)^2 + (y-yc)^2
  du = up*exp(-rc2/Lp2)
    
  ### Primitive variables
  u⃗ = SVector{3,FT}(u+du,0,0)
  e_kin = FT(1/2)*sum(abs2.(u⃗))
  e_pot = gravity * z

  ### Assign state variables for initial condition
  state.ρ = rho
  state.ρu = rho .* u⃗
  state.ρθ = rho * thetaref
end



physics = FluidPhysics(;
    advection = NonLinearAdvectionTerm(),
    dissipation = ConstantViscosity{Float64}(μ = 0, ν = 1e-2, κ = 1e-2),
    coriolis = nothing,
    buoyancy = nothing,
)

########
# Define boundary conditions
########
ρu_bcs = (
    bottom = Impenetrable(NoSlip()),
    top = Impenetrable(FreeSlip()),
)
ρθ_bcs =
    (north = Insulating(), south = Insulating(),
     bottom = Insulating(), top = Insulating())

BC = (ρθ = ρθ_bcs, ρu = ρu_bcs)

########
# Create the things
########

model = SpatialModel(
    balance_law = Fluid3D(),
    physics = physics,
    numerics = (flux = RoeNumericalFlux(),),
    grid = grid,
    boundary_conditions = BC,
    parameters = parameters,
)

simulation = Simulation(
    model = model,
    timestepper = timestepper,
    callbacks = callbacks,
    time = (; start = start_time, finish = end_time),
)

########
# Run the model
########

tic = Base.time()

evolve!(simulation, model)

toc = Base.time()
time = toc - tic
println(time)
