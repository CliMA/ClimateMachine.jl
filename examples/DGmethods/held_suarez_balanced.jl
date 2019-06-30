# # Held-Suarez on the Sphere
#
#md # !!! jupyter
#md #     This example is also available as a Jupyter notebook:
#md #     [`held_suarez_balanced.ipynb`](@__NBVIEWER_ROOT_URL__examples/DGmethods/generated/held_suarez_balanced.html)
#
# ## Introduction
#
# In this example we will set up and run the Held-Suarez test case from 
# Held and Suarez (1994) (https://journals.ametsoc.org/doi/pdf/10.1175/1520-0477%281994%29075%3C1825%3AAPFTIO%3E2.0.CO%3B2)

# Below is a program interspersed with comments.
#md # The full program, without comments, can be found in the next
#md # [section](@ref held_suarez_balanced-plain-program).
#
# ## Commented Program

#------------------------------------------------------------------------------
#--------------------------------#
#--------------------------------#
# Can be run with:
# CPU: mpirun -n 1 julia --project=@. held_suarez_balanced.jl
# GPU: mpirun -n 1 julia --project=/home/fxgiraldo/CLIMA/env/gpu held_suarez_balanced.jl
#--------------------------------#
#--------------------------------#

# ### Preliminaries
# Load in modules needed for solving the problem
using MPI
using Logging
using LinearAlgebra
using Dates
using Printf
using CLIMA
using CLIMA.Topologies
using CLIMA.MPIStateArrays
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.Vtk
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks

# Though not required, here we are explicit about which values we read out the
# `PlanetParameters` and `MoistThermodynamics`
using CLIMA.PlanetParameters: planet_radius, R_d, cp_d, grav, cv_d, MSLP
using CLIMA.MoistThermodynamics: air_temperature, air_pressure, internal_energy,
                                 soundspeed_air, air_density, gas_constant_air

# Start up MPI if this has not already been done
MPI.Initialized() || MPI.Init()
#md nothing # hide

# If `CuArrays` is in the current environment we will use CUDA, otherwise we
# drop back to the CPU
@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const DeviceArrayType = CuArray
else
  const DeviceArrayType = Array
end
#md nothing # hide

# Specify whether to enforce hydrostatic balance at PDE level or not
const PDE_level_hydrostatic_balance = true

# check whether to use default VTK directory or define something else
VTKDIR = get(ENV, "CLIMA_VTK_DIR", "vtk")

# Here we setup constants to for some of the computational parameters; the
# underscore is just syntactic sugar to indicate that these are constants.

# These are parameters related to the Euler state. Here we used the conserved
# variables for the state: perturbation in density, three components of
# momentum, and perturbation to the total energy.
const _nstate = 5
const _dρ, _ρu, _ρv, _ρw, _dρe = 1:_nstate
const _statenames = ("δρ", "ρu", "ρv", "ρw", "δρe")
#md nothing # hide

# These will be the auxiliary state which will contain the geopotential,
# gradient of the geopotential, and reference values for density and total
# energy
const _nauxstate = 6
const _a_ϕ, _a_ϕx, _a_ϕy, _a_ϕz, _a_ρ_ref, _a_ρe_ref = 1:_nauxstate
const _auxnames = ("ϕ", "ϕx", "ϕy", "ϕz", "ρ_ref", "ρe_ref")
#md nothing # hide

#------------------------------------------------------------------------------

# ### Definition of the physics
#md # Now we define a function which given the state and auxiliary state defines
#md # the physical Euler flux
function eulerflux!(F, Q, _, aux, t)
  @inbounds begin
    ## extract the states
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ

    ## compute the velocity
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the pressure
    T = air_temperature(e_int)
    P = air_pressure(T, ρ)

    e_ref_int = ρe_ref / ρ_ref - ϕ
    T_ref = air_temperature(e_ref_int)
    P_ref = air_pressure(T_ref, ρ_ref)

    ## set the actual flux
    F[1, _dρ ], F[2, _dρ ], F[3, _dρ ] = ρu          , ρv          , ρw
    if PDE_level_hydrostatic_balance
      δP = P - P_ref
      F[1, _ρu], F[2, _ρu], F[3, _ρu] = u * ρu  + δP, v * ρu     , w * ρu
      F[1, _ρv], F[2, _ρv], F[3, _ρv] = u * ρv      , v * ρv + δP, w * ρv
      F[1, _ρw], F[2, _ρw], F[3, _ρw] = u * ρw      , v * ρw     , w * ρw + δP
    else
      F[1, _ρu], F[2, _ρu], F[3, _ρu] = u * ρu  + P, v * ρu    , w * ρu
      F[1, _ρv], F[2, _ρv], F[3, _ρv] = u * ρv     , v * ρv + P, w * ρv
      F[1, _ρw], F[2, _ρw], F[3, _ρw] = u * ρw     , v * ρw    , w * ρw + P
    end
    F[1, _dρe], F[2, _dρe], F[3, _dρe] = u * (ρe + P), v * (ρe + P), w * (ρe + P)
  end
end
#md nothing # hide

# Define the geopotential source from the solution and auxiliary variables
function geopotential!(S, Q, aux, t)
  @inbounds begin
    ρ_ref, ϕx, ϕy, ϕz = aux[_a_ρ_ref], aux[_a_ϕx], aux[_a_ϕy], aux[_a_ϕz]
    dρ = Q[_dρ]
    S[_dρ ] = 0
    if PDE_level_hydrostatic_balance
      S[_ρu ] = -dρ * ϕx
      S[_ρv ] = -dρ * ϕy
      S[_ρw ] = -dρ * ϕz
    else
      ρ = ρ_ref + dρ
      S[_ρu ] = -ρ * ϕx
      S[_ρv ] = -ρ * ϕy
      S[_ρw ] = -ρ * ϕz
    end
    S[_dρe] = 0
  end
end
#md nothing # hide

# This defines the local wave speed from the current state (this will be needed
# to define the numerical flux)
function wavespeed(n, Q, aux, _...)
  @inbounds begin
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]

    ## get total energy and density
    ρ = ρ_ref + dρ
    e = (ρe_ref + dρe) / ρ

    ## velocity field
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the temperature
    T = air_temperature(e_int)

    abs(n[1] * u + n[2] * v + n[3] * w) + soundspeed_air(T)
  end
end
#md nothing # hide

# The only boundary condition needed for this test problem is the no flux
# boundary condition, the state for which is defined below. This function
# defines the plus-side (exterior) values from the minus-side (inside) values.
# This plus-side value will then be fed into the numerical flux routine in order
# to enforce the boundary condition.
function nofluxbc!(QP, _, _, nM, QM, _, auxM, _...)
  @inbounds begin
    DFloat = eltype(QM)
    ## get the minus values
    dρM, ρuM, ρvM, ρwM, dρeM = QM[_dρ], QM[_ρu], QM[_ρv], QM[_ρw], QM[_dρe]

    ## scalars are preserved
    dρP, dρeP = dρM, dρeM

    ## vectors are reflected
    nx, ny, nz = nM[1], nM[2], nM[3]

    ## reflect velocities
    mag_ρu⃗ = nx * ρuM + ny * ρvM + nz * ρwM
    ρuP = ρuM - 2mag_ρu⃗ * nx
    ρvP = ρvM - 2mag_ρu⃗ * ny
    ρwP = ρwM - 2mag_ρu⃗ * nz

    ## Construct QP state
    QP[_dρ], QP[_ρu], QP[_ρv], QP[_ρw], QP[_dρe] = dρP, ρuP, ρvP, ρwP, dρeP
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Definition of the problem
# Here we define the initial condition as well as the auxiliary state (which
# contains the reference state on which the initial condition depends)

# First it is useful to have a conversion function going between Cartesian and
# spherical coordinates (defined here in terms of radians)
function cartesian_to_spherical(DFloat, x, y, z)
    r = hypot(x, y, z)
    λ = atan(y, x)
    φ = asin(z / r)
    (r, λ, φ)
end
#md nothing # hide

# FXG: reference state
# Setup the reference state based on a N=0.0158725 uniformly stratified atmosphere with θ0=315K
function auxiliary_state_initialization!(T0, aux, x, y, z)
  @inbounds begin
    DFloat = eltype(aux)
    p0 = DFloat(MSLP)
    θ0 = DFloat(315)
    N_bv=DFloat(0.0158725)
    gravity=DFloat(grav)
    
    ## Convert to Spherical coordinates
    (r, λ, φ) = cartesian_to_spherical(DFloat, x, y, z)

    ## Calculate the geopotential ϕ
    h = r - DFloat(planet_radius) # height above the planet surface
    ϕ = gravity * h

    ## Reference Potential Temperature from constant Brunt-Vaisala definition
    θ_ref = θ0 * exp( N_bv^2*h/gravity )

    ## Reference Exner Pressure from hydrostatic equation
    π_ref = 1 + gravity^2/(cp_d*θ0*N_bv^2)*(exp(-N_bv^2*h/gravity) - 1)

    ## Calculate pressure from exner pressure definition
    P_ref = p0*(π_ref)^(cp_d/R_d)

    ## Calculate temperature
    T_ref=θ_ref*π_ref

    ## Density from the ideal gas law
    ρ_ref = air_density(T_ref, P_ref)

    ## Calculate the reference total energy
    e_int = internal_energy(T_ref)
    ρe_ref = e_int * ρ_ref + ρ_ref * ϕ

    ## Fill the auxiliary state array
    aux[_a_ϕ] = ϕ
    ## gradient of the geopotential will be computed numerically below
    aux[_a_ϕx] = 0
    aux[_a_ϕy] = 0
    aux[_a_ϕz] = 0
    aux[_a_ρ_ref]  = ρ_ref
    aux[_a_ρe_ref] = ρe_ref
  end
end
#md nothing # hide

# FXG: initial conditions
# The initial condition is the reference state defined previously. Here we define a zero perturbation
function initialcondition!(domain_height, Q, x, y, z, aux, _...)
  @inbounds begin
    DFloat = eltype(Q)
    p0 = DFloat(MSLP)

    (r, λ, φ) = cartesian_to_spherical(DFloat, x, y, z)
    h = r - DFloat(planet_radius)

    ## Get the reference pressure from the previously defined reference state
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ## Equate total field as reference field
    ρ, ρe = ρ_ref, ρe_ref

    ## Define perturbations from reference field
    dρ = ρ - ρ_ref
    dρe = ρe - ρe_ref
    
    ## Store Initial conditions
    Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe] =  dρ, 0, 0, 0, dρe
  end
end
#md nothing # hide


# This function compute the pressure perturbation for a given state. It will be
# used only in the computation of the pressure perturbation prior to writing the
# VTK output.
function compute_δP!(δP, Q, _, aux)
  @inbounds begin
    ## extract the states
    dρ, ρu, ρv, ρw, dρe = Q[_dρ], Q[_ρu], Q[_ρv], Q[_ρw], Q[_dρe]
    ρ_ref, ρe_ref, ϕ = aux[_a_ρ_ref], aux[_a_ρe_ref], aux[_a_ϕ]

    ## Compute the reference pressure
    e_ref_int = ρe_ref / ρ_ref - ϕ
    T_ref = air_temperature(e_ref_int)
    P_ref = air_pressure(T_ref, ρ_ref)

    ## Compute the fulle states
    ρ = ρ_ref + dρ
    ρe = ρe_ref + dρe
    e = ρe / ρ

    ## compute the velocity
    u, v, w = ρu / ρ, ρv / ρ, ρw / ρ

    ## internal energy
    e_int = e - (u^2 + v^2 + w^2)/2 - ϕ

    ## compute the pressure
    T = air_temperature(e_int)
    P = air_pressure(T, ρ)

    ## store the pressure perturbation
    δP[1] = P - P_ref
  end
end
#md nothing # hide

#------------------------------------------------------------------------------

# ### Initialize the DG Method
function setupDG(mpicomm, Ne_vertical, Ne_horizontal, polynomialorder,
                 ArrayType, domain_height, T0, DFloat)

  ## Create the element grid in the vertical direction
  Rrange = range(DFloat(planet_radius), length = Ne_vertical + 1,
                 stop = planet_radius + domain_height)

  ## Set up the mesh topology for the sphere
  topology = StackedCubedSphereTopology(mpicomm, Ne_horizontal, Rrange)

  ## Set up the grid for the sphere. Note that here we need to pass the
  ## `cubedshellwarp` shell `meshwarp` function so that the degrees of freedom
  ## lay on the sphere (and not just stacked cubes)
  grid = DiscontinuousSpectralElementGrid(topology;
                                          polynomialorder = polynomialorder,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          meshwarp = Topologies.cubedshellwarp)

  ## Here we use the Rusanov numerical flux which requires the physical flux and
  ## wavespeed
  numflux!(x...) = NumericalFluxes.rusanov!(x..., eulerflux!, wavespeed)

  ## We also use Rusanov to define the numerical boundary flux which also
  ## requires a definition of the state to use for the "plus" side of the
  ## boundary face (calculated here with `nofluxbc!`)
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x..., eulerflux!,
                                                            nofluxbc!,
                                                            wavespeed)

  auxinit(x...) = auxiliary_state_initialization!(T0, x...)
  ## Define the balance law solver
  spatialdiscretization = DGBalanceLaw(grid = grid,
                                       length_state_vector = _nstate,
                                       flux! = eulerflux!,
                                       source! = geopotential!,
                                       numerical_flux! = numflux!,
                                       numerical_boundary_flux! = numbcflux!,
                                       auxiliary_state_length = _nauxstate,
                                       auxiliary_state_initialization! =
                                       auxinit,
                                      )

  ## Compute Gradient of Geopotential
  DGBalanceLawDiscretizations.grad_auxiliary_state!(spatialdiscretization, _a_ϕ,
                                                    (_a_ϕx, _a_ϕy, _a_ϕz))

  spatialdiscretization
end
#md nothing # hide

# ### Initializing and run the DG method
# Note that the final time and grid size are small so that CI and docs
# generation happens in a reasonable amount of time. Running the simulation to a
# final time of `33` hours allows the wave to propagate all the way around the
# sphere and back. Increasing the numeber of horizontal elements to `~30` is
# required for stable long time simulation.
let
  mpicomm = MPI.COMM_WORLD
  mpi_logger = ConsoleLogger(MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull)

  ## parameters for defining the cubed sphere.
  Ne_vertical   = 6  # number of vertical elements (small for CI/docs reasons)
  ## Ne_vertical   = 30 # Resolution required for stable long time result
  ## cubed sphere will use Ne_horizontal * Ne_horizontal horizontal elements in
  ## each of the 6 faces
  Ne_horizontal = 3

  polynomialorder = 5

  ## top of the domain and temperature from Tomita and Satoh (2004)
  domain_height = 30e3

  ## isothermal temperature state
  T0 = 315

  ## Floating point type to use in the calculation
  DFloat = Float64

  spatialdiscretization = setupDG(mpicomm, Ne_vertical, Ne_horizontal,
                                  polynomialorder, DeviceArrayType,
                                  domain_height, T0, DFloat)

  Q = MPIStateArray(spatialdiscretization,
                    (x...) -> initialcondition!(domain_height, x...))

  ## Since we are using explicit time stepping the acoustic wave speed will
  ## dominate our CFL restriction along with the vertical element size
  element_size = (domain_height / Ne_vertical)
  acoustic_speed = soundspeed_air(DFloat(T0))
  dt = element_size / acoustic_speed / polynomialorder^2

  ## Adjust the time step so we exactly hit 1 hour for VTK output
  #dt = 60 * 60 / ceil(60 * 60 / dt)
  dt=1
  
  lsrk = LSRK54CarpenterKennedy(spatialdiscretization, Q; dt = dt, t0 = 0)

  filter = Grids.CutoffFilter(spatialdiscretization.grid)
#  filter = Grids.ExponentialFilter(spatialdiscretization.grid)

  ## Uncomment line below to extend simulation time and output less frequently
  seconds = 1
  minutes = 60
  hours = 3600
  days = 86400
#  finaltime = 0.1*days
#  outputtime = 0.01*days
  finaltime = 1000
  outputtime =100
  
  @show(polynomialorder,Ne_horizontal,Ne_vertical,dt,finaltime,finaltime/dt)

  ## We will use this array for storing the pressure to write out to VTK
  δP = MPIStateArray(spatialdiscretization; nstate = 1)

  ## Define a convenience function for VTK output
  mkpath(VTKDIR)
  function do_output(vtk_step)
    ## name of the file that this MPI rank will write
    filename = @sprintf("%s/held_suarez_balanced_mpirank%04d_step%04d",
                        VTKDIR, MPI.Comm_rank(mpicomm), vtk_step)

    ## fill the `δP` array with the pressure perturbation
    DGBalanceLawDiscretizations.dof_iteration!(compute_δP!, δP,
                                               spatialdiscretization, Q)

    ## write the vtk file for this MPI rank
    writevtk(filename, Q, spatialdiscretization, _statenames, δP, ("δP",))

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
      ## name of the pvtu file
      pvtuprefix = @sprintf("held_suarez_balanced_step%04d", vtk_step)

      ## name of each of the ranks vtk files
      prefixes = ntuple(i->
                        @sprintf("%s/held_suarez_balanced_mpirank%04d_step%04d",
                                 VTKDIR, i-1, vtk_step),
                        MPI.Comm_size(mpicomm))

      ## Write out the pvtu file
      writepvtu(pvtuprefix, prefixes, (_statenames..., "δP",))

      ## write that we have written the file
      with_logger(mpi_logger) do
        @info @sprintf("Done writing VTK: %s", pvtuprefix)
      end
    end
  end

  ## Setup callback for writing VTK every hour of simulation time and dump
  #initial file
  vtk_step = 0
  do_output(vtk_step)
  cb_vtk = GenericCallbacks.EveryXSimulationSteps(floor(outputtime / dt)) do
    vtk_step += 1
    do_output(vtk_step)
    nothing
  end

  cb_filter = GenericCallbacks.EveryXSimulationSteps(1) do
    DGBalanceLawDiscretizations.apply!(Q, 1:_nstate, spatialdiscretization,
                                       filter;
                                       horizontal=false,
                                       vertical=false)
    nothing
  end

  ## Setup a callback to display simulation runtime information
  starttime = Ref(now())
  cb_info = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (init=false)
    if init
      starttime[] = now()
    end
    with_logger(mpi_logger) do
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     norm(Q))
    end
  end

  ## Setup a callback to display simulation runtime information
  starttime = Ref(now())
  cb_info2 = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (init=false)
    if init
      starttime[] = now()
    end
    with_logger(mpi_logger) do
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     norm(Q))
    end
  end

  solve!(Q, lsrk; timeend = finaltime,
         callbacks = (cb_vtk, cb_filter, cb_info, cb_info2))

end
#md nothing # hide

# ### Finalizing MPI (if necessary)
Sys.iswindows() || MPI.finalize_atexit()
Sys.iswindows() && !isinteractive() && MPI.Finalize()
#md nothing # hide

#md # ## [Plain Program](@id held_suarez_balanced-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here:
#md # [ex\_003\_acoustic\_wave.jl](held_suarez_balanced.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
