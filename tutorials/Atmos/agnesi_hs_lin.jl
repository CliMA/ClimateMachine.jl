# # [Linear HS mountain waves](@id EX-LINHS-docs)
#
# This test is designed to test the numerical accuracy in preserving
# discrete hydrostatic balance.
#
# ## Description of experiment
# 1) Dry Nonlinear Hydrostatic Mountain Waves
# This example of a non-linear hydrostatic mountain wave can be classified as an initial value
# problem.
#
# The atmosphere is dry and the flow impinges against a witch of Agnesi mountain of heigh `hm=1 m`
# and base parameter `a=1,000` m and centered in `xc = 120 km` in a 2D domain
# `\Omega = 240 km\times 30 km`. The mountain is defined as
#
# ``
#  z = \frac{hm}{1 + \frac{x - xc}{a}}
# ``
# The 2D problem is setup in 3D by using 1 element in the y direction.
# To damp the upward moving gravity waves, a Reyleigh absorbing layer is added at `z = 15,000 m`.
#
# The initial atmosphere is defined such that it has a stability frequency `N=0.02 1/s`, where
#
# ``
#  N^2 = g\frac{\rm d \ln \theta}{ \rm dz}
# ``
# so that
#
# ``
# \theta = \theta_0 \exp\left(\frac{N^2 z}{g} \right),
# ``
# ``
# \pi = 1 + \frac{g^2}{c_p \theta_0 N^2}\left(\exp\left(\frac{-N^2 z}{g} \right)\right)
#
# where `theta_0 = 273 K`.
# ``
# so that
#
# ``
# ρ = \frac{p_{sfc}}{R_{gas}\theta}pi^{c_v/R_{gas}}
# ``
# and
# ``
# \theta \pi
# ``
#---------------------------------------------------------------------
#
# 2) Boundaries
#    - `Impenetrable(FreeSlip())` - Top and bottom: no momentum flux, no mass flux through
#      walls.
#    - `Impermeable()` - non-porous walls, i.e. no diffusive fluxes through
#       walls.
#    - Agnesi topography built via meshwarp.
#    - Laterally periodic
# 3) Domain - 500,000 m (horizontal) x 18800 m (horizontal) x 30,000m (vertical) (infinite domain in y)
# 4) Resolution - 2720m X 160 m effective resolution
# 5) Total simulation time - 22.5 hours
# 6) Mesh Aspect Ratio (Effective resolution) 17:1
# 7) Overrides defaults for
#    - CPU Initialisation
#    - Time integrator
#    - Sources
#

#md # !!! note
#md #     This experiment setup assumes that you have installed the
#md #     `ClimateMachine` according to the instructions on the landing page.
#md #     We assume the users' familiarity with the conservative form of the
#md #     equations of motion for a compressible fluid (see the
#md #     [`AtmosModel`](@ref AtmosModel-docs) page).
#md #
#md #     The following topics are covered in this example
#md #     - Package requirements
#md #     - Defining a `model` subtype for the set of conservation equations
#md #     - Defining the initial conditions
#md #     - Applying source terms
#md #     - Choosing a turbulence model
#md #     - Adding tracers to the model
#md #     - Choosing a time-integrator
#md #
#md #     The following topics are not covered in this example
#md #     - Defining new boundary conditions
#md #     - Defining new turbulence models
#md #     - Building new time-integrators
#
# ## Boilerplate (Using Modules)
#
# #### [Skip Section](@ref init)
#
# Before setting up our experiment, we recognize that we need to import some
# pre-defined functions from other packages. Julia allows us to use existing
# modules (variable workspaces), or write our own to do so.  Complete
# documentation for the Julia module system can be found
# [here](https://docs.julialang.org/en/v1/manual/modules/#).

# We need to use the `ClimateMachine` module! This imports all functions
# specific to atmospheric and ocean flow modelling.  While we do not cover the
# ins-and-outs of the contents of each of these we provide brief descriptions
# of the utility of each of the loaded packages.
#
# The setup of this problem is taken from Case 6 of:
# @article{giraldoRestelli2008a,
#   author = {{Giraldo},F.~X. and {Restelli},M.},
#   title = {A study of spectral element and discontinuous {G}alerkin methods for the {Navier-Stokes} equations in nonhydrostatic mesoscale atmospheric modeling: {E}quation sets and test cases},
#   journal = {J. Comput. Phys.},
#   year  = {2008},
#   volume = {227},
#   pages  = {3849-3877},
#},
#
using ClimateMachine
ClimateMachine.cli()

using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.VariableTemplates
using StaticArrays
using Test

using CLIMAParameters
using CLIMAParameters.Atmos.SubgridScale: C_smag
using CLIMAParameters.Planet: R_d, cp_d, cv_d, MSLP, grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# ## [Initial Conditions](@id init)
#md # !!! note
#md #     The following variables are assigned in the initial condition
#md #     - `state.ρ` = Scalar quantity for initial density profile
#md #     - `state.ρu`= 3-component vector for initial momentum profile
#md #     - `state.ρe`= Scalar quantity for initial total-energy profile
#md #       humidity
function init_agnesi_hs_lin!(bl, state, aux, (x, y, z), t)
    # Problem float-type
    FT = eltype(state)

    # Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    c::FT = c_v / R_gas
    c2::FT = R_gas / c_p
    
    Tiso::FT = 250.0
    θ0::FT = Tiso

    Brunt::FT = _grav / sqrt(c_p * Tiso)
    Brunt2::FT = Brunt * Brunt
    g2::FT = _grav * _grav

    π_exner::FT = exp(-_grav * z / (c_p * Tiso))
    θ::FT = θ0 * exp(Brunt2 * z / _grav)
    ρ::FT = p0 / (R_gas * θ) * (π_exner)^c

    # Compute perturbed thermodynamic state:
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)

    #initial velocity
    u = FT(20.0)
    
    #State (prognostic) variable assignment
    e_kin = FT(0)                                       # kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)# potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy

    state.ρ = ρ
    state.ρu = SVector{3, FT}(ρ * u, 0, 0)
    state.ρe = ρe_tot
end

function setmax(f, xmax, ymax, zmax)
    function setmaxima(xin, yin, zin)
        return f(xin, yin, zin; xmax = xmax, ymax = ymax, zmax = zmax)
    end
    return setmaxima
end

function warp_agnesi(xin, yin, zin; xmax = 1000.0, ymax = 1000.0, zmax = 1000.0)

    FT = eltype(xin)

    ac = FT(10000)
    hm = FT(1)
    xc = FT(0.5) * xmax
    zdiff = hm / (FT(1) + ((xin - xc) / ac)^2)

    # Linear relaxation towards domain maximum height
    x, y, z = xin, yin, zin + zdiff * (zmax - zin) / zmax
    return x, y, z
end

# ## [Model Configuration](@id config-helper)
# We define a configuration function to assist in prescribing the physical
# model. The purpose of this is to populate the
# [`ClimateMachine.AtmosLESConfiguration`](@ref LESConfig) with arguments
# appropriate to the problem being considered.
function config_agnesi_hs_lin(FT, N, resolution, xmax, ymax, zmax)

    u_relaxation = SVector(FT(20), FT(0), FT(0))
    sponge_ampz = FT(0.5)
    sponge_ampx = FT(0.1)
    xmin = FT(0)
    
    # Rayleigh damping
    depthsponge_x = FT(70000.0)
    zsponge_base = FT(25000.0)
    
#=    rayleigh_sponge =
    RayleighSponge{FT}(zmax, zsponge_base, sponge_ampz, u_relaxation, 2)=#
    rayleigh_sponge =
        RayleighSpongeTopLateral{FT}(xmax, zmax, depthsponge_x, zsponge_base, sponge_ampz, sponge_ampx, u_relaxation, 4)
    
    ##SR LSRK144
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )
    
    source = (Gravity(), rayleigh_sponge)
    
    #temp_profile_ref = DecayingTemperatureProfile{FT}(param_set)
    T_virt = FT(250)
    temp_profile_ref = IsothermalProfile(param_set, T_virt)
    ref_state = HydrostaticState(temp_profile_ref)
    nothing # hide
    
    _C_smag = FT(0.21)
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = Vreman(_C_smag),
        moisture = DryModel(),
        source = source,
        tracers = NoTracers(),
        init_state_conservative = init_agnesi_hs_lin!,
        ref_state = ref_state,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "Agnesi_HS_LINEAR",      # Problem title [String]
        N,                       # Polynomial order [Int]
        resolution,              # (Δx, Δy, Δz) effective resolution [m]
        xmax,                    # Domain maximum size [m]
        ymax,                    # Domain maximum size [m]
        zmax,                    # Domain maximum size [m]
        param_set,               # Parameter set.
        init_agnesi_hs_lin!,     # Function specifying initial condition
        solver_type = ode_solver,# Time-integrator type
        model = model,           # Model type
        meshwarp = setmax(warp_agnesi, xmax, ymax, zmax),
    )

    return config
end

function main()

    FT = Float64

    N = 4
    Δh = FT(1000)
    Δv = FT(240)
    resolution = (Δh, Δh, Δv)
    xmax = FT(244000)
    ymax = FT(4000)
    zmax = FT(50000)
    t0 = FT(0)
    timeend =  FT(15000) #FT(hrs * 60 * 60)

    Courant = FT(0.5)

    # Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_agnesi_hs_lin(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = Courant,
    )

    # User defined filter (TMAR positivity preserving filter)
    #cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
    #    Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
    #    nothing
    #end

    #Exponential filter:
#=    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        @views begin
            solver_config.Q.data[:, 2, :] .-= 20*solver_config.Q[:,1,:]
            Filters.apply!(
                solver_config.Q,
                2:4,
                solver_config.dg.grid,
                filter,
            )
            solver_config.Q.data[:, 2, :] .+= 20*solver_config.Q[:,1,:]
        end
        nothing
    end
=#
    #End exponential filter
    
    # Invoke solver (calls `solve!` function for time-integrator), pass the driver, solver and diagnostic config
    # information.
    result = ClimateMachine.invoke!(
        solver_config;
        #user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )

    # Check that the solution norm is reasonable.
    @test isapprox(result, FT(1); atol = 1.5e-3)
end
#
main()
