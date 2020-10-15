# # [Linear NH mountain waves (Topography)](@id EX-LIN_NH-docs)
#
# ## Description of experiment
# 1) Dry linear Non-hydrostatic Mountain Waves
# This example of a non-linear hydrostatic mountain wave can be classified as an initial value
# problem.
#
# The atmosphere is dry and the flow impinges against a witch of Agnesi mountain of heigh $h_m=1m$
# and base parameter $a=1000$m and centered in $x_c = 72km$ in a 2D domain
# $\Omega = 144km \times 30 km$. The mountain is defined as
#
# ```math
#  z = \frac{h_m}{1 + \frac{x - x_c}{a}}
# ```
# The 2D problem is setup in 3D by using 1 element in the y direction.
# To damp the upward moving gravity waves, a Reyleigh absorbing layer is added at $z = 10,000m$.
#
# The initial atmosphere is defined such that it has a stability frequency $N=0.01 s^{-1}$, where
#
# ```math
#  N^2 = g\frac{\rm d \ln \theta}{ \rm dz}
# ```
# so that
#
# ```math
# \theta = \theta_0 \exp\left(\frac{N^2 z}{g} \right),
# ```
# ```math
# \pi = 1 + \frac{g^2}{c_p \theta_0 N^2}\left(\exp\left(\frac{-N^2 z}{g} \right)\right)
# ```
#
# where $\theta_0 = 280K$.
#
# so that
#
# $ρ = \frac{p_{sfc}}{R_{gas}\theta}pi^{c_v/R_{gas}}$
# and $T = \theta \pi$
#
# 2) Boundaries
#    - `Impenetrable(FreeSlip())` - Top and bottom: no momentum flux, no mass flux through
#      walls.
#    - `Impermeable()` - non-porous walls, i.e. no diffusive fluxes through
#       walls.
#    - Agnesi topography built via meshwarp.
#    - Laterally periodic
# 3) Domain - 144,000 m (horizontal) x 1360 m (horizontal) x 30,000m (vertical) (infinite domain in y)
# 4) Resolution - 340 m X 200 m effective resolution
# 5) Total simulation time - 18,000 s
# 6) Overrides defaults for
#    - CPU Initialisation
#    - Time integrator
#    - Sources
#
#md # !!! note
#md #     This experiment setup assumes that you have installed the
#md #     `ClimateMachine` according to the instructions on the landing page.
#md #     We assume the users' familiarity with the conservative form of the
#md #     equations of motion for a compressible fluid (see the
#md #     [AtmosModel](@ref AtmosModel-docs) page).
#md #
#md #     The following topics are covered in this example
#md #     - Defining the initial conditions
#md #     - Applying source terms
#md #     - Add an idealized topography defined by a warping function
#
# ## Boilerplate (Using Modules)
#
# The setup of this problem is taken from Case 6 of:
# @article{giraldoRestelli2008a,
#   author = {{Giraldo},F.~X. and {Restelli},M.},
#   title = {A study of spectral element and discontinuous {G}alerkin methods for the {Navier-Stokes} equations in nonhydrostatic mesoscale atmospheric modeling: {E}quation sets and test cases},
#   journal = {J. Comput. Phys.},
#   year  = {2008},
#   volume = {227},
#   pages  = {3849-3877},
# },
#
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
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
function init_agnesi_hs_lin!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    ## Problem float-type
    FT = eltype(state)

    ## Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    c::FT = c_v / R_gas
    c2::FT = R_gas / c_p

    ## Define initial thermal field as isothermal
    Tiso::FT = 250.0
    θ0::FT = Tiso

    ## Assign a value to the Brunt-Vaisala frquencey:
    Brunt::FT = 0.01
    Brunt2::FT = Brunt * Brunt
    g2::FT = _grav * _grav

    π_exner::FT = exp(-_grav * z / (c_p * Tiso))
    θ::FT = θ0 * exp(Brunt2 * z / _grav)
    ρ::FT = p0 / (R_gas * θ) * (π_exner)^c

    ## Compute perturbed thermodynamic state:
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)

    ## initial velocity
    u = FT(10.0)

    ## State (prognostic) variable assignment
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

# Define a warping function to build an analytic topography:
function warp_agnesi(xin, yin, zin; xmax = 1000.0, ymax = 1000.0, zmax = 1000.0)

    FT = eltype(xin)

    ac = FT(1000)
    hm = FT(1)
    xc = FT(0.5) * xmax
    zdiff = hm / (FT(1) + ((xin - xc) / ac)^2)

    ## Linear relaxation towards domain maximum height
    x, y, z = xin, yin, zin + zdiff * (zmax - zin) / zmax
    return x, y, z
end

# ## [Model Configuration](@id config-helper)
# We define a configuration function to assist in prescribing the physical
# model. The purpose of this is to populate the
# `AtmosLESConfiguration` with arguments
# appropriate to the problem being considered.
function config_agnesi_hs_lin(FT, N, resolution, xmax, ymax, zmax)
    ##
    ## Explicit Rayleigh damping:
    ##
    ## ``
    ##   \tau_s = \alpha * \sin\left(0.5\pi \frac{z - z_s}{zmax - z_s} \right)^2,
    ## ``
    ## where
    ## ``sponge_ampz`` is the wave damping coefficient (1/s)
    ## ``z_s`` is the level where the Rayleigh sponge starts
    ## ``zmax`` is the domain top
    ##
    ## Setup the parameters for the gravity wave absorbing layer
    ## at the top of the domain
    ##
    ## u_relaxation(xvelo, vvelo, wvelo) contains the background velocity values to which
    ## the sponge relaxes the vertically moving wave
    u_relaxation = SVector(FT(10), FT(0), FT(0))

    ## Wave damping coefficient (1/s)
    sponge_ampz = FT(0.5)

    ## Vertical level where the absorbing layer starts
    z_s = FT(10000.0)

    ## Pass the sponge parameters to the sponge calculator
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, z_s, sponge_ampz, u_relaxation, 2)

    ## Define the time integrator:
    ## We chose an explicit single-rate LSRK144 for this problem
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    ## Setup the source terms for this problem:
    source = (Gravity(), rayleigh_sponge)

    ## Define the reference state:
    T_virt = FT(280)
    temp_profile_ref = IsothermalProfile(param_set, T_virt)
    ref_state = HydrostaticState(temp_profile_ref)
    nothing # hide

    _C_smag = FT(0.0)
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = init_agnesi_hs_lin!,
        ref_state = ref_state,
        turbulence = Vreman(_C_smag),
        moisture = DryModel(),
        source = source,
        tracers = NoTracers(),
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "Agnesi_NH_LINEAR",      # Problem title [String]
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

# Define a `main` method (entry point)
function main()

    FT = Float64

    ## Define the polynomial order and effective grid spacings:
    N = 4

    ## Define the domain size and spatial resolution
    Nx = 20
    Ny = 20
    Nz = 20
    xmax = FT(144000)
    ymax = FT(4000)
    zmax = FT(30000)
    Δx = xmax / FT(Nx)
    Δy = ymax / FT(Ny)
    Δz = zmax / FT(Nz)
    resolution = (Δx, Δy, Δz)

    t0 = FT(0)
    timeend = FT(100)

    ## Define the max Courant for the time time integrator (ode_solver).
    ## The default value is 1.7 for LSRK144:
    CFL = FT(1.5)

    ## Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_agnesi_hs_lin(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )

    ## Set up the spectral filter to remove the solutions spurious modes
    ## Define the order of the exponential filter: use 32 or 64 for this problem.
    ## The larger the value, the less dissipation you get:
    filterorder = 64
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
            state_auxiliary = solver_config.dg.state_auxiliary,
        )
        nothing
    end
    ## End exponential filter

    ## Invoke solver (calls `solve!` function for time-integrator),
    ## pass the driver, solver and diagnostic config information.
    result = ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cbfilter,),
        check_euclidean_distance = true,
    )

    ## Check that the solution norm is reasonable.
    @test FT(0.9) < result < FT(1)
    ## Some energy is lost from the sponge, so we cannot
    ## expect perfect energy conservation.
end

# Call `main`
main();
