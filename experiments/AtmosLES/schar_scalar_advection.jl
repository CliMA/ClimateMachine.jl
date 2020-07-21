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

### Citation
#@article{
#    author = {Schär, Christoph and 
#              Leuenberger, Daniel and 
#              Fuhrer, Oliver and 
#              Lüthi, Daniel and 
#              Girard, Claude},
#    title = "{A New Terrain-Following Vertical Coordinate Formulation 
#              for Atmospheric Prediction Models}",
#    journal = {Monthly Weather Review},
#    volume = {130},
#    number = {10},
#    pages = {2459-2480},
#    year = {2002},
#    month = {10},
#    issn = {0027-0644},
#    doi = {10.1175/1520-0493(2002)130<2459:ANTFVC>2.0.CO;2},
#    url = {https://doi.org/10.1175/1520-0493(2002)130<2459:ANTFVC>2.0.CO;2},
#}

# ## [Initial Conditions]
function init_schar!(bl, state, aux, (x, y, z), t)
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

    Tiso::FT = 250.0
    θ0::FT = Tiso

    ## Calculate the Brunt-Vaisaila frequency for an isothermal field
    ## Hydrostatic background state
    Brunt::FT = _grav / sqrt(c_p * Tiso)
    Brunt2::FT = Brunt * Brunt
    g2::FT = _grav * _grav

    π_exner::FT = exp(-_grav * z / (c_p * Tiso))
    θ::FT = θ0 * exp(Brunt2 * z / _grav)
    ρ::FT = p0 / (R_gas * θ) * (π_exner)^c

    ## Compute perturbed thermodynamic state:
    T = θ * π_exner
    e_int = internal_energy(bl.param_set, T)
    ts = PhaseDry(bl.param_set, e_int, ρ)

    ## Initial velocity
    z₁::FT = 4000
    z₂::FT = 5000
    u₀::FT = 10
    zscale = (z - z₁) / (z₂ - z₁)
    if z₂ <= z
        u = FT(1)
    elseif z₁ <= z < z₂
        u = (sinpi(zscale / 2))^2
    elseif z <= z₁
        u = FT(0)
    end
    u *= u₀

    ## Initial scalar anomaly profile
    ## Equivalent to a nondiffusive tracer
    Ax::FT = 25000
    Az::FT = 3000
    x₀::FT = 25000
    z₀::FT = 9000
    r = ((x - x₀) / Ax)^2 + ((z - z₀) / Az)^2
    if r <= 1
        χ = (cospi(r / 2))^2
    else
        χ = 0
    end

    ## State (prognostic) variable assignment
    e_kin = FT(1 / 2) * u^2                               # kinetic energy
    e_pot = gravitational_potential(bl.orientation, aux)# potential energy
    ρe_tot = ρ * total_energy(e_kin, e_pot, ts)         # total energy

    state.ρ = ρ
    state.ρu = SVector{3, FT}(ρ * u, 0, 0)
    state.ρe = ρe_tot
    state.tracers.ρχ = ρ * SVector{1, FT}(χ)
end

# Define a `setmax` method
function setmax(f, xmax, ymax, zmax)
    function setmaxima(xin, yin, zin)
        return f(xin, yin, zin; xmax = xmax, ymax = ymax, zmax = zmax)
    end
    return setmaxima
end

function warp_schar(
    xin,
    yin,
    zin;
    xmax = 150000.0,
    ymax = 5000.0,
    zmax = 25000.0,
)
    FT = eltype(xin)
    a::FT = 25000 ## Half-width parameter [m]
    r = sqrt(xin^2 + yin^2)
    h₀::FT = 3000 ## Peak height [m]
    λ::FT = 8000 ## Wavelength
    h_star =
        abs(xin - xmax / 2) <= a ? h₀ * (cospi((xin - xmax / 2) / 2a))^2 : FT(0)
    h = h_star * (cospi((xin - xmax / 2) / λ))^2
    x, y, z = xin, yin, zin + h * (zmax - zin) / zmax
    return x, y, z
end

function config_schar(FT, N, resolution, xmax, ymax, zmax)
    u_relaxation = SVector(FT(10), FT(0), FT(0))

    ## Wave damping coefficient (1/s)
    sponge_ampz = FT(0.5)

    ## Vertical level where the absorbing layer starts
    z_s = FT(20000.0)

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
    T_virt = FT(250)
    temp_profile_ref = IsothermalProfile(param_set, T_virt)
    ref_state = HydrostaticState(temp_profile_ref)
    nothing # hide

    # Define a warping function to build an analytic topography:

    _C_smag = FT(0.21)
    _δχ = SVector{1, FT}(0)
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        turbulence = Vreman(_C_smag),
        moisture = DryModel(),
        source = source,
        tracers = NTracers{1, FT}(_δχ),
        init_state_prognostic = init_schar!,
        ref_state = ref_state,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "ScharScalarAdvection",  # Problem title [String]
        N,                       # Polynomial order [Int]
        resolution,              # (Δx, Δy, Δz) effective resolution [m]
        xmax,                    # Domain maximum size [m]
        ymax,                    # Domain maximum size [m]
        zmax,                    # Domain maximum size [m]
        param_set,               # Parameter set.
        init_schar!,             # Function specifying initial condition
        solver_type = ode_solver,# Time-integrator type
        model = model,           # Model type
        meshwarp = setmax(warp_schar, xmax, ymax, zmax),
    )

    return config
end

# Define a `main` method (entry point)
function main()

    FT = Float64

    ## Define the polynomial order and effective grid spacings:
    N = 4

    ## Define the domain size and spatial resolution
    xmax = FT(150000)
    ymax = FT(2500)
    zmax = FT(25000)
    Δx = FT(500)
    Δy = FT(500)
    Δz = FT(500)
    resolution = (Δx, Δy, Δz)

    t0 = FT(0)
    timeend = FT(10000)

    ## Define the max Courant for the time time integrator (ode_solver).
    ## The default value is 1.7 for LSRK144:
    CFL = FT(1.5)

    ## Assign configurations so they can be passed to the `invoke!` function
    driver_config = config_schar(FT, N, resolution, xmax, ymax, zmax)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )

    ## State Conservation Callback
    # State variable
    Q = solver_config.Q
    # Volume geometry information
    vgeo = driver_config.grid.vgeo
    M = vgeo[:, Grids._M, :]
    # Unpack prognostic vars
    ρχ₀ = Q[:, 6, :]
    # DG variable sums
    Σρχ₀ = sum(ρχ₀ .* M)
    cb_check_tracer = GenericCallbacks.EveryXSimulationSteps(1000) do
        Q = solver_config.Q
        δρχ = (sum(Q[:, 6, :] .* M) .- Σρχ₀) ./ Σρχ₀
        @show (abs(δρχ))
        nothing
    end

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
        user_callbacks = (cbfilter, cb_check_tracer),
        check_euclidean_distance = true,
    )

    ## Check that the solution norm is reasonable.
    @test isapprox(result, FT(1); atol = 1.5e-4)
end

# Call `main`
main()
