using ClimateMachine, MPI
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods

using ClimateMachine.ODESolvers

using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, Omega, planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

ClimateMachine.init()
FT = Float64

# Shared functions
include("domains.jl")
include("interface.jl")
include("abstractions.jl")
include("callbacks.jl")

# Main balance law and its components
include("CplTestingBL.jl") #include("test_model.jl") # umbrella model: TestEquations

using .CplTestingBL


const dt = 3600.0
nstepsA = 5
nstepsO = 5

#  Background atmos and ocean diffusivities
const κᵃʰ = FT(1e4) * 0.0
const κᵃᶻ = FT(1e-1)
const κᵒʰ = FT(1e3) * 0.0
const κᵒᶻ = FT(1e-4)


function main(::Type{FT}) where {FT}

    # BL and problem for this
    diffusionA = DiffusionCubedSphereProblem{FT}((;
                            τ = 8*60*60,
                            l = 7,
                            m = 4,
                            )...,
                        )
    
    diffusionO = DiffusionCubedSphereProblem{FT}((;
                            τ = 8*60*60,
                            l = 7,
                            m = 4,
                            )...,
                        )                        

    # Domain
    ΩO = AtmosDomain(radius = FT(planet_radius(param_set)), height = FT(4e3))
    ΩA = AtmosDomain(radius = FT(planet_radius(param_set)) + FT(4e3), height = FT(4e3))

    # Grid
    nelem = (;horizontal = 8, vertical = 4)
    polynomialorder = (;horizontal = 5, vertical = 5)
    
    # gridA = DiscontinuousSpectralElementGrid(ΩA, nelem, polynomialorder)
    # gridO = DiscontinuousSpectralElementGrid(ΩO, nelem, polynomialorder)

    #dx = min_node_distance(grid, HorizontalDirection())

    # Numerics-specific options
    numerics = (; flux = CentralNumericalFluxFirstOrder() ) # add  , overintegration = 1

    # Timestepping
    Δt_ = dt
    t_time, end_time = ( 0  , 2Δt_ )

    # Callbacks (TODO)
    callbacks = ()

    # Collect spatial info, timestepping, balance law and DGmodel for the two components

    # 1. Atmos component
    ## Set atmosphere initial state function
    function atmos_init_theta(xc, yc, zc, npt, el)
        return 30.0
    end
    ## Set atmosphere shadow boundary flux function
    function atmos_theta_shadow_boundary_flux(θᵃ, θᵒ, npt, el, xc, yc, zc)
        if zc == 0.0
            tflux = (1.0 / τ_airsea) * (θᵃ - θᵒ)
        else
            tflux = 0.0
        end
        return tflux
    end
    ## Set atmsophere diffusion coeffs
    function atmos_calc_kappa_diff(_...)
        return κᵃʰ, κᵃʰ, κᵃᶻ
    end
    ## Set atmos source!
    function atmos_source_theta(θᵃ, npt, el, xc, yc, zc, θᵒ)
        tsource = 0.0
        if zc == 0.0
            #tsource = -(1. / τ_airsea)*( θᵃ-θᵒ )
        end
        return tsource
    end
    ## Set penalty term tau (for debugging)
    function atmos_get_penalty_tau(_...)
        return FT(3.0 * 0.0)
    end
    ## Create atmos component
    bl_prop = CplTestingBL.prop_defaults()

    bl_prop = (;bl_prop..., init_theta = atmos_init_theta, 
                theta_shadow_boundary_flux = atmos_theta_shadow_boundary_flux)


    bl_prop = (bl_prop..., init_theta = atmos_init_theta)
    bl_prop =
        (bl_prop..., theta_shadow_boundary_flux = atmos_theta_shadow_boundary_flux)
    bl_prop = (bl_prop..., calc_kappa_diff = atmos_calc_kappa_diff)
    bl_prop = (bl_prop..., source_theta = atmos_source_theta)
    bl_prop = (bl_prop..., get_penalty_tau = atmos_get_penalty_tau)
    bl_prop = (bl_prop..., coupling_lambda = coupling_lambda)
    mA = Coupling.CplTestModel(;
        domain = ΩA,
        equations = CplTestBL(
            bl_prop,
            (CoupledPrimaryBoundary(), ExteriorBoundary()),
        ),
        nsteps = nstepsA,
        dt = Δt_ / nstepsA,
        timestepper = LSRK54CarpenterKennedy,
        NFSecondOrder = CplTestingBL.PenaltyNumFluxDiffusive(),
    )

    # 2. Ocean component
    ## Set initial temperature profile
    function ocean_init_theta(xc, yc, zc, npt, el)
        return 20.0
    end
    ## Set boundary source imported from atmos
    function ocean_source_theta(θ, npt, el, xc, yc, zc, air_sea_flux_import)
        sval = 0.0
        if zc == 0.0
            #sval=air_sea_flux_import
        end
        return sval
    end
    ## Set ocean diffusion coeffs
    function ocean_calc_kappa_diff(_...)
        # return κᵒʰ,κᵒʰ,κᵒᶻ*FT(100.)
        return κᵒʰ, κᵒʰ, κᵒᶻ # m^2 s^-1
    end
    ## Set penalty term tau (for debugging)
    function ocean_get_penalty_tau(_...)
        return FT(0.15 * 0.0)
    end
    ## Create ocean component
    bl_prop = CplTestingBL.prop_defaults()
    bl_prop = (bl_prop..., init_theta = ocean_init_theta)
    bl_prop = (bl_prop..., source_theta = ocean_source_theta)
    bl_prop = (bl_prop..., calc_kappa_diff = ocean_calc_kappa_diff)
    bl_prop = (bl_prop..., get_penalty_tau = ocean_get_penalty_tau)
    bl_prop = (bl_prop..., coupling_lambda = coupling_lambda)
    mO = Coupling.CplTestModel(;
        domain = ΩO,
        equations = CplTestBL(
            bl_prop,
            (ExteriorBoundary(), CoupledSecondaryBoundary()),
        ),
        nsteps = nstepsO,
        dt = Δt_ / nstepsO,
        timestepper = LSRK54CarpenterKennedy,
        NFSecondOrder = CplTestingBL.PenaltyNumFluxDiffusive(),
    )

    # Create a Coupler State object for holding imort/export fields.
    # Try using Dict here - not sure if that will be OK with GPU
    coupler = CplState()
    register_cpl_field!(cState, :Ocean_SST, deepcopy(mO.state.θ[mO.boundary]), mO.grid, DateTime(0), u"°C")
    register_cpl_field!(cState, :Atmos_MeanAirSeaθFlux, deepcopy(mA.state.F_accum[mA.boundary]), mA.grid, DateTime(0), u"°C")
    

    # Instantiate a coupled timestepper that steps forward the components and
    # implements mapings between components export bondary states and
    # other components imports.

    compA = (pre_step = preatmos, component_model = mA, post_step = postatmos)
    compO = (pre_step = preocean, component_model = mO, post_step = postocean)
    component_list = (atmosphere = compA, ocean = compO)
    cpl_solver = Coupling.CplSolver(
        component_list = component_list,
        coupler = coupler,
        coupling_dt = couple_dt,
        t0 = 0.0,
)

    cbvector = create_callbacks(simulation, simulation.odesolver)
    
    return simulation, end_time, cbvector
end


function run(cpl_solver, numberofsteps, cbvector)
    # Run the model
    solve!(
        nothing,
        cpl_solver;
        numberofsteps = numberofsteps,
        callbacks = cbvector,
    )
end

simulation, end_time, cbvector = main(Float64);
println("Initialized. Running...")
@time run(simulation, end_time, cbvector)



# potentially move out:
function preatmos(csolver)
    mA = csolver.component_list.atmosphere.component_model
    mO = csolver.component_list.ocean.component_model
    # Set boundary SST used in atmos to SST of ocean surface at start of coupling cycle.
    mA.discretization.state_auxiliary.θ_secondary[mA.boundary] .= 
        Coupling.get(csolver.coupler, :Ocean_SST, mA.grid, DateTime(0), u"°C")
    # Set atmos boundary flux accumulator to 0.
    mA.state.F_accum .= 0

    @info(
        "preatmos",
        time = csolver.t,
        total_θ_atmos = weightedsum(mA.state, 1),
        total_θ_ocean = weightedsum(mO.state, 1),
        total_θ = weightedsum(mA.state, 1) + weightedsum(mO.state, 1),
        atmos_θ_surface_max = maximum(mA.state.θ[mA.boundary]),
        ocean_θ_surface_max = maximum(mO.state.θ[mO.boundary]),
    )
end

function postatmos(csolver)
    mA = csolver.component_list.atmosphere.component_model
    mO = csolver.component_list.ocean.component_model
    # Pass atmos exports to "coupler" namespace
    # 1. Save mean θ flux at the Atmos boundary during the coupling period
    Coupling.put!(csolver.coupler, :Atmos_MeanAirSeaθFlux, mA.state.F_accum[mA.boundary] ./ csolver.dt,
        mA.grid, DateTime(0), u"°C")

    @info(
        "postatmos",
        time = time = csolver.t + csolver.dt,
        total_θ_atmos = weightedsum(mA.state, 1),
        total_θ_ocean = weightedsum(mO.state, 1),
        total_F_accum = mean(mA.state.F_accum[mA.boundary]) * 1e6 * 1e6,
        total_θ =
            weightedsum(mA.state, 1) +
            weightedsum(mO.state, 1) +
            mean(mA.state.F_accum[mA.boundary]) * 1e6 * 1e6,
        F_accum_max = maximum(mA.state.F_accum[mA.boundary]),
        F_avg_max = maximum(mA.state.F_accum[mA.boundary] ./ csolver.dt),
        atmos_θ_surface_max = maximum(mA.state.θ[mA.boundary]),
        ocean_θ_surface_max = maximum(mO.state.θ[mO.boundary]),
    )
end

function preocean(csolver)
    mA = csolver.component_list.atmosphere.component_model
    mO = csolver.component_list.ocean.component_model
    # Set mean air-sea theta flux
    mO.discretization.state_auxiliary.F_prescribed[mO.boundary] .= 
        Coupling.get(csolver.coupler, :Atmos_MeanAirSeaθFlux, mO.grid, DateTime(0), u"°C")
    # Set ocean boundary flux accumulator to 0. (this isn't used)
    mO.state.F_accum .= 0

    @info(
        "preocean",
        time = csolver.t,
        F_prescribed_max =
            maximum(mO.discretization.state_auxiliary.F_prescribed[mO.boundary]),
        F_prescribed_min =
            maximum(mO.discretization.state_auxiliary.F_prescribed[mO.boundary]),
        ocean_θ_surface_max = maximum(mO.state.θ[mO.boundary]),
        ocean_θ_surface_min = maximum(mO.state.θ[mO.boundary]),
    )
end

function postocean(csolver)
    mA = csolver.component_list.atmosphere.component_model
    mO = csolver.component_list.ocean.component_model
    @info(
        "postocean",
        time = csolver.t + csolver.dt,
        ocean_θ_surface_max = maximum(mO.state.θ[mO.boundary]),
        ocean_θ_surface_min = maximum(mO.state.θ[mO.boundary]),
    )

    # Pass ocean exports to "coupler" namespace
    #  1. Ocean SST (value of θ at z=0)
    Coupling.put!(csolver.coupler, :Ocean_SST, mO.state.θ[mO.boundary], mO.grid, DateTime(0), u"°C")
end
