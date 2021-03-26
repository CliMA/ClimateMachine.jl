using ClimateMachine, MPI
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods

using ClimateMachine.ODESolvers

using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude

using CLIMAParameters
using CLIMAParameters.Planet: MSLP, R_d, day, grav, Omega, planet_radius

using ClimateMachine.Coupling
using Unitful
using Dates: DateTime
using Statistics
using StaticArrays

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
include("CplMainBL.jl") #include("test_model.jl") # umbrella model: TestEquations
#using .CplTestingBL

const dt = 3600.0
nstepsA = 10
nstepsO = 5

#  Background atmos and ocean diffusivities
const κᵃʰ = FT(1e4) * 0.0
const κᵃᶻ = FT(1e-1)
const κᵒʰ = FT(1e3) * 0.0
const κᵒᶻ = FT(1e-4)
const τ_airsea = FT(60 * 86400)
const L_airsea = FT(500)
const λ_airsea = FT(L_airsea / τ_airsea)
function coupling_lambda()
    return (λ_airsea)
end

function main(::Type{FT}) where {FT}
    
    # Domain
    # better might be to use something like DeepSphericalShellDomain directly?
    ΩO = AtmosDomain(radius = FT(planet_radius(param_set)) - FT(4e3), height = FT(4e3))
    ΩA = AtmosDomain(radius = FT(planet_radius(param_set)) , height = FT(4e3))

    # Grid
    nelem = (;horizontal = 8, vertical = 4)
    polynomialorder = (;horizontal = 5, vertical = 5)
    
    gridA = DiscontinuousSpectralElementGrid(ΩA, nelem, polynomialorder)
    gridO = DiscontinuousSpectralElementGrid(ΩO, nelem, polynomialorder)

    #dx = min_node_distance(grid, HorizontalDirection())

    # Numerics-specific options
    numerics = (NFsecondorder = PenaltyNumFluxDiffusive(),) # add  , overintegration = 1

    # Timestepping
    Δt_ = dt
    t_time, end_time = ( 0  , 20Δt_ )

    # Collect spatial info, timestepping, balance law and DGmodel for the two components
    function boundary_mask( xc, yc, zc )
            bndy_bit_mask = @. ( xc^2 + yc^2 + zc^2 )^0.5 ≈ planet_radius(param_set)
            return bndy_bit_mask
    end

    # 1. Atmos component
    mA = Coupling.CplTestModel(;
        grid = gridA,
        equations = CplTestBL(
            bl_propA,
            (CoupledPrimaryBoundary(), ExteriorBoundary()),
            param_set,
        ),
        boundary_z = boundary_mask,
        nsteps = nstepsA,
        dt = Δt_ / nstepsA,
        timestepper = LSRK54CarpenterKennedy,
        numerics...,
    )

    # 2. Ocean component
    mO = Coupling.CplTestModel(;
        grid = gridO,
        equations = CplTestBL(
            bl_propO,
            (ExteriorBoundary(), CoupledSecondaryBoundary()),
            param_set,
        ),
        boundary_z = boundary_mask,
        nsteps = nstepsO,
        dt = Δt_ / nstepsO,
        timestepper = LSRK54CarpenterKennedy,
        numerics...,
    )

    # Create a Coupler State object for holding import/export fields.
    # Try using Dict here - not sure if that will be OK with GPU
    coupler = CplState()
    register_cpl_field!(coupler, :Ocean_SST, deepcopy(mO.state.θ[mO.boundary]), mO.grid, DateTime(0), u"°C")
    register_cpl_field!(coupler, :Atmos_MeanAirSeaθFlux, deepcopy(mA.state.F_accum[mA.boundary]), mA.grid, DateTime(0), u"°C")
    

    # Instantiate a coupled timestepper that steps forward the components and
    # implements mapings between components export bondary states and
    # other components imports.

    compA = (pre_step = preatmos, component_model = mA, post_step = postatmos)
    compO = (pre_step = preocean, component_model = mO, post_step = postocean)
    component_list = (atmosphere = compA, ocean = compO)
    cpl_solver = Coupling.CplSolver(
        component_list = component_list,
        coupler = coupler,
        coupling_dt = Δt_,
        t0 = 0.0,
    )
    
    callbacks = (
        VTKOutput((
            iteration = string(1Δt_)*"ssecs" ,
            overdir ="output",
            overwrite = true,
            number_sample_points = 0
            )...,),
    )

    simulation = (;
        coupled_odesolver = cpl_solver,
        odesolver = cpl_solver.component_list.atmosphere.component_model.stepper,
        state = cpl_solver.component_list.atmosphere.component_model.state,
        dgmodel =  cpl_solver.component_list.atmosphere.component_model.discretization,
        callbacks = callbacks,
        simtime = (t_time, end_time),
        name = "Coupler_UnitTest_atmosphere_long",
        )

    return simulation
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

## Prop atmos functions (or delete to use defaults)
atmos_θⁱⁿⁱᵗ(npt, el, xc, yc, zc) = 30.0                   # Set atmosphere initial state function

function is_surface(xc, yc, zc)
 # Sphere case - could dispatch on domain type, maybe?
 height_from_surface=(xc^2 + yc^2 + zc^2)^0.5 - planet_radius(param_set)
 return height_from_surface ≈ 0
end

atmos_θ_shadowflux(θᵃ, θᵒ, npt, el, xc, yc, zc) = is_surface(xc,yc,zc) ? (1.0 / τ_airsea) * (θᵃ - θᵒ) : 0.0 # Set atmosphere shadow boundary flux function
atmos_calc_kappa_diff(_...) = κᵃʰ, κᵃʰ, κᵃᶻ               # Set atmos diffusion coeffs
atmos_source_θ(θᵃ, npt, el, xc, yc, zc, θᵒ) = FT(0.0)     # Set atmos source!
atmos_get_penalty_tau(_...) = FT(3.0 * 0.0)               # Set penalty term tau (for debugging)

## Set atmos advective velocity (constant in time)
# uˡᵒⁿ(λ, ϕ, r) = 1e-6 * r * cos(ϕ)
uˡᵒⁿ(λ, ϕ, r) = 0 * 1e-6 * r * cos(ϕ)
atmos_uⁱⁿⁱᵗ(npt, el, x, y, z) = (     0 * r̂(x,y,z) 
                                    + 0 * ϕ̂(x,y,z)
                                    + uˡᵒⁿ(lon(x,y,z), lat(x,y,z), rad(x,y,z)) * λ̂(x,y,z) ) 

## Collect atmos props
bl_propA = prop_defaults()
bl_propA = (;bl_propA..., 
            init_theta = atmos_θⁱⁿⁱᵗ, 
            theta_shadow_boundary_flux = atmos_θ_shadowflux, 
            calc_kappa_diff = atmos_calc_kappa_diff,
            source_theta = atmos_source_θ,
            get_penalty_tau = atmos_get_penalty_tau,
            coupling_lambda = coupling_lambda,
            init_u = atmos_uⁱⁿⁱᵗ
            )

## Prop ocean functions (or delete to use defaults)
tropical_heating(λ, ϕ, r) = 30.0 + 10.0 * cos(ϕ) * sin(5λ)
ocean_θⁱⁿⁱᵗ(npt, el, x, y, z) = tropical_heating( lon(x,y,z), lat(x,y,z), rad(x,y,z) )                    # Set ocean initial state function
ocean_calc_kappa_diff(_...) = κᵒʰ, κᵒʰ, κᵒᶻ               # Set ocean diffusion coeffs
ocean_source_θ(θᵃ, npt, el, xc, yc, zc, θᵒ) = FT(0.0)     # Set ocean source!
ocean_get_penalty_tau(_...) = FT(0.15 * 0.0)               # Set penalty term tau (for debugging)
ocean_uⁱⁿⁱᵗ(xc, yc, zc, npt, el) = SVector(FT(0.0), FT(0.0), FT(0.0)) # Set ocean advective velocity

## Collect ocean props
bl_propO = prop_defaults()
bl_propO = (;bl_propO..., 
            init_theta = ocean_θⁱⁿⁱᵗ, 
            calc_kappa_diff = ocean_calc_kappa_diff,
            source_theta = ocean_source_θ,
            get_penalty_tau = ocean_get_penalty_tau,
            coupling_lambda = coupling_lambda,
            init_u = ocean_uⁱⁿⁱᵗ,
            )

simulation = main(Float64);
nsteps = Int(simulation.simtime[2] / dt)
cbvector = create_callbacks(simulation, simulation.odesolver)
println("Initialized. Running...")
@time run(simulation.coupled_odesolver, nsteps, cbvector)
