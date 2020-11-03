# Test heat equation agrees with analytic solution to problem 55 on page 28 in https://ocw.mit.edu/courses/mathematics/18-303-linear-partial-differential-equations-fall-2006/lecture-notes/heateqni.pdf
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.DGMethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state

@testset "Heat analytic unit test" begin
    ClimateMachine.init()
    FT = Float32

    function init_soil!(land, state, aux, localgeo, time)
        myFT = eltype(state)
        ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, time)
        θ_l =
            volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
        ρc_s = volumetric_heat_capacity(
            θ_l,
            θ_i,
            land.soil.param_functions.ρc_ds,
            land.param_set,
        )

        state.soil.heat.ρe_int = myFT(volumetric_internal_energy(
            θ_i,
            ρc_s,
            land.soil.heat.initialT(aux),
            land.param_set,
        ))
    end

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.495,
        ν_ss_gravel = 0.1,
        ν_ss_om = 0.1,
        ν_ss_quartz = 0.1,
        ρc_ds = 0.43314518988433487,
        κ_solid = 8,
        ρp = 2700,
        κ_sat_unfrozen = 0.57,
        κ_sat_frozen = 2.29,
    )

    heat_surface_state = (aux, t) -> eltype(aux)(0.0)

    tau = FT(1) # period (sec)
    A = FT(5) # amplitude (K)
    ω = FT(2 * pi / tau)
    heat_bottom_state = (aux, t) -> A * cos(ω * t)
    T_init = (aux) -> eltype(aux)(0.0)

    soil_water_model = PrescribedWaterModel(
        (aux, t) -> eltype(aux)(0.0),
        (aux, t) -> eltype(aux)(0.0),
    )

    soil_heat_model = SoilHeatModel(
        FT;
        initialT = T_init,
        dirichlet_bc = Dirichlet(
            surface_state = heat_surface_state,
            bottom_state = heat_bottom_state,
        ),
        neumann_bc = Neumann(surface_flux = nothing, bottom_flux = nothing),
    )

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil!,
    )

    N_poly = 5
    nelem_vert = 10

    # Specify the domain boundaries
    zmax = FT(1)
    zmin = FT(0)

    driver_config = ClimateMachine.SingleStackConfiguration(
        "LandModel",
        N_poly,
        nelem_vert,
        zmax,
        param_set,
        m;
        zmin = zmin,
        numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
    )

    t0 = FT(0)
    timeend = FT(2)
    dt = FT(1e-4)
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    mygrid = solver_config.dg.grid
    aux = solver_config.dg.state_auxiliary
    ClimateMachine.invoke!(solver_config)
    t = ODESolvers.gettime(solver_config.solver)

    z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
    z = Array(aux[:, z_ind, :][:])

    T_ind = varsindex(vars_state(m, Auxiliary(), FT), :soil, :heat, :T)
    T = Array(aux[:, T_ind, :][:])

    num =
        exp.(sqrt(ω / 2) * (1 + im) * (1 .- z)) .-
        exp.(-sqrt(ω / 2) * (1 + im) * (1 .- z))
    denom = exp(sqrt(ω / 2) * (1 + im)) - exp.(-sqrt(ω / 2) * (1 + im))
    analytic_soln = real(num .* A * exp(im * ω * timeend) / denom)
    MSE = mean((analytic_soln .- T) .^ 2.0)
    @test eltype(aux) == FT
    @test MSE < 1e-5
end
