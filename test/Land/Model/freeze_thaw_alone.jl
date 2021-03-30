# Test that freeze thaw alone conserves water mass

using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
using CLIMAParameters.Planet: ρ_cloud_liq
using CLIMAParameters.Planet: ρ_cloud_ice

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

@testset "Freeze thaw alone" begin
    struct tmp_model <: BalanceLaw end
    struct tmp_param_set <: AbstractParameterSet end

    function get_grid_spacing(
        N_poly::Int64,
        nelem_vert::Int64,
        zmax::FT,
        zmin::FT,
    ) where {FT}
        test_config = ClimateMachine.SingleStackConfiguration(
            "TmpModel",
            N_poly,
            nelem_vert,
            zmax,
            tmp_param_set(),
            tmp_model();
            zmin = zmin,
        )

        Δ = min_node_distance(test_config.grid)
        return Δ
    end

    function init_soil_water!(land, state, aux, coordinates, time)
        ϑ_l = eltype(state)(land.soil.water.initialϑ_l(aux))
        θ_i = eltype(state)(land.soil.water.initialθ_i(aux))
        state.soil.water.ϑ_l = ϑ_l
        state.soil.water.θ_i = θ_i
        param_set = land.param_set

        θ_l =
            volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
        ρc_ds = land.soil.param_functions.ρc_ds
        ρc_s = volumetric_heat_capacity(θ_l, θ_i, ρc_ds, param_set)

        state.soil.heat.ρe_int = volumetric_internal_energy(
            θ_i,
            ρc_s,
            land.soil.heat.initialT(aux),
            param_set,
        )
    end

    FT = Float64
    ClimateMachine.init()

    N_poly = 1
    nelem_vert = 30
    zmax = FT(0)
    zmin = FT(-1)
    t0 = FT(0)
    timeend = FT(60 * 60 * 24)
    dt = FT(1800)

    Δ = get_grid_spacing(N_poly, nelem_vert, zmax, zmin)
    freeze_thaw_source = PhaseChange{FT}(Δz = Δ)
    ρp = FT(2700) # kg/m^3
    ρc_ds = FT(2e06) # J/m^3/K


    soil_param_functions = SoilParamFunctions{FT}(
        Ksat = 0.0,
        S_s = 1e-4,
        porosity = 0.75,
        ν_ss_gravel = 0.0,
        ν_ss_om = 0.0,
        ν_ss_quartz = 0.5,
        ρc_ds = ρc_ds,
        ρp = ρp,
        κ_solid = 1.0,
        κ_sat_unfrozen = 1.0,
        κ_sat_frozen = 1.0,
    )


    bottom_flux = (aux, t) -> eltype(aux)(0.0)
    surface_flux = (aux, t) -> eltype(aux)(0.0)
    bc = LandDomainBC(
        bottom_bc = LandComponentBC(
            soil_water = Neumann(bottom_flux),
            soil_heat = Dirichlet((aux, t) -> eltype(aux)(280)),
        ),
        surface_bc = LandComponentBC(
            soil_water = Neumann(surface_flux),
            soil_heat = Dirichlet((aux, t) -> eltype(aux)(290)),
        ),
    )
    ϑ_l0 = (aux) -> eltype(aux)(1e-10)
    θ_i0 = (aux) -> eltype(aux)(0.33)

    soil_water_model = SoilWaterModel(FT; initialϑ_l = ϑ_l0, initialθ_i = θ_i0)

    T_init = (aux) -> eltype(aux)(aux.z * 10 + 290)
    soil_heat_model = SoilHeatModel(FT; initialT = T_init)

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = (freeze_thaw_source,)
    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
        source = sources,
        init_state_prognostic = init_soil_water!,
    )

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



    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )
    state_types = (Prognostic(), Auxiliary())
    initial =
        Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
    ClimateMachine.invoke!(solver_config)
    final =
        Dict[dict_of_nodal_states(solver_config, state_types; interp = true)]
    m_init =
        ρ_cloud_liq(param_set) * sum(initial[1]["soil.water.ϑ_l"]) .+
        ρ_cloud_ice(param_set) * sum(initial[1]["soil.water.θ_i"])
    m_final =
        ρ_cloud_liq(param_set) * sum(final[1]["soil.water.ϑ_l"]) .+
        ρ_cloud_ice(param_set) * sum(final[1]["soil.water.θ_i"])

    @test abs(m_final - m_init) < 1e-10
end
