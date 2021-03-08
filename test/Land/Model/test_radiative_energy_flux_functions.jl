# Test functions used in runoff modeling.
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.RadiativeEnergyFlux
using ClimateMachine.Land.SoilWaterParameterizations
using ClimateMachine.Land.SoilHeatParameterizations
using ClimateMachine.Land.Runoff
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

@testset "Radiative energy flux testing" begin
    F = Float32
    user_nswf = t -> F(2 * t)
    user_swf = t -> F(2 * t)
    user_α = t -> F(0.2 * t)
    prescribed_swf_and_a =
        PrescribedSwFluxAndAlbedo(F; α = user_α, swf = user_swf)
    prescribed_nswf = PrescribedNetSwFlux(F; nswf = user_nswf)
    flux_from_prescribed_swf_and_albedo =
        compute_net_radiative_energy_flux.(
            Ref(prescribed_swf_and_a),
            [1, 2, 3, 4],
        )

    flux_from_prescribed_nswf =
        compute_net_radiative_energy_flux.(Ref(prescribed_nswf), [1, 2, 3, 4])

    @test flux_from_prescribed_swf_and_albedo ≈
          F.(([0.8, 0.6, 0.4, 0.2]) .* ([2, 4, 6, 8]))
    @test eltype(flux_from_prescribed_swf_and_albedo) == F

    @test flux_from_prescribed_nswf ≈ F.([2, 4, 6, 8])
    @test eltype(flux_from_prescribed_nswf) == F
end

@testset "Heat analytic unit test" begin
    ClimateMachine.init()

    FT = Float64

    function init_soil!(land, state, aux, localgeo, time)
        myfloat = eltype(state)
        ϑ_l, θ_i = get_water_content(land.soil.water, aux, state, time)
        θ_l =
            volumetric_liquid_fraction(ϑ_l, land.soil.param_functions.porosity)
        ρc_s = volumetric_heat_capacity(
            θ_l,
            θ_i,
            land.soil.param_functions.ρc_ds,
            land.param_set,
        )

        state.soil.heat.ρe_int = myfloat(volumetric_internal_energy(
            θ_i,
            ρc_s,
            land.soil.heat.initialT(aux),
            land.param_set,
        ))
    end


    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.4,
        ν_ss_gravel = 0.2,
        ν_ss_om = 0.2,
        ν_ss_quartz = 0.2,
        ρc_ds = 1.0, # diffusivity = k/ρc. When no water, ρc = ρc_ds.
        κ_solid = 1.0,
        ρp = 1.0,
        κ_sat_unfrozen = 0.57,
        κ_sat_frozen = 2.29,
    )

    #  Prescribed net short wave flux, initial temperature
    A = FT(5)
    prescribed_nswf = t -> FT(-A * t)
    T_init = FT(300.0)
    T_init_func = aux -> T_init

    # Flux entering = flux leaving soil column
    bc = LandDomainBC(
        bottom_bc = LandComponentBC(
            soil_heat = Neumann((aux, t) -> FT(-A * t)),
        ),
        surface_bc = LandComponentBC(
            soil_heat = SurfaceDrivenHeatBoundaryConditions(
                FT;
                nswf_model = PrescribedNetSwFlux(FT; nswf = prescribed_nswf),
            ),
        ),
    )

    soil_water_model = PrescribedWaterModel()
    soil_heat_model = SoilHeatModel(FT; initialT = T_init_func)

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        boundary_conditions = bc,
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
    timeend = FT(1)
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

    k = k_dry(param_set, soil_param_functions)
    diffusivity = k / soil_param_functions.ρc_ds

    A = A / k # Because the flux is applied on κ∇T. κ∇T = A*t in clima, whereas ∇T = A*t in the analytic soln.
    approx_sum_term = sum(
        (
            -2 * A / (diffusivity * pi^4) *
            cos.(n * pi * z) *
            (pi * n * sin(pi * n) + cos(pi * n) - 1) *
            (1 - exp(-timeend * diffusivity * (pi * n)^2)) / n^4
        ) for n in 1:100
    )

    approx_analytic_soln =
        A * timeend .* z .+ T_init .- A * timeend / 2 .+ approx_sum_term

    MSE = mean((approx_analytic_soln .- T) .^ 2.0)

    @test eltype(aux) == FT
    @test MSE < 1e-5
end
