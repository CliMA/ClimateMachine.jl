# ∂ϑ
# --  = -∇ ⋅[ K(ϑ)∇h ]
# ∂t

# where h = ψ(ϑ) + z

# Often, |ψ| >> z, and likewise for  |∂ψ/∂z| c.w. 1
# Limits: as soil -> dry, ψ -> -∞, ∂ψ/∂ϑ -> +∞, K -> 0, , ∂K/∂ϑ -> ∞
#         as soil -> saturated, ψ -> 0, ∂ψ/∂ϑ -> +∞, K -> Ksat, ∂K/∂ϑ -> constant
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test
using OrdinaryDiffEq
using LinearAlgebra: norm


using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using ClimateMachine
using ClimateMachine.Land
using ClimateMachine.Land.SoilWaterParameterizations
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
using ClimateMachine.Mesh.Filters
import ClimateMachine.DGMethods: calculate_dt

function calculate_dt(dg, model::LandModel, Q, Courant_number, t, direction)
    Δt = one(eltype(Q))
    CFL = DGMethods.courant(diffusive_courant, dg, model, Q, Δt, t, direction)
    println("CFL", CFL)
    return Courant_number / CFL
end

function diffusive_courant(
    land::LandModel,
    state::Vars,
    aux::Vars,
    diffusive::Vars,
    Δx,
    Δt,
    t,
    direction,
)
        myf = eltype(state)
        soil = land.soil
        water = land.soil.water
        ϑ_l = state.soil.water.ϑ_l
        T = get_temperature(land.soil.heat)
        S_l = effective_saturation(
            soil.param_functions.porosity,
            ϑ_l,
        )
        
        hydraulic_k = soil.param_functions.Ksat * hydraulic_conductivity(
                water.impedance_factor,
                water.viscosity_factor,
                water.moisture_factor,
                water.hydraulics,
                state.soil.water.θ_ice,
                soil.param_functions.porosity,
                T,
                S_l,
        )
        ψ = pressure_head(
            water.hydraulics,
            soil.param_functions.porosity,
            soil.param_functions.S_s,
            ϑ_l,
        )
        K∇h = hydraulic_k  * abs(ψ) / Δx
    return Δt * K∇h  / Δx
end


#@testset "Richard's equation - Haverkamp test" begin
    ClimateMachine.init()
    FT = Float64

    function init_soil_water!(land, state, aux, coordinates, time)
        myfloat = eltype(aux)
        state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
        state.soil.water.θ_ice = myfloat(land.soil.water.initialθ_ice(aux))
    end

    soil_heat_model = PrescribedTemperatureModel{FT}()

    soil_param_functions = SoilParamFunctions{FT}(
        porosity = 0.495,
        Ksat = 0.0443 / (3600 * 100),
        S_s = 1e-3,
    )
    # Mimics initial rainfall on drier soil.
    surface_value = FT(0.494)
    bottom_flux_multiplier = FT(1.0)
    initial_moisture = FT(0.24)

    surface_state = (aux, t) -> surface_value
    bottom_flux = (aux, t) -> aux.soil.water.K * bottom_flux_multiplier
    ϑ_l0 = (aux) -> initial_moisture

    soil_water_model = SoilWaterModel(
        FT;
        moisture_factor = MoistureDependent{FT}(),
        hydraulics = vanGenuchten{FT}(n = 2.0),
        initialϑ_l = ϑ_l0,
        dirichlet_bc = Dirichlet(
            surface_state = surface_state,
            bottom_state = nothing,
        ),
        neumann_bc = Neumann(surface_flux = nothing, bottom_flux = bottom_flux),
    )

    m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
    sources = ()
    m = LandModel(
        param_set,
        m_soil;
        source = sources,
        init_state_prognostic = init_soil_water!,
    )


    N_poly = 7
    nelem_vert = 10

    # Specify the domain boundaries
    zmax = FT(0)
    zmin = FT(-1)

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

    ode_solver_type = ImplicitSolverType(OrdinaryDiffEq.KenCarp4(
        autodiff = false,
        linsolve = LinSolveGMRES(tol = 1e-6),
    ))

    t0 = FT(0)
    timeend = FT(60)

    use_implicit_solver = true
    if use_implicit_solver
        given_Fourier = FT(1e-4)

        solver_config = ClimateMachine.SolverConfiguration(
            t0,
            timeend,
            driver_config;
            ode_solver_type = ode_solver_type,
            Courant_number = given_Fourier,
            CFL_direction = VerticalDirection(),
        )
    else
        given_Fourier = FT(1e-5)#this being so small - implies our courant calculation is wrong? 

        solver_config = ClimateMachine.SolverConfiguration(
            t0,
            timeend,
            driver_config;
            Courant_number = given_Fourier,#how is this functionally different from the below?
            CFL_direction = VerticalDirection(),
        )
    end;
# explicit dt
#    dt = FT(6)
#    solver_config = ClimateMachine.SolverConfiguration(
#        t0,
#        timeend,
#        driver_config,
#        ode_dt = dt,
#    )
    thegrid = solver_config.dg.grid
    Q = solver_config.Q
    aux = solver_config.dg.state_auxiliary


    # Set up user-defined callbacks
    filterorder = 64
    filter = ExponentialFilter(thegrid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            Q,
            ("soil.water.ϑ_l",),
            thegrid,
            filter,
        )
        nothing
    end

    ClimateMachine.invoke!(solver_config)#,user_callbacks = (cbfilter,),)
    ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
    ϑ_l = Array(Q[:, ϑ_l_ind, :][:])
    z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
    z = Array(aux[:, z_ind, :][:])

    #plot!(ϑ_l, z,label = "no filter, npoly = 10, nelem = 10")
