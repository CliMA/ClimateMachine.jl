# ∂ϑ
# --  = -∇ ⋅[ K(ϑ)∇h ]
# ∂t

# where h = ψ(ϑ) + z

# Often, |ψ| >> z, and likewise for  |∂ψ/∂z| c.w. 1
# Limits: as soil -> dry, ψ -> -∞, ∂ψ/∂ϑ -> +∞, K -> 0, , ∂K/∂ϑ -> ∞
#         as soil -> saturated, ψ -> 0, ∂ψ/∂ϑ -> +∞, K -> Ksat, ∂K/∂ϑ -> constant

# This current set up is effectively 1D (vertical; z), but there is a single element
# in (x,y) with (Npoly+1) nodal points in each horizontal direction
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Test
using OrdinaryDiffEq
using LinearAlgebra: norm
#using Plots

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

        ψ = pressure_head(
            water.hydraulics,
            soil.param_functions.porosity,
            soil.param_functions.S_s,
            ϑ_l,
        )
        K∇h = soil.param_functions.Ksat  * abs(ψ) / Δx
    return Δt * K∇h  / Δx
end

function init_soil_water!(land, state, aux, coordinates, time)
    myfloat = eltype(aux)
    state.soil.water.ϑ_l = myfloat(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_ice = myfloat(land.soil.water.initialθ_ice(aux))
end



ClimateMachine.init()
FT = Float64

soil_heat_model = PrescribedTemperatureModel{FT}()

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.287,
    Ksat = 34 / (3600 * 100),
    S_s = 1e-3,
)
# Mimics initial rainfall on drier soil.
surface_state = (aux, t) -> eltype(aux)(0.267)
bottom_flux = (aux, t) -> aux.soil.water.K * eltype(aux)(1.0)
ϑ_l0 = (aux) -> eltype(aux)(0.1)

soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = 3.96, α = 2.7, m = 1.0),
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


#Spatial resolution
N_poly = 3
nelem_vert = 10

# Specify the grid, domain, boundary specifications
mpicomm = MPI.COMM_WORLD
array_type = ClimateMachine.array_type()

periodicity = (true, true, false)
meshwarp = (x...) -> identity(x)
numerical_flux_first_order = CentralNumericalFluxFirstOrder()
numerical_flux_second_order = CentralNumericalFluxSecondOrder()
numerical_flux_gradient = CentralNumericalFluxGradient()
solver_type = ExplicitSolverType()
boundary = ((0, 0), (0, 0), (1, 2))

zmax = FT(0)
zmin = FT(-1)
stretch = SingleExponentialStretching{FT}(-1.0)
xmin, xmax = zero(FT), one(FT)
ymin, ymax = zero(FT), one(FT)
brickrange = (
    grid1d(xmin, xmax, nelem = 1),
    grid1d(ymin, ymax, nelem = 1),
    grid1d(zmin, zmax, stretch, nelem = nelem_vert),
)

topology = StackedBrickTopology(
    mpicomm,
    brickrange,
    periodicity = periodicity,
    boundary = boundary,
)

grid = DiscontinuousSpectralElementGrid(
    topology,
    FloatType = FT,
    DeviceArray = array_type,
    polynomialorder = N_poly,
    meshwarp = meshwarp,
)

driver_config = ClimateMachine.DriverConfiguration(
    ClimateMachine.SingleStackConfigType(),
    "LandModel",
    N_poly,
    FT,
    array_type,
    solver_type,
    param_set,
    m,
    mpicomm,
    grid,
    numerical_flux_first_order,
    numerical_flux_second_order,
    numerical_flux_gradient,
    ClimateMachine.SingleStackSpecificInfo(),
)


## Time solver choices
tol = FT(1)
ode_solver_type = ImplicitSolverType(OrdinaryDiffEq.KenCarp4(
    autodiff = false,
    linsolve = LinSolveGMRES(tol = tol),
))

t0 = FT(0)
timeend = FT(60)

use_implicit_solver = true
if use_implicit_solver
    given_Fourier = FT(9e-2)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        ode_solver_type = ode_solver_type,
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
else
    given_Fourier = FT(9e-2)

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
end;



#Do the integration
@time ClimateMachine.invoke!(solver_config)




# to plot

#Q = solver_config.Q
#aux = solver_config.dg.state_auxiliary
#ϑ_l_ind = varsindex(vars_state(m, Prognostic(), FT), :soil, :water, :ϑ_l)
#ϑ_l = Array(Q[:, ϑ_l_ind, :][:])
#z_ind = varsindex(vars_state(m, Auxiliary(), FT), :z)
#z = Array(aux[:, z_ind, :][:])

#    plot(ϑ_l, z)
