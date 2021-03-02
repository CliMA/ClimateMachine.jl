# Test that Richard's equation agrees with solution from Bonan's book,
# simulation 8.2
using MPI
using OrderedCollections
using StaticArrays
using Statistics
using Dierckx
using Test
using Pkg.Artifacts
using DelimitedFiles

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
using ClimateMachine.SystemSolvers
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.SingleStackUtils
using ClimateMachine.BalanceLaws:
    BalanceLaw, Prognostic, Auxiliary, Gradient, GradientFlux, vars_state
using ClimateMachine.ArtifactWrappers

ClimateMachine.init()
FT = Float64

function init_soil_water!(land, state, aux, localgeo, time)
    state.soil.water.ϑ_l = eltype(aux)(land.soil.water.initialϑ_l(aux))
    state.soil.water.θ_i = eltype(aux)(land.soil.water.initialθ_i(aux))
end

soil_heat_model = PrescribedTemperatureModel()

soil_param_functions = SoilParamFunctions{FT}(
    porosity = 0.4,
    Ksat = 6.94e-6/ (60), #m/s
    S_s = 5e-4,
)
# Keep in mind that what is passed is aux⁻
# Fluxes are multiplied by ẑ (normal to the surface, -normal to the bottom,
# where normal point outs of the domain.)
sigmoid(x, offset, width) = typeof(x)(exp((x-offset)/width)/(1+exp((x-offset)/width)))
bottom_flux_value = FT(0.0)
surface_value = FT(soil_param_functions.porosity)

surface_state = (aux, t) -> surface_value
bottom_flux = (aux, t) -> bottom_flux_value
ϑ_l0 = (aux) -> eltype(aux)(0.4- 0.025 * sigmoid(aux.z, -1.0,0.02))

bc = GeneralBoundaryConditions(
    Dirichlet(surface_state = surface_state, bottom_state = nothing),
    Neumann(surface_flux = nothing, bottom_flux = bottom_flux),
)
vg_a = FT(100.0)
vg_n = FT(2.0)
soil_water_model = SoilWaterModel(
    FT;
    moisture_factor = MoistureDependent{FT}(),
    hydraulics = vanGenuchten{FT}(n = vg_n, α = vg_a),
    initialϑ_l = ϑ_l0,
    boundaries = bc,
)

m_soil = SoilModel(soil_param_functions, soil_water_model, soil_heat_model)
sources = ()
m = LandModel(
    param_set,
    m_soil;
    source = sources,
    init_state_prognostic = init_soil_water!,
)

N_poly = 1
nelem_vert = 50

# Specify the domain boundaries
zmax = FT(0)
zmin = FT(-2)

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

include(joinpath(@__DIR__,"Benchmarks/benchmarksetup.jl"))

# linearsolver = BatchedGeneralizedMinimalResidual(
#     vdg,
#     Q;
#     max_subspace_size = 30,
#     atol = -1.0,
#     rtol = 1e-9,
# )
# nonlinearsolver =
#     JacobianFreeNewtonKrylovSolver(Q, linearsolver; tol = 1e-9)

# JacobianFreeNewtonKrylovSolver(
#     Q,
#     BatchedGeneralizedMinimalResidual(
#         vdg,
#         Q;
#         max_subspace_size = 30,
#         atol = 1e-9,
#         rtol = 1e-9,
#     );
#     # GeneralizedMinimalResidual(Q; M = 30, atol = 1e-6, rtol = 1e-6);
#     M = Int(1e3),
#     tol = 1e-9,
# )

# nlalg = JacobianFreeNewtonKrylovAlgorithm(
#     GeneralizedMinimalResidualAlgorithm(;
#         preconditioner = ColumnwiseLUPreconditioningAlgorithm(; update_period = 50),
#         M = 50,
#         atol = 1e-6,
#         rtol = 1e-6,
#     );
#     maxiters = 1000,
#     atol = 1e-6,
#     rtol = 1e-6,
#     autodiff = false,
# )

nlalg = AndersonAccelerationAlgorithm(
    StandardPicardAlgorithm(; maxiters = 3000, atol = 1e-6, rtol = 1e-6);
    depth = 6, ω = 0.5)
t0 = FT(0)
timeend = FT(60*60)
dts = FT[100]
c = callcount(nlalg, driver_config, t0, timeend, dts)
@elapsed ClimateMachine.invoke!(setup_solver(nlalg, driver_config, t0, timeend, dts[1]))
solvetime = @elapsed ClimateMachine.invoke!(setup_solver(nlalg, driver_config, t0, timeend, dts[1]))

# @save joinpath(@__DIR__,"infiltration_truth1.jld2") truedata = dons_arr[2]
# the_truth_interp_function = #
# the_truth = the_truth_interp_function(current_z)
# rmse = sqrt.((the_truth .- current_profile).^2.0)
# check rmse < 1e-4


# as long as the solution is "good" rmse < 1e-4, what is the tradeoff
#between time to solution (N steps will vary)
# or # of f calls