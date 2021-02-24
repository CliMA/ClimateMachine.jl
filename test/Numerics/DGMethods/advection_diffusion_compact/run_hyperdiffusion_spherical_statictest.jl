using MPI
using ClimateMachine
using Logging
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.ODESolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.Mesh.Grids: min_node_distance
using ClimateMachine.Atmos: SphericalOrientation, latitude, longitude

using ClimateMachine.Orientations
using CLIMAParameters
using CLIMAParameters.Planet
using GLMakie

using ClimateMachine.BalanceLaws:
    BalanceLaw,
    Prognostic,
    Auxiliary,
    Gradient,
    GradientFlux,
    GradientHyperFlux,
    GradientLaplacian,
    Hyperdiffusive

# import and update Earth parameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 60e3

include("hyperdiffusion_model.jl")
include("run_function.jl")
include("initial_condition.jl")
include("spherical_harmonics_kernels.jl")
include("output_writer.jl")
include("makie_gui.jl")

# Initialise CM
ClimateMachine.init()
ArrayType = ClimateMachine.array_type()
mpicomm = MPI.COMM_WORLD

output_dir = "output_hyper_EveryDir_LSRKEuler_projection_D1_diffdir_hack_ultimate_all_vert_kernels_1dt_Dfncx_tstdel_comment_all_indiv_knls"

# Set global model parameters
direction = EveryDirection      # direction model kernels 
diff_direction = EveryDirection # direction for diffusion kernels
dim = 3                         # no dimensions
_a = planet_radius(param_set)   # planet radius (m)

height = 30.0e3 #_a * 0.01      # domain height (m)

FT = Float64 

polynomialorder = 5

horz_num_elem = 8
vert_num_elem = 4
τ = 1                            # timescale for hyperdiffusion

topl = StackedCubedSphereTopology(
    mpicomm,
    horz_num_elem,
    grid1d(_a, _a + height, nelem = vert_num_elem)
)

grid = DiscontinuousSpectralElementGrid(
    topl,
    FloatType = FT,
    DeviceArray = ArrayType,
    polynomialorder = polynomialorder,
    meshwarp = ClimateMachine.Mesh.Topologies.cubedshellwarp,
)

# run model
dg, model, rhs_DGsource, rhs_anal, rel_error = run(mpicomm, ArrayType, dim, topl, grid, polynomialorder, FT, direction, diff_direction, τ*3600, 7, 4 )

# get vtk output- plot with paraview
do_output_vtk(mpicomm, output_dir, dg, rhs_DGsource, rhs_anal, model)

# plot output with makie

#makie_gui(topl, grid, rhs_DGsource, rhs_anal)# this wont work until this spherical topology is supported

nothing



