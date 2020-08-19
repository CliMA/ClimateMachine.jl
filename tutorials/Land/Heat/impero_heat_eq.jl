include("boiler_plate.jl")

using Impero

# ``
# \frac{∂ T}{∂ t} + ∇ ⋅ (F(α, T, t)) = 0
# ``

# where
#  - ``F(α, T, t) = -α ∇T`` is the second-order flux
###################

# TODO: Sort out field abstractions...
# Does `Field` _need_ to know about the mesh?

T   = Field(Dict("field_type" => Prognostic(), "name" = "T"))    # Prognostic quantity
α∇T = Field(Dict("field_type" => Auxiliary(), "name" = "α∇T"))   # Diagnostic/auxiliary quantity

pde_equation = [
    α∇T   == α * ∇(T), ## auxiliary equation
    ## RHS argument is gradient state
    ## LHS is gradient flux state
    ∂t(T) == ∇⋅(α∇T), ##  Actual PDE / "Balance law"
]

domain = ... # for LocalGeometry / meshes
pde_system = PDESystem(pde_equation,
                       domain;
                       initial_condition=...,
                       bcs=...,
                       metadata=...)
# vars_state = to_balance_law(model, pde_system, FT)

# Define the float type (`Float64` or `Float32`)
FT = Float64;
# Initialize ClimateMachine for CPU.
ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));


# Create an instance of the `HeatModel`:
model = HeatModel{FT, typeof(param_set)}(; param_set = param_set);
# # Spatial discretization
# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;
# Specify the number of vertical elements
nelem_vert = 10;
# Specify the domain height
zmax = FT(1);

# Establish a `ClimateMachine` single stack configuration
# this can be a from the domain and the pde_system
driver_config = ClimateMachine.SingleStackConfiguration(
    "HeatEquation",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    model,
    periodic = (true, true, true), # move out
    boundary = ((0,0), (0,0), (0,0)), # move out
    numerical_flux_second_order = CentralNumericalFluxSecondOrder(), # move out
);

# # Time discretization / solver

# Specify simulation time (SI units)
t0 = FT(0)
timeend = FT(40)

given_Fourier = FT(0.7)

solver_config = ClimateMachine.SolverConfiguration(
    t0,
    timeend,
    driver_config;
    Courant_number = given_Fourier,
    CFL_direction = VerticalDirection(),
)

# # Solve
ClimateMachine.invoke!(solver_config);
