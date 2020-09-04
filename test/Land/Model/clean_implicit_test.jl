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
using ForwardDiff

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


using IterativeSolvers
function GMRES(;mainkwargs...)
    function (::Type{Val{:init}}, _, _)
        (x,A,b,matrix_update=false; kwargs...) -> gmres!(x,A,b; mainkwargs..., kwargs...)
    end
end

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
FT = ForwardDiff.Dual{Nothing,Float64,1}

# in the full model, we couple heat and water equations. but right now we are just
# testing water, the one that has the numerical issues, and setting T = constant for all time
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
    linsolve = GMRES(; verbose = true),
))

t0 = FT(0)
timeend = FT(60)

## Time solver choices
use_implicit_solver = true
if use_implicit_solver
    given_Fourier = FT(9e-2)# This is about where it breaks right now, 3e-2 works.

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        ode_solver_type = ode_solver_type,
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
else
    given_Fourier = FT(9e-2) # this is where the explicit solver breaks too; domain errors

    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config;
        Courant_number = given_Fourier,
        CFL_direction = VerticalDirection(),
    )
end;

using ForwardDiff

Q = solver_config.Q   # state variable
dQ = similar(Q)       # result of f
dg = solver_config.dg # f
t = solver_config.t0
dg(dQ, Q, nothing, t) # compute f(Q) and store in dQ

D = ForwardDiff.Dual{Nothing,Float64,1}
R = similar(Q, Array, D)
R .= Q
dR = similar(R)
dg(dR,R, nothing, t)

# dQ/dt = f(Q,t)
# explicit Euler
#   Q_t+1 = Q_t + δt * f(Q_t)
# implicit Euler
#   Q_t+1 = Q_t + δt * f(Q_t+1)
#   need to solve:
#      Q_t+1 - δt * f(Q_t+1) = Q_t 
#   if f(Q) = LQ
#      (I - δt L) Q_t+1 = Q_t

# f : ℝⁿ → ℝⁿ
# ∇f : ℝⁿ → ℝⁿˣⁿ (Jacobian)
# ∇f ⋅ v = d/dϵ f(Q + ϵv) (Jacobian action / Gateaux derivative)

# e.g. if f(x) = Lx => ∇f(x) = L => ∇f(x) ⋅v = Lv


# how to do this?
#  - symbolic : what is the expression that gives the Jacobian
#  - numeric/finite differencing :
#     for each i = 1:n
#       (f(Q + ϵ_i v) - f(Q)) / ϵ where ϵ_i[j] = ϵ if i==j ; 0 otherwise
#     need to evaluate f n+1 times
#  - automatic differentiation
#     forward vs reverse
#     d = x + ϵ * y
#     i * i = -1
#     ϵ * ϵ =  0

# https://github.com/JuliaDiff/ForwardDiff.jl/issues/319#issuecomment-685006123

#Do the integration
# @time ClimateMachine.invoke!(solver_config)
