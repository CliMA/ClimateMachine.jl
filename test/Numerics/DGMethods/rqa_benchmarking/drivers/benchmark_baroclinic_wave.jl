using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies: StackedCubedSphereTopology, grid1d, equiangular_cubed_sphere_warp
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.Atmos: AtmosFilterPerturbations
using ClimateMachine.DGMethods: ESDGModel, init_ode_state, courant
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics: soundspeed_air
using ClimateMachine.TemperatureProfiles
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cv_d, Omega, planet_radius, MSLP
import CLIMAParameters

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
using CUDA

const output_vtk = false

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();
#const X = 20
const X = 1
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371e6 / X
CLIMAParameters.Planet.Omega(::EarthParameterSet) = 7.2921159e-5 * X

# No this isn't great but w/e 
include("../interface/grid/domains.jl")
include("../interface/grid/grids.jl")
include("../interface/esdg_balance_law_interface.jl")
include("../interface/esdg_linear_balance_law_interface.jl")
include("../interface/simulations.jl")
include("../interface/utilities/callbacks.jl")
include("../interface/utilities/sphere_utils.jl")
include("../interface/models.jl")

ClimateMachine.init()

function sphr_to_cart_vec(vec, lat, lon)
    FT = eltype(vec)
    slat, clat = sin(lat), cos(lat)
    slon, clon = sin(lon), cos(lon)
    u = MVector{3, FT}(
        -slon * vec[1] - slat * clon * vec[2] + clat * clon * vec[3],
        clon * vec[1] - slat * slon * vec[2] + clat * slon * vec[3],
        clat * vec[2] + slat * vec[3],
    )
    return u
end

function init_state_prognostic!(
    bl::DryAtmosModel,
    state,
    aux,
    localgeo,
    t,
)
    coords = localgeo.coord
    FT = eltype(state)

    # parameters
    _grav::FT = grav(param_set)
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _Ω::FT = Omega(param_set)
    _a::FT = planet_radius(param_set)
    _p_0::FT = MSLP(param_set)

    k::FT = 3
    T_E::FT = 310
    T_P::FT = 240
    T_0::FT = 0.5 * (T_E + T_P)
    Γ::FT = 0.005
    A::FT = 1 / Γ
    B::FT = (T_0 - T_P) / T_0 / T_P
    C::FT = 0.5 * (k + 2) * (T_E - T_P) / T_E / T_P
    b::FT = 2
    H::FT = _R_d * T_0 / _grav
    z_t::FT = 15e3
    λ_c::FT = π / 9
    φ_c::FT = 2 * π / 9
    d_0::FT = _a / 6
    V_p::FT = 1
    M_v::FT = 0.608
    p_w::FT = 34e3             ## Pressure width parameter for specific humidity
    η_crit::FT = 10 * _p_0 / p_w ## Critical pressure coordinate
    q_0::FT = 0                ## Maximum specific humidity (default: 0.018)
    q_t::FT = 1e-12            ## Specific humidity above artificial tropopause
    φ_w::FT = 2π / 9           ## Specific humidity latitude wind parameter

    # grid
    λ = @inbounds atan(coords[2], coords[1])
    φ = @inbounds asin(coords[3] / norm(coords, 2))
    z = norm(coords) - _a

    r::FT = z + _a
    γ::FT = 1 # set to 0 for shallow-atmosphere case and to 1 for deep atmosphere case

    # convenience functions for temperature and pressure
    τ_z_1::FT = exp(Γ * z / T_0)
    τ_z_2::FT = 1 - 2 * (z / b / H)^2
    τ_z_3::FT = exp(-(z / b / H)^2)
    τ_1::FT = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2::FT = C * τ_z_2 * τ_z_3
    τ_int_1::FT = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2::FT = C * z * τ_z_3
    I_T::FT =
        (cos(φ) * (1 + γ * z / _a))^k -
        k / (k + 2) * (cos(φ) * (1 + γ * z / _a))^(k + 2)

    # base state virtual temperature, pressure, specific humidity, density
    T_v::FT = (τ_1 - τ_2 * I_T)^(-1)
    p::FT = _p_0 * exp(-_grav / _R_d * (τ_int_1 - τ_int_2 * I_T))

    # base state velocity
    U::FT =
        _grav * k / _a *
        τ_int_2 *
        T_v *
        (
            (cos(φ) * (1 + γ * z / _a))^(k - 1) -
            (cos(φ) * (1 + γ * z / _a))^(k + 1)
        )
    u_ref::FT =
        -_Ω * (_a + γ * z) * cos(φ) +
        sqrt((_Ω * (_a + γ * z) * cos(φ))^2 + (_a + γ * z) * cos(φ) * U)
    v_ref::FT = 0
    w_ref::FT = 0

    # velocity perturbations
    F_z::FT = 1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3
    if z > z_t
        F_z = FT(0)
    end
    d::FT = _a * acos(sin(φ) * sin(φ_c) + cos(φ) * cos(φ_c) * cos(λ - λ_c))
    c3::FT = cos(π * d / 2 / d_0)^3
    s1::FT = sin(π * d / 2 / d_0)
    if 0 < d < d_0 && d != FT(_a * π)
        u′::FT =
            -16 * V_p / 3 / sqrt(3) *
            F_z *
            c3 *
            s1 *
            (-sin(φ_c) * cos(φ) + cos(φ_c) * sin(φ) * cos(λ - λ_c)) /
            sin(d / _a)
        v′::FT =
            16 * V_p / 3 / sqrt(3) * F_z * c3 * s1 * cos(φ_c) * sin(λ - λ_c) /
            sin(d / _a)
    else
        u′ = FT(0)
        v′ = FT(0)
    end
    w′::FT = 0
    u_sphere = SVector{3, FT}(u_ref + u′, v_ref + v′, w_ref + w′)
    #u_sphere = SVector{3, FT}(u_ref, v_ref, w_ref)
    u_cart = sphr_to_cart_vec(u_sphere, φ, λ)

    ## temperature & density
    T::FT = T_v
    ρ::FT = p / (_R_d * T)
    ## potential & kinetic energy
    e_pot = aux.Φ
    e_kin::FT = 0.5 * u_cart' * u_cart
    e_int = _cv_d * T

    ## Assign state variables
    state.ρ = ρ
    state.ρu = ρ * u_cart
    if total_energy
        state.ρe = ρ * (e_int + e_kin + e_pot)
    else
        state.ρe = ρ * (e_int + e_kin)
    end

    nothing
end

function run(
    discretized_domain,
    model,
    timeend,
    FT,
)
    # 1 grid stuff
    domain_height = FT(30e3)
    grid = discretized_domain.numerical
    # end of grid stuff

    #2  initial conditions
    # end of initial conditions

    #3 physics stuff
    # end of physics stuff
    
    #4 set up balance law
    # end of balance law

    #5 set up dg models
    esdg = ESDGModel(
        model,
        grid,
        volume_numerical_flux_first_order = KGVolumeFlux(),
        surface_numerical_flux_first_order = RusanovNumericalFlux(),
    )

    #6 set up linear model
    linearmodel = DryAtmosAcousticGravityLinearModel(model)
    lineardg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = esdg.state_auxiliary,
    )
    # end of dg models

    # determine the time step construction
    # element_size = (domain_height / numelem_vert)
    # acoustic_speed = soundspeed_air(param_set, FT(330))

    # dx = min_node_distance(grid)
    # cfl = 3
    # dt = cfl * dx / acoustic_speed
    dt = 10.0

    Q = init_ode_state(esdg, FT(0))

    linearsolver = ManyColumnLU()
    odesolver = ARK2GiraldoKellyConstantinescu(
        esdg,
        lineardg,
        LinearBackwardEulerSolver(linearsolver; isadjustable = false),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
    )
    #end of time step construction

    # run it
    solve!(
        Q,
        odesolver;
        timeend = timeend,
        callbacks = (),
        adjustfinalstep = false,
    )
end

# Boilerplate

########
# Set up parameters
########
parameters = (
    a    = 6.371229e6,
    Ω    = 7.292e-5,
    g    = 9.80616,
    H    = 30e3,
    R_d  = 287.0,
    pₒ   = 1.0e5,
    k    = 3.0,
    Γ    = 0.005,
    T_E  = 310.0,
    T_P  = 240.0,
    b    = 2.0,
    z_t  = 15e3,
    λ_c  = π / 9,
    ϕ_c  = 2 * π / 9,
    V_p  = 1.0,
    κ    = 2/7,
)
FT = Float64

########
# Set up domain
########
domain = SphericalShell(
    radius = planet_radius(param_set), 
    height = 30e3,
)
grid = DiscretizedDomain(
    domain;
    elements = (vertical = 5, horizontal = 8),
    polynomial_order = 3,
    overintegration_order = (vertical = 0, horizontal = 0),
)

########
# Set up inital condition
########
# additional initial condition parameters
T_0(𝒫)  = 0.5 * (𝒫.T_E + 𝒫.T_P) 
A(𝒫)    = 1.0 / 𝒫.Γ
B(𝒫)    = (T_0(𝒫) - 𝒫.T_P) / T_0(𝒫) / 𝒫.T_P
C(𝒫)    = 0.5 * (𝒫.k + 2) * (𝒫.T_E - 𝒫.T_P) / 𝒫.T_E / 𝒫.T_P
H(𝒫)    = 𝒫.R_d * T_0(𝒫) / 𝒫.g
d_0(𝒫)  = 𝒫.a / 6

# convenience functions that only depend on height
τ_z_1(𝒫,r)   = exp(𝒫.Γ * (r - 𝒫.a) / T_0(𝒫))
τ_z_2(𝒫,r)   = 1 - 2 * ((r - 𝒫.a) / 𝒫.b / H(𝒫))^2
τ_z_3(𝒫,r)   = exp(-((r - 𝒫.a) / 𝒫.b / H(𝒫))^2)
τ_1(𝒫,r)     = 1 / T_0(𝒫) * τ_z_1(𝒫,r) + B(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_2(𝒫,r)     = C(𝒫) * τ_z_2(𝒫,r) * τ_z_3(𝒫,r)
τ_int_1(𝒫,r) = A(𝒫) * (τ_z_1(𝒫,r) - 1) + B(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
τ_int_2(𝒫,r) = C(𝒫) * (r - 𝒫.a) * τ_z_3(𝒫,r)
F_z(𝒫,r)     = (1 - 3 * ((r - 𝒫.a) / 𝒫.z_t)^2 + 2 * ((r - 𝒫.a) / 𝒫.z_t)^3) * ((r - 𝒫.a) ≤ 𝒫.z_t)

# convenience functions that only depend on longitude and latitude
d(𝒫,λ,ϕ)     = 𝒫.a * acos(sin(ϕ) * sin(𝒫.ϕ_c) + cos(ϕ) * cos(𝒫.ϕ_c) * cos(λ - 𝒫.λ_c))
c3(𝒫,λ,ϕ)    = cos(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))^3
s1(𝒫,λ,ϕ)    = sin(π * d(𝒫,λ,ϕ) / 2 / d_0(𝒫))
cond(𝒫,λ,ϕ)  = (0 < d(𝒫,λ,ϕ) < d_0(𝒫)) * (d(𝒫,λ,ϕ) != 𝒫.a * π)

# base-state thermodynamic variables
I_T(𝒫,ϕ,r)   = (cos(ϕ) * r / 𝒫.a)^𝒫.k - 𝒫.k / (𝒫.k + 2) * (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 2)
T(𝒫,ϕ,r)     = (τ_1(𝒫,r) - τ_2(𝒫,r) * I_T(𝒫,ϕ,r))^(-1) * (𝒫.a/r)^2
p(𝒫,ϕ,r)     = 𝒫.pₒ * exp(-𝒫.g / 𝒫.R_d * (τ_int_1(𝒫,r) - τ_int_2(𝒫,r) * I_T(𝒫,ϕ,r)))
θ(𝒫,ϕ,r)     = T(𝒫,ϕ,r) * (𝒫.pₒ / p(𝒫,ϕ,r))^𝒫.κ

# base-state velocity variables
U(𝒫,ϕ,r)     = 𝒫.g * 𝒫.k / 𝒫.a * τ_int_2(𝒫,r) * T(𝒫,ϕ,r) * ((cos(ϕ) * r / 𝒫.a)^(𝒫.k - 1) - (cos(ϕ) * r / 𝒫.a)^(𝒫.k + 1))
u(𝒫,ϕ,r)  = -𝒫.Ω * r * cos(ϕ) + sqrt((𝒫.Ω * r * cos(ϕ))^2 + r * cos(ϕ) * U(𝒫,ϕ,r))
v(𝒫,ϕ,r)  = 0.0
w(𝒫,ϕ,r)  = 0.0

# velocity perturbations
δu(𝒫,λ,ϕ,r)  = -16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * (-sin(𝒫.ϕ_c) * cos(ϕ) + cos(𝒫.ϕ_c) * sin(ϕ) * cos(λ - 𝒫.λ_c)) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δv(𝒫,λ,ϕ,r)  = 16 * 𝒫.V_p / 3 / sqrt(3) * F_z(𝒫,r) * c3(𝒫,λ,ϕ) * s1(𝒫,λ,ϕ) * cos(𝒫.ϕ_c) * sin(λ - 𝒫.λ_c) / sin(d(𝒫,λ,ϕ) / 𝒫.a) * cond(𝒫,λ,ϕ)
δw(𝒫,λ,ϕ,r)  = 0.0

# CliMA prognostic variables
# compute the total energy
uˡᵒⁿ(𝒫,λ,ϕ,r)   = u(𝒫,ϕ,r) + δu(𝒫,λ,ϕ,r)
uˡᵃᵗ(𝒫,λ,ϕ,r)   = v(𝒫,ϕ,r) + δv(𝒫,λ,ϕ,r)
uʳᵃᵈ(𝒫,λ,ϕ,r)   = w(𝒫,ϕ,r) + δw(𝒫,λ,ϕ,r)

e_int(𝒫,λ,ϕ,r)  = (𝒫.R_d / 𝒫.κ - 𝒫.R_d) * T(𝒫,ϕ,r)
e_kin(𝒫,λ,ϕ,r)  = 0.5 * ( uˡᵒⁿ(𝒫,λ,ϕ,r)^2 + uˡᵃᵗ(𝒫,λ,ϕ,r)^2 + uʳᵃᵈ(𝒫,λ,ϕ,r) )
e_pot(𝒫,λ,ϕ,r)  = 𝒫.g * r

ρ₀(𝒫,λ,ϕ,r)    = p(𝒫,ϕ,r) / 𝒫.R_d / T(𝒫,ϕ,r)
ρuˡᵒⁿ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵒⁿ(𝒫,λ,ϕ,r)
ρuˡᵃᵗ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uˡᵃᵗ(𝒫,λ,ϕ,r)
ρuʳᵃᵈ(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * uʳᵃᵈ(𝒫,λ,ϕ,r)
if total_energy
    ρe(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * (e_int(𝒫,λ,ϕ,r) + e_kin(𝒫,λ,ϕ,r) + e_pot(𝒫,λ,ϕ,r))
else
    ρe(𝒫,λ,ϕ,r) = ρ₀(𝒫,λ,ϕ,r) * (e_int(𝒫,λ,ϕ,r) + e_kin(𝒫,λ,ϕ,r))
end

# Cartesian Representation (boiler plate really)
ρ₀ᶜᵃʳᵗ(𝒫, x...)  = ρ₀(𝒫, lon(x...), lat(x...), rad(x...))
ρu⃗₀ᶜᵃʳᵗ(𝒫, x...) = (   ρuʳᵃᵈ(𝒫, lon(x...), lat(x...), rad(x...)) * r̂(x...) 
                     + ρuˡᵃᵗ(𝒫, lon(x...), lat(x...), rad(x...)) * ϕ̂(x...)
                     + ρuˡᵒⁿ(𝒫, lon(x...), lat(x...), rad(x...)) * λ̂(x...) ) 
ρeᶜᵃʳᵗ(𝒫, x...) = ρe(𝒫, lon(x...), lat(x...), rad(x...))

########
# Set up model physics
########
if total_energy
    sources = (Coriolis(),)
else
    sources = (Coriolis(), Gravity())
end
T_profile =
    DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))

########
# Set up model
########
model = DryAtmosModel{FT}(
    SphericalOrientation(),
    initial_conditions = (ρ = ρ₀ᶜᵃʳᵗ, ρu = ρu⃗₀ᶜᵃʳᵗ, ρe = ρeᶜᵃʳᵗ),
    ref_state = DryReferenceState(T_profile),
    sources = sources,
    parameters = parameters,
)

########
# Set up time steppers
########
timeend = 10 * 24 * 3600

########
# Run the simulation
########
result = run(
    grid,
    model,
    timeend,
    FT,
)

nothing