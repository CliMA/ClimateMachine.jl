# # Eddy Diffusivity- Mass Flux test

# To put this in the form of ClimateMachine's [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw), we'll re-write the equation as:

# "tendency"       "second order flux"  "first order flux"    "non-conservative source"
# \frac{∂ F}{∂ t} =    - ∇ ⋅ ( F1 + F2 )                       + S

# where F1 is the flux-componenet that has no gradient term
# where F2 is the flux-componenet that has a  gradient term

# -------------- Subdomains:
# The model has a grid mean (the dycore stats vector),i=1:N updrafts and a single enviroment subdomain (subscript "0") 
# The grid mean is prognostic in first moment and diagnostic in second moment.
# The updrafts are prognostic in first moment and set to zero in second moment.
# The environment is diagnostic in first moment and prognostic in second moment.

# ## Equations solved:
# -------------- First Moment Equations:
#                grid mean
# ``
#     "tendency"           "second order flux"   "first order flux"                 "non-conservative source"
# \frac{∂ ρ}{∂ t}         =                         - ∇ ⋅ (ρu)
# \frac{∂ ρ u}{∂ t}       = - ∇ ⋅ (-ρaK ∇u0       ) - ∇ ⋅ (ρu u' - ρ*MF_{u} )         + S_{surface Friction}
# \frac{∂ ρ e_{int}}{∂ t} = - ∇ ⋅ (-ρaK ∇e_{int,0}) - ∇ ⋅ (u ρ_{int} - ρ*MF_{e_int} ) + S_{microphysics}
# \frac{∂ ρ q_{tot}}{∂ t} = - ∇ ⋅ (-ρaK ∇E_{tot,0}) - ∇ ⋅ (u ρ_{tot} - ρ*MF_{q_tot} ) + S_{microphysics}
# MF_ϕ = \sum{a_i * (w_i-w0)(ϕ_i-ϕ0)}_{i=1:N}
# K is the Eddy_Diffusivity, given as a function of enviromental variables
# ``

#                i'th updraft equations (no second order flux)
# ``
#     "tendency"                 "first order flux"    "non-conservative sources"
# \frac{∂ ρa_i}{∂ t}           = - ∇ ⋅ (ρu_i)         + (E_{i0}           - Δ_{i0})
# \frac{∂ ρa_i u_i}{∂ t}       = - ∇ ⋅ (ρu_i u_i')    + (E_{i0}*u_0       - Δ_{i0}*u_i)       + ↑*(ρa_i*b - a_i\frac{∂p^†}{∂z})
# \frac{∂ ρa_i e_{int,i}}{∂ t} = - ∇ ⋅ (ρu*e_{int,i}) + (E_{i0}*e_{int,0} - Δ_{i0}*e_{int,i}) + ρS_{int,i}
# \frac{∂ ρa_i q_{tot,i}}{∂ t} = - ∇ ⋅ (ρu*q_{tot,i}) + (E_{i0}*q_{tot,0} - Δ_{i0}*q_{tot,i}) + ρS_{tot,i}
# b = 0.01*(e_{int,i} - e_{int})/e_{int}
#
#                environment equations first moment
# ``
# a0 = 1-sum{a_i}{i=1:N}
# u0 = (1-sum{a_i*u_i}{i=1:N})/a0
# E_int0 = (1-sum{a_i*E_int_i}{i=1:N})/a0
# q_tot0 = (1-sum{a_i*q_tot_i}{i=1:N})/a0
#
#                environment equations second moment
# ``
#     "tendency"           "second order flux"       "first order flux"  "non-conservative source"
# \frac{∂ ρa_0ϕ'ψ'}{∂ t} =  - ∇ ⋅ (-ρa_0⋅K⋅∇ϕ'ψ')  - ∇ ⋅ (u ρa_0⋅ϕ'ψ')   + 2ρa_0⋅K(∂_z⋅ϕ)(∂_z⋅ψ)  + (E_{i0}*ϕ'ψ' - Δ_{i0}*ϕ'ψ') + ρa_0⋅D_{ϕ'ψ',0} + ρa_0⋅S_{ϕ'ψ',0}
# ``

# --------------------- Ideal gas law and subdomain density
# ``
# T_i, q_l  = saturation ajusment(e_int, q_tot)
# TempShamEquil(e_int,q_tot,p)
# ρ_i = <p>/R_{m,i} * T_i
# b = -g(ρ_i-ρ_h)<ρ>

# where
#  - `t`        is time
#  - `z`        is height
#  - `ρ`        is the density
#  - `u`        is the 3D velocity vector
#  - `e_int`    is the internal energy
#  - `q_tot`    is the total specific humidity
#  - `K`        is the eddy diffusivity
#  - `↑`        is the upwards pointing unit vector
#  - `b`        is the buoyancy
#  - `E_{i0}`   is the entrainment rate from the enviroment into i
#  - `Δ_{i0}`   is the detrainment rate from i to the enviroment 
#  - `ϕ'ψ'`     is a shorthand for \overline{ϕ'ψ'}_0 the enviromental covariance of ϕ and ψ
#  - `D`        is a covariance dissipation
#  - `S_{ϕ,i}`  is a source of ϕ in the i'th subdomain
#  - `∂_z`      is the vertical partial derivative 

# --------------------- Initial Conditions 
# Initial conditions are given for all variables in the grid mean, and subdomain variables assume thier grid mean values 
# ``
# ------- grid mean:
# ρ = hydrostatis reference state - need to compute that state 
# ρu = 0
# ρe_int = convert from input profiles 
# ρq_tot = convert from input profiles 
# ------- updrafts:
# ρa_i = 0.1/N
# ρau = 0
# ρae_int = a*gm.ρe_int
# ρaq_tot = a*gm.ρq_tot
# ------- environment:
# cld_frac = 0.0
# `ϕ'ψ'` = initial covariance profile
# TKE = initial TKE profile
# ``

# --------------------- Boundary Conditions 
#           grid mean 
# ``
# surface: ρ = 
# z_min: ρu = 0
# z_min: ρe_int = 300*cp ; cp=1000
# z_min: ρq_tot = 0.0
# ``

#           i'th updraft 
# ``
# z_min: ρ = 1
# z_min: ρu = 0
# z_min: ρe_int = 302*cp ; cp=1000
# z_min: ρq_tot = 0.0
# ``

# Solving these equations is broken down into the following steps:
# 1) Preliminary configuration
# 2) PDEs
# 3) Space discretization
# 4) Time discretization
# 5) Solver hooks / callbacks
# 6) Solve
# 7) Post-processing

# questions for Charlie
# 1. how do we implemnt an intial profile rathee than an initial value?
#            to this question I would ask how do we indetify the vertical level in which we are?
# 2. Can use Thermodynamic state to compute the reference profile of hydrostatic bakance and aidiabatic 
# 3. 

# # Preliminary configuration

# ## Loading code

# First, we'll load our pre-requisites
#  - load external packages:
using MPI
using Distributions
using NCDatasets
using OrderedCollections
using Plots
using StaticArrays

#  - load CLIMAParameters and set up to use it:

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

#  - load necessary ClimateMachine modules:
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.DGmethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates

import ClimateMachine.VariableTemplates: Vars, Grad
@inline function Base.getindex(v::Vars{NTuple{N,T},A,offset}, i) where {N,T,A,offset}
  # 1 <= i <= N
  array = parent(v)
  return Vars{T, A, offset + (i-1)*varsize(T)}(array)
end
@inline function Base.getindex(v::Grad{NTuple{N,T},A,offset}, i) where {N,T,A,offset}
  # 1 <= i <= N
  array = parent(v)
  return Grad{T, A, offset + (i-1)*varsize(T)}(array)
end


#  - import necessary ClimateMachine modules: (`import`ing enables us to
#  provide implementations of these structs/methods)
import ClimateMachine.DGmethods:
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    boundary_state!

# ## Initialization

# Define the float type (`Float64` or `Float32`)
FT = Float64;
# Initialize ClimateMachine for CPU.
ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
# Load some helper functions (soon to be incorporated into
# `ClimateMachine/src`)
include(joinpath(clima_dir, "tutorials", "Land", "helper_funcs.jl"));
include(joinpath(clima_dir, "tutorials", "Land", "plotting_funcs.jl"));

# # Define the set of Partial Differential Equations (PDEs)

# ## Define the model

# Model parameters can be stored in the particular [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw), in this case, a `SingleStack`:

Base.@kwdef struct Environment{FT} <: BalanceLaw
end

Base.@kwdef struct Updraft{FT} <: BalanceLaw
end

Base.@kwdef struct EntrainmentDetrainment{FT} <: BalanceLaw
end

Base.@kwdef struct EDMF{FT, N} <: BalanceLaw
    updraft::NTuple{N,Updraft{FT}} = ntuple(i->Updraft{FT}(), N)
    environment::Environment{FT} = Environment{FT}()
    entr_detr::EntrainmentDetrainment{FT} = EntrainmentDetrainment{FT}()
end


Base.@kwdef struct SingleStack{FT, N} <: BalanceLaw
    "Parameters"
    param_set::AbstractParameterSet = param_set
    "Entrainment factor"
    c_ε::FT = 0.13
    "Detrainment factor"
    c_δ::FT = 0.52
    "Trubulent Entrainment factor"
    c_t::FT = ?
    "Detrainment RH power"
    β::FT = 2
    "Logistic function scale"
    μ_0::FT = 0.0004
    "Updraft mixing fraction"
    χ::FT = 0.25
    "Entainmnet TKE scale"
    c_λ::FT = 0.3
    "Pressure drag"
    α_d::FT = 10.0
    "Pressure advection"
    α_a::FT = 0.1
    "Pressure buoyancy"
    α_b::FT = 0.12
    "Eddy Viscosity"
    c_m::FT = 0.14
    "Eddy Diffusivity"
    c_k::FT = 0.22
    "Static Stability coefficient"
    c_b::FT = 0.63
    "Von Karman constant"
    κ::FT = 0.4
    "Ratio of rms turbulent velocity to friction velocity"
    κ_star ::FT = 1.94
    "Empirical stability function coefficient"
    a1 ::FT = -100 
    "Empirical stability function coefficient"
    a2 ::FT = -0.2 
    "neutral Prandtl number"
    Pr_n::FT = 0.74 
    "fixed ustar" # YAIR - need to change this 
    ustar::FT = 0.28
    "Sufcae area"
    a_surf::FT = 0.1 
    "enviromental cloud fraction"
    cf_initial::FT = 0.0 # need to define a function for cf
    "Domain height"
    zmax::FT = 3000
    # "Initial density"
    # ρ_IC::FT = 1
    # "Initial conditions for temperature"
    # initialT::FT = 295.15
    "Surface specific humidity [kg/kg]"
    P_surf::FT = 0.0016
    "Surface pressure [pasc]"
    P_surf::FT = 101300.0
    "Surface internal energy []"
    surface_e_int::FT = 300.0*1004.0
    "Surface total specific humidity [kg/kg]"
    surface_q_tot::FT = 0.0
    "Surface I-flux [m^3/s^3]"
    e_int_surface_flux::FT = 0.0
    "Surface q_tot-flux [m/s*kg/kg]"
    q_tot_surface_flux::FT = 0.0
    "Top I-flux [m^3/s^3]"
    e_int_top_flux::FT = 0.0
    "Top q_tot-flux [m/s*kg/kg]"
    q_tot_top_flux::FT = 0.0
    # # add reference state
    # ref_state::RS = HydrostaticState(                   # quickly added at end
    #     LinearTemperatureProfile(                       # quickly added at end
    #         FT(200),                                    # quickly added at end
    #         FT(280),                                    # quickly added at end
    #         FT(grav(param_set)) / FT(cp_d(param_set)),  # quickly added at end
    #     ),                                              # quickly added at end
    #     FT(0),                                          # quickly added at end
    # ),                                                  # quickly added at end
    "EDMF scheme"
    edmf::EDMF{FT, N} = EDMF{FT, N}()
end

N = 2
# Create an instance of the `SingleStack`:
m = SingleStack{FT, N}();

# This model dictates the flow control, using [Dynamic Multiple
# Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), for which
# kernels are executed.

# ## Define the variables

# All of the methods defined in this section were `import`ed in # [Loading
# code](@ref) to let us provide implementations for our `SingleStack` as they
# will be used by the solver.

# Specify auxiliary variables for `SingleStack`

# vars_state_auxiliary(::Updraft, FT) = @vars(T::FT)
function vars_state_auxiliary(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_auxiliary(m[i], FT), N)...}
end

function vars_state_auxiliary(::Updraft, FT)
    @vars(buoyancy::FT,
          upd_top::FT,
          )
end

function vars_state_auxiliary(::Environment, FT)
    @vars(buoyancy::FT,
          K_eddy::FT,
          cld_frac::FT,
          )
end

function vars_state_auxiliary(m::EDMF, FT)
    @vars(environment::vars_state_auxiliary(m.environment, FT),
          updraft::vars_state_auxiliary(m.updraft, FT)
          );
end

function vars_state_auxiliary(m::SingleStack, FT)
    @vars(z::FT,
          buoyancy::FT,
          ρ_0::FT,
          p_0::FT,
          edmf::vars_state_auxiliary(m.edmf, FT),
          # ref_state::vars_state_auxiliary(m.ref_state, FT) # quickly added at end
          );
end

# Specify state variables, the variables solved for in the PDEs, for
# `SingleStack`
function vars_state_conservative(::Updraft, FT)
    @vars(ρa::FT,
          ρau::SVector{3, FT},
          ρae_int::FT,
          ρaq_tot::FT,
          )
end

function vars_state_conservative(::Environment, FT)
    @vars(ρatke::SVector{3, FT},
          ρae_int_cv::FT,
          ρaq_tot_cv::FT,
          ρae_int_q_tot_cv::FT,
          )
end

function vars_state_conservative(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_conservative(m[i], FT), N)...}
end


function vars_state_conservative(m::EDMF, FT)
    @vars(environment::vars_state_conservative(m.environment, FT),
          updraft::vars_state_conservative(m.updraft, FT)
          );
end

function vars_state_conservative(m::SingleStack, FT)
    @vars(ρ::FT,
          ρu::SVector{3, FT},
          ρe_int::FT,
          ρq_tot::FT,
          edmf::vars_state_conservative(m.edmf, FT));
end

function vars_state_gradient(::Updraft, FT)
    @vars(u::SVector{3, FT},
          e_int::FT,
          e_int::FT,
          )
end

function vars_state_gradient(::Environment, FT)
    @vars(e_int::FT,
          q_tot::FT,
          u::SVector{3, FT},
          tke::TF,
          e_int_cv::TF,
          q_tot_cv::TF,
          e_int_q_tot_cv::TF,
          )
end


function vars_state_gradient(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_gradient(m[i], FT), N)...}
end


function vars_state_gradient(m::EDMF, FT)
    @vars(environment::vars_state_gradient(m.environment, FT),
          updraft::vars_state_gradient(m.updraft, FT)
          );
end

function vars_state_gradient(m::SingleStack, FT)
    @vars(edmf::vars_state_gradient(m.edmf, FT));
end


function vars_state_gradient_flux(m::NTuple{N,Updraft}, FT) where {N}
    return Tuple{ntuple(i->vars_state_gradient_flux(m[i], FT), N)...}
end

vars_state_gradient_flux(::Updraft, FT) = @vars()

function vars_state_gradient_flux(::Environment, FT)
    @vars(∇e_int::SVector{3, FT},
          ∇q_tot::SVector{3, FT},
          ∇u::SMatrix{3, 3, FT, 9},
          ∇tke::TF,
          ∇e_int_cv::TF,
          ∇q_tot_cv::TF,
          ∇e_int_q_tot_cv::TF,
          ∇θ_ρ::TF, # used in a diagnostic equation for the mixing length 
          )
end

function vars_state_gradient_flux(m::EDMF, FT)
    @vars(environment::vars_state_gradient_flux(m.environment, FT),
          updraft::vars_state_gradient_flux(m.updraft, FT)
          );
end

function vars_state_gradient_flux(m::SingleStack, FT)
    @vars(edmf::vars_state_gradient_flux(m.edmf, FT));
end
# ## Define the compute kernels

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state_auxiliary`
function init_state_auxiliary!(m::SingleStack{FT,N}, aux::Vars, geom::LocalGeometry) where {FT,N}
    aux.z = geom.coord[3]
    aux.T = m.initialT

    # Compute the refernece profile ρ_0,p_0 to be stored in grid mean auxiliary vars
    #       consider a flag for SingleStack setting that assines gm.ρ = ρ_0; gm.p = p_0
    
    # status:
    # Need to find the right way to integrate the hydrostatic reference profile
    # to obtain both ρ_0 and p_0 by integrating log(p) in z based on dlog(p)/dz = -g/(R_m*T)
    # with constant θ_liq and q_tot
    # it is not clear to in function dynamically assine p in LiquidIcePotTempSHumEquil_given_pressure
    # at each level 


    # aux.ref_state.p = p_0
    # aux.ref_state.ρ = p_0 / (R_m * T)
    # R_m = R_m()


    # Alias convention:
    gm_a = aux
    en_a = aux.edmf.environment
    up_a = aux.edmf.updraft

    gm_a.buoyancy = 0.0
    gm_a.ρ_0 = hydrostatic_ref()  # yair 
    gm_a.p_0 = hydrostatic_ref()  # yair 
    
    en_a.buoyancy = 0.0
    en_a.cld_frac = 0.0
    
    for i in 1:N
        up_a[i].buoyancy = 0.0
        up_a[i].upd_top = 0.0
    end

    en_a.T = gm_a.T
    en_a.cld_frac = m.cf_initial
end;

# The following two functions should compute the hydrostatic and adiabatic reference state 
# this state integrates upwards the equations d(log(p))/dz = -g/(R_m*T) 
# with q_tot and θ_liq at each z level equal thier respective surface values. 

function integral_load_auxiliary_state!(
    m::SingleStack{FT,N},
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    # need to define thermo_state with set values of thetali and qt from surafce values (moist adiabatic)
    # something like _p = exp(input_logP) where input_logP is log of the p at the lower pressure  level
    ts = LiquidIcePotTempSHumEquil_given_pressure(param_set, m.θ_liq_flux_surf, _p, m.q_tot_flux_surf) # thermodynamic state
    q = PhasePartition(ts)
    T = air_temperature(ts)
    _R_m = gas_constant_air(param_set, q)
    integrand.a = -g / (Rm * T)
end

function integral_set_auxiliary_state!(
    m::SingleStack{FT,N},
    aux::Vars,
    integral::Vars,
)
    aux.int.a = integral.a + log(m.P_surf)
    aux.p_0 = exp(aux.int.a)
end

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρcT` is available here because we've specified `ρcT` in
# `vars_state_conservative`
function init_state_conservative!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
) where {FT,N}

    # Alias convention:
    gm_a = aux
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft

    # gm.ρ = aux.ref_state.ρ # quickly added at end
    
    # GCM setting - Initialize the grid mean profiles of prognostic variables (ρ,e_int,q_tot,u,v,w) 
    z = aux.z

    # SCM setting - need to have sepearete cases coded and called from a folder - see what LES does 
    # a moist_thermo state is used here to convert the input θ,q_tot to e_int, q_tot profile 
    ts = LiquidIcePotTempSHumEquil_given_pressure(param_set, θ_liq, P, q_tot)
    T = air_temperature(ts)
    ρ = air_density(ts)
    
    gm.ρ = 
    gm.ρu = 0.0
    gm.ρe_int = 0.0
    gm.ρq_tot = 0.0

    a_up = m.a_updraft_initial/FT(N)
    for i in 1:N
        up[i].ρa = ρ * a_up
        up[i].ρau = gm.ρu * a_up
        up[i].ρae_int = gm.ρe_int * a_up
        up[i].ρaq_tot = gm.ρq_tot * a_up
    end

    # initialize environment covariance 
    en.ρae_int_cv = 
    en.ρaq_tot_cv = 
    en.ρae_int_q_tot_cv = 

end;

# The remaining methods, defined in this section, are called at every
# time-step in the solver by the [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `single_stack_nodal_update_aux!`, or
# any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::SingleStack,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)   # Charlie - please check this call to the oveload func
    nodal_update_auxiliary_state!(single_stack_nodal_update_aux!, dg, m, Q, t, elems) 
    return true # TODO: remove return true
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in
# `vars_state_auxiliary`
function single_stack_nodal_update_aux!(
    m::SingleStack{FT,N},
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}

    gm_a = aux
    en_a = aux.edmf.environment
    up_a = aux.edmf.updraft
    gm = state
    en = state.edmf.environment
    up = state.edmf.updraft

    #   -------------  Compute buoyanies of subdomains 
    ρinv = 1/gm.ρ
    b_upds = 0.0
    a_upds = 0.0
    for i in 1:N
        # computing buoyancy with PhaseEquil (i.e. qv_star) that uses gm.ρ instead of ρ_i that is unkown
        ts = PhaseEquil(param_set ,up[i].e_int, gm.ρ, up[i].q_tot)

        ρ_i = air_density(ts)
        up_a[i].buoyancy = -grav*(ρ_i-aux.ref_state.ρ)*ρinv
        b_upds += up_a[i].buoyancy
        a_upds += up_a[i].ρa*ρinv
    end
    # compute the buoyancy of the environment
    env_e_int = (gm.e_int - up[i].e_int*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    env_q_tot = (gm.q_tot - up[i].q_tot*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    ts = PhaseEquil(param_set ,env_e_int, gm.ρ, env_q_tot)
    env_ρ = air_density(ts)
    env_q_liq = PhasePartition(ts).liq
    env_q_ice = PhasePartition(ts).ice
    b_env = -grav*(env_ρ-aux.ref_state.ρ)*ρinv
    # subtract the grid mean
    b_gm = (1 - a_ups)*b_env
    for i in 1:N
        b_gm += up_a[i].buoayncy*up_a[i].ρa*ρinv
    end
    up_a[i].buoyancy -= b_gm
    gm_a.buoyancy = 0

    #   -------------  Compute env cld_frac
    en_a.cld_frac = 0.0 # here a quadrature model shoiuld be called
    #   -------------  Compute upd_top
    for i in 1:N
        up_a[i].upd_top = 0.0 # ?? this is a difficult one as we need to find the maximum z in which up[i].a > minval
    end
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
# the gradient of `ρcT`. Here, we specify how `ρcT` is computed. Note that
#  - `transform.ρcT` is available here because we've specified `ρcT` in
#  `vars_state_gradient`
function compute_gradient_argument!(
    m::SingleStack{FT,N},
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT, N}
    # Alias convention:
    gm_t = transform
    up_t = transform.edmf.updraft
    en_t = transform.edmf.environment
    gm = state
    up = state.edmf.updraft
    en = state.edmf.environment

    ρ = gm.ρ
    env_u = (gm.u - up[i].u*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    env_e_int = (gm.e_int - up[i].e_int*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    env_q_tot = (gm.q_tot - up[i].q_tot*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
    en_t.u     = env_u
    en_t.q_tot = env_q_tot
    en_t.e_int = env_e_int

    ts = thermo_state(SingleStack, state, aux)
    en_t.θ_ρ = virtual_pottemp(ts)

    # for i in 1:N
    #     up_t[i].u = up[i].ρau/up[i].ρa
    # end
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
# `compute_gradient_argument!`. Note that:
#  - `diffusive.α∇ρcT` is available here because we've specified `α∇ρcT` in
#  `vars_state_gradient_flux`
#  - `∇transform.ρcT` is available here because we've specified `ρcT`  in
#  `vars_state_gradient`
function compute_gradient_flux!(
    m::SingleStack{FT,N},
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    # Alias convention:
    gm_d = diffusive
    up_d = diffusive.edmf.updraft
    en_d = diffusive.edmf.environment
    gm_∇t = ∇transform
    up_∇t = ∇transform.edmf.updraft
    en_∇t = ∇transform.edmf.environment
    gm = state
    up = state.edmf.updraft
    # gm_d.α∇ρcT = -m.α * gm_∇t.ρcT
    # gm_d.μ∇u = -m.μ * gm_∇t.u
    ρinv = 1/gm.ρ
    en_d.∇θ_ρ = en_∇t.θ_ρ

    for i in 1:N
        up_d[i].∇u = up_∇t[i].u
    end
end;

# We have no sources, nor non-diffusive fluxes.
function source!(
    m::SingleStack{FT, N},
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
) where {FT, N}

    # Alias convention:
    gm = state
    en = state
    up = state.edmf.updraft
    gm_s = source
    en_s = source
    up_s = source.edmf.updraft

    # should be conditioned on updraft_area > minval 
    a_env = 1-sum([up[i].ρa for i in 1:N])*ρinv
    ρinv = 1/gm.ρ
    for i in 1:N
        # get enviroment values for e_int, q_tot , u[3]
        env_u = (gm.u - up[i].u*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
        env_e_int = (gm.e_int - up[i].e_int*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)
        env_q_tot = (gm.q_tot - up[i].q_tot*up[i].ρa*ρinv)/(1-up[i].ρa*ρinv)

        # first moment sources 
        εt, ε, δ = entr_detr(m, m.edmf.entr_detr, state, diffusive, aux, t, direction, i)
        l = mixing_length(m, m.edmf.mix_len, source, state, diffusive, aux, t, direction, δ, εt)
        K_eddy = m.c_k*l*sqrt(en.tke)
        dpdz, dpdz_tke_i = perturbation_pressure(m, m.edmf.pressure, source, state, diffusive, aux, t, direction, i)

           # entrainment and detrainment 
        w_i = up[i].ρu[3]*ρinv
        up_s[i].ρa      += up[i].ρa * w_i * (ε - δ)
        up_s[i].ρau     += up[i].ρa * w_i * ((ε+εt)*up_s[i].ρau     - (δ+εt)*env_u)
        up_s[i].ρae_int += up[i].ρa * w_i * ((ε+εt)*up_s[i].ρae_int - (δ+εt)*env_e_int)
        up_s[i].ρaq_tot += up[i].ρa * w_i * ((ε+εt)*up_s[i].ρaq_tot - (δ+εt)*env_q_tot)

           # perturbation pressure
        up_s[i].ρau[3]  += up[i].ρa * dpdz

        # second moment sources 
        en.ρatke += dpdz_tke_i

        # sources  for the grid mean 
    end
end;

include(joinpath("closures", "entr_detr.jl"))
include(joinpath("closures", "pressure.jl"))
include(joinpath("closures", "mixing_lenfth.jl"))
include(joinpath("closures", "quadrature.jl"))

function flux_first_order!(
    m::SingleStack{FT,N},
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}

    # Alias convention:
    gm = state
    up = state.edmf.updraft
    gm_f = flux
    up_f = flux.edmf.updraft

    # gm
    ρinv = 1/gm.ρ
    gm_f.ρ = gm.ρu
    u = gm.ρu * ρinv
    gm_f.ρu = gm.ρu * u'
    gm_f.ρe_int = u * gm.ρe_int
    gm_f.ρq_tot = u * gm.ρq_tot

    # up
    for i in 1:N
        up_f[i].ρa = up[i].ρau
        u = up[i].ρau / up[i].ρa
        up_f[i].ρau = up[i].ρau * u'
        up_f[i].ρacT = u * up[i].ρacT
    end

end;

function flux_second_order!(
    m::SingleStack{FT,N},
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
) where {FT,N}
    # Alias convention:
    gm   = state
    up   = state.edmf.updraft
    en   = state.edmf.environment
    gm_a = aux
    up_a = aux.edmf.updraft
    en_a = aux.edmf.environment
    gm_d = diffusive
    up_d = diffusive.edmf.updraft
    en_d = diffusive.edmf.environment
    gm_f = flux
    up_f = flux.edmf.updraft
    en_f = flux.edmf.environment
    
    # compute the mixing length and eddy diffusivity 
    # (I am repeating this after doing in sources assumign that it is better to compute this twice than to add mixing length as a aux.variable)
    l = mixing_length(m, m.edmf.mix_len, source, state, diffusive, aux, t, direction, δ, εt)
    K_eddy = m.c_k*l*sqrt(en.tke)
    # flux_second_order in the grid mean is the enviroment turbulent diffussion 
    en_ρa = gm.ρ-sum([up[i].ρa for i in 1:N])
    gm_f.ρe_int += en_ρa*K_eddy*en_d.∇e_int # check prentel number here 
    gm_f.ρq_tot += en_ρa*K_eddy*en_d.∇q_tot # check prentel number here 
    gm_f.ρu     += en_ρa*K_eddy*en_d.∇u     # check prentel number here 
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = α∇ρcT``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρcT` (first order unknown)
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) where {FT,N}
    gm = state⁺
    up = state⁺.edmf.updraft
    if bctype == 1 # bottom
        gm.ρ = m.surface_ρ # find out how is the density at the surafce computed in the LES? 
        gm.ρu = SVector(0,0,0)
        gm.ρe_int = gm.ρ * m.surface_e_int
        gm.ρq_tot = gm.ρ * m.surface_q_tot

        # placeholder to add a function for updraft surface value
        # this dunction should use surface covariancve in the grid mean from a corresponing function
        for i in 1:N
            upd_a_surf, upd_e_int_surf, upd_q_tot_surf  = compute_updraft_surface_BC(i)
            up[i].ρau = SVector(0,0,0)
            up[i].ρa = upd_a_surf
            up[i].ρae_int = upd_e_int_surf
            up[i].ρaq_tot = upd_q_tot_surf 
        end

    elseif bctype == 2 # top
        ## importnet - can the clima implemntation allow for upwinding ?
        # if yes not BC on upd are needed at the top (currently set to GM)
        # if not many issues might  come up with area fraction at the upd top 
        
        gm.ρ = # placeholder to find out how density at the top is computed in the LES? 
        gm.ρu = SVector(0,0,0)

        for i in 1:N
            up[i].ρau = SVector(0,0,0)
            up[i].ρa = 0.0
            up[i].ρae_int = gm.ρe_int*up[i].ρa*ρinv
            up[i].ρaq_tot = gm.ρq_tot*up[i].ρa*ρinv
        end

    end
end;

# The boundary conditions for `ρcT` are specified here for second-order
# unknowns
function boundary_state!(
    nf,
    m::SingleStack{FT,N},
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
) where {FT,N}
    gm = state⁺
    up = state⁺.edmf.updraft
    gm_d = diff⁺
    up_d = diff⁺.edmf.updraft
    if bctype == 1 # bottom
        gm_d.ρ∇e_int = gm.ρ * m.e_int_surface_flux # e_int_surface_flux has units of w'e_int'
        gm_d.ρ∇q_tot = gm.ρ * m.q_tot_surface_flux # q_tot_surface_flux has units of w'q_tot'

        # placeholder to add function for surface TKE and covariances 
        tke, e_int_cv ,q_tot_cv ,e_int_q_tot_cv = get_surface_covariance()
        en_d.ρatke = gm.ρ * area_en * tke
        en_d.ρae_int_cv = gm.ρ * area_en * e_int_cv
        en_d.ρaq_tot_cv = gm.ρ * area_en * q_tot_cv
        en_d.ρae_int_q_tot_cv = gm.ρ * area_en * e_int_q_tot_cv
        
        gm_d.ρ∇u[1] = gm.ρ * m.u_surface_flux # u_surface_flux has units of w'u'
        gm_d.ρ∇u[2] = gm.ρ * m.v_surface_flux # v_surface_flux has units of w'v'
        gm_d.ρ∇e_int = gm.ρ * m.e_int_surface_flux # e_int_surface_flux has units of w'e_int'
        gm_d.ρ∇q_tot = gm.ρ * m.q_tot_surface_flux # q_tot_surface_flux has units of w'q_tot'

    elseif bctype == 2 # top
        # for now zero flux at the top 
        en_d.ρatke = -n⁻ * 0.0
        en_d.ρae_int_cv = -n⁻ * 0.0
        en_d.ρaq_tot_cv = -n⁻ * 0.0
        en_d.ρae_int_q_tot_cv = -n⁻ * 0.0

        gm_d.ρ∇u[1] = -n⁻ * 0.0 
        gm_d.ρ∇u[2] = -n⁻ * 0.0 
        gm_d.ρ∇e_int = -n⁻ * 0.0 
        gm_d.ρ∇q_tot = -n⁻ * 0.0 
    end
end;

# # Spatial discretization

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Specify the number of vertical elements
nelem_vert = 10;

# Specify the domain height
zmax = m.zmax;

# Establish a `ClimateMachine` single stack configuration
driver_config = ClimateMachine.SingleStackConfiguration(
    "SingleStack",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

# # Time discretization

# Specify simulation time (SI units)
t0 = FT(0)
timeend = FT(10)

# We'll define the time-step based on the [Fourier
# number](https://en.wikipedia.org/wiki/Fourier_number)
Δ = min_node_distance(driver_config.grid)

given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 / m.α;
dt = Fourier_bound

# # Configure a `ClimateMachine` solver.

# This initializes the state vector and allocates memory for the solution in
# space (`dg` has the model `m`, which describes the PDEs as well as the
# function used for initialization). This additionally initializes the ODE
# solver, by default an explicit Low-Storage
# [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
# method.

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# ## Inspect the initial conditions

# Let's export a plot of the initial state
output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100 # convert from meters to cm
z_key = "z"
z_label = "z [cm]"
z = get_z(driver_config.grid, z_scale)
# state_vars = get_vars_from_stack(
#     driver_config.grid,
#     solver_config.Q,
#     m,
#     vars_state_conservative,
# );
# aux_vars = get_vars_from_stack(
#     driver_config.grid,
#     solver_config.dg.state_auxiliary,
#     m,
#     vars_state_auxiliary;
#     exclude = [z_key]
# );
# all_vars = OrderedDict(state_vars..., aux_vars...);
# @show keys(all_vars)
# export_plot_snapshot(
#     z,
#     all_vars,
#     ("ρcT", "edmf.environment.ρacT", "edmf.updraft.ρacT"),
#     joinpath(output_dir, "initial_energy.png"),
#     z_label,
# );
# # ![](initial_energy.png)

# export_plot_snapshot(
#     z,
#     all_vars,
#     (
#      "ρu[1]",
#      "edmf.environment.ρau[1]",
#      "edmf.updraft.ρau[1]"
#      ),
#     joinpath(output_dir, "initial_velocity.png"),
#     z_label,
# );
# ![](initial_energy.png)

# It matches what we have in `init_state_conservative!(m::SingleStack, ...)`, so
# let's continue.

# # Solver hooks / callbacks

# Define the number of outputs from `t0` to `timeend`
const n_outputs = 5;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a dictionary for `z` coordinate (and convert to cm) NCDatasets IO:
dims = OrderedDict(z_key => collect(z));

# Create a DataFile, which is callable to get the name of each file given a step
output_data = DataFile(joinpath(output_dir, "output_data"));

# all_data = Dict([k => Dict() for k in 0:n_outputs]...)
# all_data[0] = deepcopy(all_vars)

# The `ClimateMachine`'s time-steppers provide hooks, or callbacks, which
# allow users to inject code to be executed at specified intervals. In this
# callback, the state and aux variables are collected, combined into a single
# `OrderedDict` and written to a NetCDF file (for each output step `step`).
step = [0];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    solver_config.solver,
) do (init = false)
    # state_vars = get_vars_from_stack(
    #     driver_config.grid,
    #     solver_config.Q,
    #     m,
    #     vars_state_conservative,
    # )
    # aux_vars = get_vars_from_stack(
    #     driver_config.grid,
    #     solver_config.dg.state_auxiliary,
    #     m,
    #     vars_state_auxiliary;
    #     exclude = [z_key],
    # )
    # all_vars = OrderedDict(state_vars..., aux_vars...)
    step[1] += 1
    # all_data[step[1]] = deepcopy(all_vars)
    nothing
end;

# # Solve

# This is the main `ClimateMachine` solver invocation. While users do not have
# access to the time-stepping loop, code may be injected via `user_callbacks`,
# which is a `Tuple` of [`GenericCallbacks`](@ref).
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

# # Post-processing

# Our solution has now been calculated and exported to NetCDF files in
# `output_dir`. Let's collect them all into a nested dictionary whose keys are
# the output interval. The next level keys are the variable names, and the
# values are the values along the grid:

# all_data = collect_data(output_data, step[1]);

# To get `T` at ``t=0``, we can use `T_at_t_0 = all_data[0]["T"][:]`
# @show keys(all_data[0])

# Let's plot the solution:

# export_plot(
#     z,
#     all_data,
#     ("ρu[1]","ρu[2]",),
#     joinpath(output_dir, "solution_vs_time.png"),
#     z_label,
# );
# ![](solution_vs_time.png)

# The results look as we would expect: a fixed temperature at the bottom is
# resulting in heat flux that propagates up the domain. To run this file, and
# inspect the solution in `all_data`, include this tutorial in the Julia REPL
# with:

# ```julia
# include(joinpath("tutorials", "Land", "Heat", "heat_equation.jl"))
# ```
