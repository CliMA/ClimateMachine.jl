# General Julia modules
using ArgParse
using Dierckx
using Distributions
using DocStringExtensions
using LinearAlgebra
using NCDatasets
using Pkg.Artifacts
using Printf
using Random
using StaticArrays
using Test
using UnPack

# ClimateMachine Modules
using ClimateMachine

using ClimateMachine.ArtifactWrappers
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.GenericCallbacks
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Grids
using ClimateMachine.Orientations
using ClimateMachine.Thermodynamics
using ClimateMachine.BalanceLaws
using ClimateMachine.TurbulenceClosures
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.Writers


# My additions
using  Dates # added by me for forcing fcn
import Statistics

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Planet parameters

# using CLIMAParameters
# using CLIMAParameters.Planet: e_int_v0, grav, day
# struct EarthParameterSet <: AbstractEarthParameterSet end
# const param_set = EarthParameterSet()

using CLIMAParameters
using CLIMAParameters.Planet: R_d, planet_radius, grav, MSLP, molmass_ratio, e_int_v0, day, LH_v0
using CLIMAParameters.Atmos.Microphysics

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet    <: AbstractIceParameterSet    end
struct RainParameterSet   <: AbstractRainParameterSet   end
struct SnowParameterSet   <: AbstractSnowParameterSet   end


struct MicrophysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet # see dycoms example, https://github.com/CliMA/ClimateMachine.jl/blob/3cd5f471bbee32e8cee037e70bc36c0f35b05a5c/experiments/AtmosLES/dycoms.jl
    liq::L
    ice::I
    rai::R
    sno::S
end
const microphys = MicrophysicsParameterSet(LiquidParameterSet(), IceParameterSet(), RainParameterSet(), SnowParameterSet())

struct EarthParameterSet{M} <: AbstractEarthParameterSet
    microphys::M
end

const param_set = EarthParameterSet(microphys) # is this bad to do? It's being modified, no? 

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Physics specific imports
using ClimateMachine.Atmos: altitude, recover_thermo_state
import ClimateMachine.Atmos: source!, atmos_source!, filter_source
import ClimateMachine.BalanceLaws: source, eq_tends

# # Physics specific imports
# using ClimateMachine.Atmos: altitude, recover_thermo_state
# import ClimateMachine.Atmos: source!, atmos_source!

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Read command line


function parse_commandline(;parse_clargs=true, init_driver=false)

    # Working precision
    FT = Float64

    # Provision for custom command line arguments
    # Convenience args for slurm-array launches
    cfsite_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(cfsite_args, "HadGEM2-A_SiteInfo")
    @add_arg_table! cfsite_args begin
        "--data_path"
            help = "Specify data path for loading model data from"
            arg_type = String
            # default = "/home/jbenjami/Research_Schneider/Data/cfsites/CMIP5/CFMIP2/forcing/"
        "--model"
            help = "Specify model data we're loading"
            arg_type = String
            # default = "HadGEM2-A" # or just nothing perhaps is safer?
        "--exper"
            help = "Specify the experiment we're loading from"
            arg_type = String
            # default = "amip"
        "--rip"
            help = "Specify the RIP (ensemble) value we're loading from"
            arg_type = String
            # default = "r1i1p1"
        "--years"
            help = "Specify the years we want to use data from"
            arg_type = String
            default = "all"
        "--months"
            help = "Specify the months we wish to use"
            arg_type = String
            default = "all"
        "--days"
            help = "Specify the days we wish to use"
            arg_type = String
            default = "all"
        "--hours"
            help = "Specify the hours we wish to use"
            arg_type = String
            default = "all"
        "--minutes"
            help = "Specify the minutes we wish to use"
            arg_type = String
            default = "all"
        "--seconds"
            help = "Specify the seconds we wish to use"
            arg_type = String
            default = "all"
        "--sites"
            # metavar = "site<number>"
            help = "Specify CFSite-IDs for GCM data (averages over selected sites)"
            arg_type = String
            # default = "all"
        "--delta_h"
            help = "Specify horizontal resolution (m)"
            arg_type = FT
            default = FT(75)
        "--delta_v"
            help = "Specify vertical resolution (m)"
            arg_type = FT
            default = FT(20)
        "--xmax"
            help = "Specify maximum x extent (m)"
            arg_type = FT
            default = FT(1800)
        "--ymax"
            help = "Specify maximum y extent (m)"
            arg_type = FT
            default = FT(1800)
        "--zmax"
            help = "Specify maximum z extent (m)"
            arg_type = FT
            default = FT(4000)
        "--tmax"
            help = "Specify maximum time of the simulation (s)"
            arg_type = FT
            default = FT(3600*6)
        "--moisture_model"
            help = "Moisture model to use - eq or non-eq"
            arg_type = String
            default = "nonequilibrium"
        "--precipitation_model"
            help = "precip model"
            arg_type = String
            default = "no_precipitation"

        "--tau_cond_evap"
            help = "Liquid condensation/evaporation relaxation timescale"
            arg_type = typeof(τ_cond_evap(param_set.microphys.liq)) # I think it has to be this way if you want an actual number out, rather then a method? param_set is an actual instance, but param_set has no method at the top level
            default = τ_cond_evap(param_set.microphys.liq) #FT(10) # see https://github.com/CliMA/CLIMAParameters.jl/blob/master/src/Atmos/atmos_parameters.jl
        "--tau_sub_dep"
            help = "Ice sublimation/deposition relaxation timescale"
            arg_type = typeof(τ_sub_dep(param_set.microphys.ice))
            default = τ_sub_dep(param_set.microphys.ice) #FT(10) # see https://github.com/CliMA/CLIMAParameters.jl/blob/master/src/Atmos/atmos_parameters.jl

        "--tau_cond_evap_scale"
            help = "tau_cond_evap scaling factor away from default or prescibed value"
            arg_type = FT
            default = FT(1)
        "--tau_sub_dep_scale"
            help = "τ_sub_dep scaling factor away from default or prescibed value"
            arg_type = FT
            default = FT(1)
        
        "--timestep"
            help = "Timestep for the model solver"
            arg_type = FT # let it resolve to whatever you put in (same type as others), but will default to nothing

        "--solver_type"
            help = "The solver type for the model"
            arg_type = String
            default = "imex_solver"

    end
    return ClimateMachine.init(parse_clargs = parse_clargs, custom_clargs = cfsite_args,init_driver=init_driver)

end

cl_args = parse_commandline()

# These const declarations are essential
const _τ_cond_evap       = cl_args["tau_cond_evap"]
const _τ_sub_dep         = cl_args["tau_sub_dep"]
const _τ_cond_evap_scale = cl_args["tau_cond_evap_scale"]
const _τ_sub_dep_scale   = cl_args["tau_sub_dep_scale"]

# -- I think here we're assigning a fcn method to EarthParameter Set, which is now top level in param_set, and hence it doesn't look deeper to param_set.microphys... ?
# ... not sure if this is the way but who knows... Maybe i should also overwrite the fcn in param_set.microphys... but idk if that makes a diff...
CLIMAParameters.Atmos.Microphysics.τ_cond_evap(::EarthParameterSet) = _τ_cond_evap_scale * _τ_cond_evap # for some reason something like τ_cond_evap(param_set) fails (recursive memory error?), shrug, anyway see https://juliahub.com/docs/CLIMAParameters/B1Qj2/0.1.8/
CLIMAParameters.Atmos.Microphysics.τ_sub_dep(::EarthParameterSet)   = _τ_sub_dep_scale   * _τ_sub_dep   # for some reason something like τ_cond_evap(param_set) fails (recursive memory error?), shrug, anyway see https://juliahub.com/docs/CLIMAParameters/B1Qj2/0.1.8/

# CLIMAParameters.Atmos.Microphysics.τ_cond_evap(::AbstractLiquidParameterSet) = _τ_cond_evap_scale * _τ_cond_evap # for some reason something like τ_cond_evap(param_set) fails (recursive memory error?), shrug, anyway see https://juliahub.com/docs/CLIMAParameters/B1Qj2/0.1.8/
# CLIMAParameters.Atmos.Microphysics.τ_sub_dep(::AbstractIceParameterSet)      = _τ_sub_dep_scale   * _τ_sub_dep   # for some reason something like τ_cond_evap(param_set) fails (recursive memory error?), shrug, anyway see https://juliahub.com/docs/CLIMAParameters/B1Qj2/0.1.8/

CLIMAParameters.Atmos.Microphysics.τ_cond_evap(::LiquidParameterSet) = _τ_cond_evap_scale * _τ_cond_evap # for some reason something like τ_cond_evap(param_set) fails (recursive memory error?), shrug, anyway see https://juliahub.com/docs/CLIMAParameters/B1Qj2/0.1.8/
CLIMAParameters.Atmos.Microphysics.τ_sub_dep(::IceParameterSet)      = _τ_sub_dep_scale   * _τ_sub_dep   # for some reason something like τ_cond_evap(param_set) fails (recursive memory error?), shrug, anyway see https://juliahub.com/docs/CLIMAParameters/B1Qj2/0.1.8/

# print("Running model with τ_cond_evap: ", τ_cond_evap(param_set), "\n") #  CLIMAParameters.Atmos.Microphysics.τ_cond_evap(param_set) also works
# print("Running model with τ_sub_dep: "  , τ_sub_dep(param_set)  , "\n") #  CLIMAParameters.Atmos.Microphysics.τ_sub_dep(param_set)   also works

print("Running model with τ_cond_evap: ", τ_cond_evap(param_set.microphys.liq), "\n") #  CLIMAParameters.Atmos.Microphysics.τ_cond_evap(param_set) also works
print("Running model with τ_sub_dep: "  , τ_sub_dep(param_set.microphys.ice)  , "\n") #  CLIMAParameters.Atmos.Microphysics.τ_sub_dep(param_set)   also works

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Citation for problem setup
## CMIP6 Test Dataset - cfsites
## [Webb2017](@cite)

"""
CMIP6 Test Dataset - cfsites
@Article{gmd-10-359-2017,
AUTHOR = {Webb, M. J. and Andrews, T. and Bodas-Salcedo, A. and Bony, S. and Bretherton, C. S. and Chadwick, R. and Chepfer, H. and Douville, H. and Good, P. and Kay, J. E. and Klein, S. A. and Marchand, R. and Medeiros, B. and Siebesma, A. P. and Skinner, C. B. and Stevens, B. and Tselioudis, G. and Tsushima, Y. and Watanabe, M.},
TITLE = {The Cloud Feedback Model Intercomparison Project (CFMIP) contribution to CMIP6},
JOURNAL = {Geoscientific Model Development},
VOLUME = {10},
YEAR = {2017},
NUMBER = {1},
PAGES = {359--384},
URL = {https://www.geosci-model-dev.net/10/359/2017/},
DOI = {10.5194/gmd-10-359-2017}
}
"""

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Note PV is "Prognostic Variable" not potential vorticity lmao oof see https://github.com/CliMA/ClimateMachine.jl/pull/1742

"""
    GCMRelaxation{PV, FT} <: TendencyDef{Source, PV}

"""
struct GCMRelaxation{PV <: Union{Mass, TotalMoisture}, FT} <:
       TendencyDef{Source, PV}
    τ_relax::FT
end
GCMRelaxation(::Type{FT}, args...) where {FT} = (
    GCMRelaxation{Mass, FT}(args...),
    GCMRelaxation{TotalMoisture, FT}(args...),
)


function source(
    s::GCMRelaxation{Mass},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # TODO: write correct tendency
    return 0
end

function source(
    s::GCMRelaxation{TotalMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # TODO: write correct tendency
    return 0
end

# Ideally we will add a way here to relax over only certain regions.


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

"""
    LargeScaleProcess{PV <: Union{Mass,Energy,TotalMoisture}} <: TendencyDef{Source, PV}

# Energy tendency ∂_t ρe

Temperature tendency for the LES configuration based on quantities
from a GCM. Quantities are included in standard CMIP naming format.
Tendencies included here are

    tntha = temperature tendency due to horizontal advection
    tntva = temperature tendency due to vertical advection
    tntr = temperature tendency due to radiation fluxes
    ∂T∂z = temperature vertical gradient from GCM values

# Moisture tendency ∂_t ρq_tot

Moisture tendency for the LES configuration based on quantities
from a GCM. Quantities are included in standard CMIP naming format.
Tendencies included here are

    tnhusha = moisture tendency due to horizontal advection
    tnhusva = moisture tendency due to vertical advection
    ∂qt∂z = moisture vertical gradient from GCM values
"""
struct LargeScaleProcess{PV <: Union{Mass, Energy, TotalMoisture}} <:
       TendencyDef{Source, PV} end

LargeScaleProcess() = (
    LargeScaleProcess{Mass}(),
    LargeScaleProcess{Energy}(),
    LargeScaleProcess{TotalMoisture}(),
)

# ------------------------------------------------------------------- #

function source(
    s::LargeScaleProcess{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # Establish problem float-type
    FT = eltype(state)
    # Establish vertical orientation
    k̂ = vertical_unit_vector(m, aux)
    _e_int_v0 = e_int_v0(m.param_set)
    # Unpack vertical gradients
    ∂qt∂z = diffusive.lsforcing.∇ᵥhus
    ∂T∂z = diffusive.lsforcing.∇ᵥta
    w_s = aux.lsforcing.w_s
    cvm = cv_m(ts)
    # Compute tendency terms
    # Temperature contribution
    T_tendency = aux.lsforcing.Σtemp_tendency + ∂T∂z * w_s
    # Moisture contribution
    q_tot_tendency =
        compute_q_tot_tend(m, state, aux, t, ts, direction, diffusive)

    return cvm * state.ρ * T_tendency + _e_int_v0 * state.ρ * q_tot_tendency
end

# ------------------------------------------------------------------- #

function compute_q_tot_tend(m, state, aux, t, ts, direction, diffusive) # is this an overwrite? why is it below the above
    # Establish problem float-type
    FT = eltype(state)
    k̂ = vertical_unit_vector(m, aux)
    # Establish vertical orientation
    ∂qt∂z = diffusive.lsforcing.∇ᵥhus
    w_s = aux.lsforcing.w_s
    # Compute tendency terms
    return aux.lsforcing.Σqt_tendency + ∂qt∂z * w_s
end

# ------------------------------------------------------------------- #

function source(
    s::LargeScaleProcess{Mass},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    q_tot_tendency =
        compute_q_tot_tend(m, state, aux, t, ts, direction, diffusive)
    return state.ρ * q_tot_tendency
end

# ------------------------------------------------------------------- #

function source(
    s::LargeScaleProcess{TotalMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    q_tot_tendency =
        compute_q_tot_tend(m, state, aux, t, ts, direction, diffusive)
    return state.ρ * q_tot_tendency
end

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Large-scale subsidence forcing
"""
    Subsidence <: AbstractSource

Subsidence tendency, given a vertical velocity at the large scale,
obtained from the GCM data.

    wap = GCM vertical velocity [Pa s⁻¹]. Note the conversion required
"""
struct SubsidenceTendency <: AbstractSource end

struct LargeScaleSubsidence{PV <: Union{Mass, Energy, TotalMoisture}} <:
       TendencyDef{Source, PV} end

LargeScaleSubsidence() = (
    LargeScaleSubsidence{Mass}(),
    LargeScaleSubsidence{Energy}(),
    LargeScaleSubsidence{TotalMoisture}(),
)

function source(
    s::LargeScaleSubsidence{Mass},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # Establish vertical orientation
    k̂ = vertical_unit_vector(m, aux)
    # Establish subsidence velocity
    w_s = aux.lsforcing.w_s
    return -state.ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
end
function source(
    s::LargeScaleSubsidence{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # Establish vertical orientation
    k̂ = vertical_unit_vector(m, aux)
    # Establish subsidence velocity
    w_s = aux.lsforcing.w_s
    return -state.ρ * w_s * dot(k̂, diffusive.∇h_tot)
end
function source(
    s::LargeScaleSubsidence{TotalMoisture},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    # Establish vertical orientation
    k̂ = vertical_unit_vector(m, aux)
    # Establish subsidence velocity
    w_s = aux.lsforcing.w_s
    return -state.ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
end

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


# Sponge relaxation
"""
    LinearSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}

Two parameter sponge (α_max, γ) for velocity relaxation to a reference
state.
    α_max = Sponge strength (can be interpreted as timescale)
    γ = Sponge exponent
    z_max = Domain height
    z_sponge = Altitude at which sponge layer starts
"""
# struct LinearSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}
#     "Maximum domain altitude (m)"
#     z_max::FT
#     "Altitude at with sponge starts (m)"
#     z_sponge::FT
#     "Sponge Strength 0 ⩽ α_max ⩽ 1"
#     α_max::FT
#     "Sponge exponent"
#     γ::FT
# end


struct LinearSponge{PV <: Union{Momentum, Energy}, FT} <: TendencyDef{Source, PV}
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Sponge exponent"
    γ::FT
end


# ------------------------------------------------------------------- #

# LinearSponge(::Type{FT}, args...) where {FT} =
    # LinearSponge{Momentum, FT}(args...)

# we choose to implement:
# -- Momentum: Yes, it is good to relax energy
# -- Mass    : No, mass/density relaxations seem to sometimes set off gravity waves and destabilize the model
# -- Moisture: No, it is probably not worth it since there's almost no moisture in the sponge
# -- Energy  : Yes, we would like to fix the upper atmospheric temperature as an energy sink

# works this way for gcm_relax
LinearSponge(::Type{FT}, args...) where {FT} = (
    LinearSponge{Momentum, FT}(args...),
    LinearSponge{Energy, FT}(args...),
    # LinearSponge{Moisture, FT}(args...),
)

# ------------------------------------------------------------------- #


function source(
    s::LinearSponge{Momentum},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    #Unpack sponge parameters
    FT = eltype(state)
    @unpack z_max, z_sponge, α_max, γ = s
    # Establish sponge relaxation velocity
    u_geo = SVector(aux.lsforcing.ua, aux.lsforcing.va, 0)
    z = altitude(m, aux)
    # Accumulate sources
    if z_sponge <= z
        r = (z - z_sponge) / (z_max - z_sponge)
        #ZS: different sponge formulation?
        β_sponge = α_max .* sinpi(r / FT(2)) .^ γ
        return -β_sponge * (state.ρu .- state.ρ * u_geo)
    else
        return SVector{3, FT}(0, 0, 0)
    end
end


function source(
    s::LinearSponge{Energy},
    m,
    state,
    aux,
    t,
    ts,
    direction,
    diffusive,
)
    #Unpack sponge parameters
    FT = eltype(state)
    @unpack z_max, z_sponge, α_max, γ = s
    # Establish sponge relaxation velocity
    # u_geo = SVector(aux.lsforcing.ua, aux.lsforcing.va, 0)
    z = altitude(m, aux)
    # Accumulate sources
    if z_sponge <= z
        r = (z - z_sponge) / (z_max - z_sponge)
        #ZS: different sponge formulation?
        β_sponge = α_max .* sinpi(r / FT(2)) .^ γ
        #
        # ts = recover_thermo_state(atmos, state, aux)       
        cvm = cv_m(ts)                                  
        T   = air_temperature(ts) 
        T_tendency  =  (T .-  aux.lsforcing.ta ) 
        return  -β_sponge * (cvm * state.ρ * T_tendency) 
    else
        return 0
    end
end

# function source(
#     s::LinearSponge{Moisture},
#     m,
#     state,
#     aux,
#     t,
#     ts,
#     direction,
#     diffusive,
# )
#     #Unpack sponge parameters
#     FT = eltype(state)
#     @unpack z_max, z_sponge, α_max, γ = s
#     # Establish sponge relaxation velocity
#     # u_geo = SVector(aux.lsforcing.ua, aux.lsforcing.va, 0)
#     z = altitude(m, aux)
#     # Accumulate sources
#     if z_sponge <= z
#         r = (z - z_sponge) / (z_max - z_sponge)
#         #ZS: different sponge formulation?
#         β_sponge = α_max .* sinpi(r / FT(2)) .^ γ
#         #
#         T_tendency  =  (T - aux.lsforcing.ta ) 
#         return  -β_sponge * (state.moisture.ρq_tot .-  aux.lsforcing.ρq_tot)
#     else
#         return 0
#     end
# end


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# I believe these are editing the sources in place  and updating with our inputs

atmos_source!(s::GCMRelaxation, args...) = nothing
atmos_source!(s::LargeScaleProcess, args...) = nothing
atmos_source!(s::LargeScaleSubsidence, args...) = nothing
atmos_source!(s::LinearSponge, args...) = nothing

filter_source(pv::PV, m, s::GCMRelaxation{PV}) where {PV} = s
filter_source(pv::PV, m, s::LargeScaleProcess{PV}) where {PV} = s
filter_source(pv::PV, m, s::LargeScaleSubsidence{PV}) where {PV} = s
filter_source(pv::PV, m, s::LinearSponge{PV}) where {PV} = s

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# We first specify the NetCDF file from which we wish to read our
# GCM values.
# Utility function to read and store variables directly from the
# NetCDF file
"""
    str2var(str::String, var::Any)

Helper function allowing variables read in from the GCM file
to be made available to the LES simulation.
"""
function str2var(str::String, var::Any)
    str = Symbol(str)
    @eval(($str) = ($var))
end
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


function get_gcm_info(; # should be overwritten by default using command_line arguments
    data_path = "",  
    model     = "",
    exper     = "",
    rip       = "",
    sites     = "all", 
    years     = "all",
    months    = "all",
    days      = "all",
    hours     = "all",
    minutes   = "all",
    seconds   = "all"
    )

    @printf("--------------------------------------------------\n")
    @info @sprintf("""\n
    Experiment: GCM-driven LES(ClimateMachine)
    """)

    @printf("\n")
    print("LES = " * model * "-"* exper * "-" *rip * "  |  site: " * string(sites) * " years: " * string(years) * " months: " * string(months) * " days: " * string(days) * " hours: " * string(hours) * " minutes: " * string(minutes) * " seconds: " * string(seconds) * "\n")
    @printf("--------------------------------------------------\n")
    filepath = data_path * "/" * model * "/" * exper * "/"
    # "/central/groups/esm/zhaoyi/GCMForcedLES/forcing/clima/" *
    # forcingfile *
    # ".nc"
    print(filepath * "\n")
    # filenames = filter(contains(r".nc"), readdir(filepath,join=true)) # works in 1.5 but deprecated before?
    filenames = filter(x->occursin(".nc",x), readdir(filepath,join=true)) # julia 1.4.2

    req_varnames = (
    "zg",
    "ta",
    "hus",
    "ua",
    "va",
    "pfull",
    "tntha",
    "tntva",
    "tntr",
    "tnhusha",
    "tnhusva",
    "wap",
    "hfls",
    "hfss",
    "ts",
    "alpha",
    "cli", # taken to help initialize the model... (noneq seems to not like initializing w/o it wit imex_solver)
    "clw"
    )

    # Load NETCDF dataset (HadGEM2-A information)
    # Load the NCDataset (currently we assume all time-stamps are 
    # in the same NCData file). We store this information in `data`. 

    # NCDataset below chokes on merging empty w/ nonempty files so we try to fix that
    # data     = Array{Any}(undef,length(filenames))
    times    = Array{Any}(undef,length(filenames)) # loads times incorrectly (seems to copy from first file)
    indices  = collect(1:length(filenames))

    for (index, file) in enumerate(filenames)
    data = NCDataset(file)
    if length(data["time"]) == 0
    indices = filter!(e->e≠index,indices) # drop those empty files
    else
    times[index] = data["time"] # save the right times
    end
    end

    times = times[indices]
    filenames = filenames[indices]
    time = cat(times...;dims=1) # unpacks using ellipsis

    data = NCDataset(filenames, aggdim = "time", deferopen = false)  # loads times incorrectly (seems to copy from first file)

    if years === "all"
    years = Dates.year.(time) # get all the years
    years = Array(minimum(years):1:maximum(years)) # get an individula vector
    end  
    #
    if months === "all" # double or triple equals? also is there an `is` operator?
    # months =  Array(1:1:12)
    months = Dates.month.(time) # get all the years
    months = Array(minimum(month):1:maximum(month)) # get an individula vector
    end
    #
    if days === "all" # double or triple equals? also is there an `is` operator?
    # dates =  Array(1:1:31)
    days = Dates.day.(time) # get all the years
    days = Array(minimum(days):1:maximum(days)) # get an individula vector
    end
    #
    if hours === "all" # double or triple equals? also is there an `is` operator?
    # hours =  Array(0:1:23)
    hours = Dates.hour.(time) # get all the years
    hours = Array(minimum(hours):1:maximum(hours)) # get an individula vector
    end
    #
    if minutes === "all" # double or triple equals? also is there an `is` operator?
    # minutes =  Array(0:1:59)
    minutes = Dates.minute.(time) # get all the years
    minutes = Array(minimum(minutes):1:maximum(minutes)) # get an individula vector
    end
    #
    if seconds === "all" # double or triple equals? also is there an `is` operator?
    # minutes =  Array(0:1:59)
    seconds = Dates.second.(time) # get all the years
    seconds = Array(minimum(seconds):1:maximum(seconds)) # get an individula vector
    end



    time_mask =
    (Dates.year.(time)   .∈  Ref(years))   .& 
    (Dates.month.(time)  .∈  Ref(months))  .& 
    (Dates.day.(time)    .∈  Ref(days))    .& 
    (Dates.hour.(time)   .∈  Ref(hours))   .& 
    (Dates.minute.(time) .∈  Ref(minutes)) .& 
    (Dates.second.(time) .∈  Ref(seconds)) 


    if sites === "all"
    sites = data["site"][:]
    end
    site_mask = data["site"] .∈ Ref(sites)

    print("time_mask sum: " , string(sum(time_mask)),   " | site_mask_sum: ", string(sum(time_mask)),"\n")


    # To assist the user / inform them of the data processing step
    # we print out some useful information, such as groupnames 
    # and a list of available variables
    @printf("Storing information for site %s ...\n", sites)
    for (varname, var) in data #.group[groupid]
        for reqvar in req_varnames
        # reqvar ≠ varname && continue  # short circuits to skip rest of loop when ≠ is true, but evaluates if reqvar = varname allowing computation for this 
            if reqvar == varname
                print("handling " * varname,"\n")
                # if varname == "hfls" || varname == "hfss" || varname == "ts" # surface properties, have no lev as dim 1
                if any(varname == y for y in ("hfls", "hfss", "ts"))
                    var = var[:,:]
                    var = var[site_mask,time_mask]
                    var = Statistics.mean(var, dims = [1,2])[1] # should work
                else
                    var = var[:,:,:] # Loading in advance makes it much faster
                    # print(size(var),"\n")
                    var = var[:,site_mask,time_mask] # seems slow for some reason, also check order
                    # print(size(var),"\n")
                    # Get average over time dimension
                    var = Statistics.mean(var, dims = [2, 3])
                end
                # print(size(var),"\n")
                # Assign the variable values to the appropriate converted string
                str2var(varname, var)
            end
        end
        # Store key variables
    end
    @printf("Complete\n")
    @printf("--------------------------------------------------\n")
    # @printf("Group data storage complete\n")

    return (
        zg      = zg,
        ta      = ta,
        hus     = hus,
        ua      = ua,
        va      = va,
        pfull   = pfull,
        tntha   = tntha,
        tntva   = tntva,
        tntr    = tntr,
        tnhusha = tnhusha,
        tnhusva = tnhusva,
        wap     = wap,
        hfls    = hfls,
        hfss    = hfss,
        ts      = ts,
        alpha   = alpha,
        cli     = cli,
        clw     = clw
    )

end



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


# Initialise the CFSite experiment :D!
const seed = MersenneTwister(0)
function init_cfsites!(problem, bl, state, aux, localgeo, t, spl)
    FT = eltype(state)
    (x, y, z) = localgeo.coord
    _grav = grav(bl.param_set)

    # Unpack splines, interpolate to z coordinate at
    # present grid index. (Functions are all pointwise)

    ta      = FT(spl.spl_ta(z))
    hus     = FT(spl.spl_hus(z))
    ua      = FT(spl.spl_ua(z))
    va      = FT(spl.spl_va(z))
    pfull   = FT(spl.spl_pfull(z))
    tntha   = FT(spl.spl_tntha(z))
    tntva   = FT(spl.spl_tntva(z))
    tntr    = FT(spl.spl_tntr(z))
    tnhusha = FT(spl.spl_tnhusha(z))
    tnhusva = FT(spl.spl_tnhusva(z))
    wap     = FT(spl.spl_wap(z))
    ρ_gcm   = FT(1 / spl.spl_alpha(z))

    cli     = FT(spl.spl_cli(z))
    clw     = FT(spl.spl_clw(z))

    w_s     = -wap / ρ_gcm / _grav 
    include_vertical_velocity = FT(false)

    u_vec   = SVector(ua, va, w_s*include_vertical_velocity) # initialize the velocity vector


    # Compute field properties based on interpolated data
    ρ     =     air_density(bl.param_set, ta, pfull, PhasePartition(hus))
    e_int = internal_energy(bl.param_set, ta       , PhasePartition(hus))
    e_kin = FT(sum(u_vec.^2)/2) 
    e_pot = _grav * z

    e_tot = e_kin + e_pot + e_int # is this essential? Could one do  total_energy(e_kin, e_pot, ts)
    q_tot = hus + clw + cli

    # Assignment of state variables
    state.ρ = ρ
    state.ρu = ρ * u_vec
    state.ρe = ρ * (e_kin + e_pot + e_int)
    state.moisture.ρq_tot = ρ * hus

    if bl.moisture isa NonEquilMoist # added from bomex, intialize to 0 if nonequilibrium (can try initializing to model also one day)
        # state.moisture.ρq_liq = FT(0)
        # state.moisture.ρq_ice = FT(0)
        state.moisture.ρq_liq = clw # tryn this out
        state.moisture.ρq_ice = cli
    end

    if bl.precipitation isa Rain
        state.precipitation.ρq_rai = FT(0)
    end 

    if z <= FT(400)
        state.ρe += rand(seed) * FT(1 / 100) * (state.ρe)
        state.moisture.ρq_tot +=
            rand(seed) * FT(1 / 100) * (state.moisture.ρq_tot)
    end


    tntr_value = FT(-1/86400) # K/day * 1d/86400s

    # Assign and store the ref variable for sources
    aux.lsforcing.ta = ta
    aux.lsforcing.hus = hus
    aux.lsforcing.Σtemp_tendency = (tntha + tntva + tntr) * 0 + tntr * tntr_value/tntr # ignore the temperature fluxes from advection since they might be large and random, keep some radiative cooling
    aux.lsforcing.ua = ua
    aux.lsforcing.va = va
    aux.lsforcing.Σqt_tendency = tnhusha + tnhusva
    aux.lsforcing.w_s = -wap / ρ_gcm / _grav

    # since we defined this in terms of the splines, we should use the ρ not the ρ_gcm later
    aux.lsforcing.ta             = ta # 
    aux.lsforcing.hus            = hus
    aux.lsforcing.Σtemp_tendency = (tntha + tntva + tntr) * 0 + tntr * tntr_value/tntr # ignore the temperature fluxes from advection since they might be large and random, radiation pretty irrelevant
    aux.lsforcing.ua             = ua
    aux.lsforcing.va             = va
    aux.lsforcing.Σqt_tendency   = (tnhusha + tnhusva) * 0 # Added to try to reign in the madness, no fluxes of moisture from environment
    aux.lsforcing.w_s            = w_s * include_vertical_velocity  # this is bad for forcing since it can be large in covective states, kill

    # use ρ not ρ_gcm so the relaxation works as intended towards the initial state w/o instabilities....
    # aux.lsforcing.ρe                = ρ * e_tot_gcm
    aux.lsforcing.ρ              = ρ
    aux.lsforcing.ρq_tot         = ρ * q_tot

    return nothing
end

function config_cfsites(
    ::Type{FT},
    N,
    resolution,
    xmax,
    ymax,
    zmax,
    hfls,
    hfss,
    ts,
    sites,
    moisture_model,
    precipitation_model,
    solver_type,
) where {FT}
    # Boundary Conditions
    u_star = FT(0.28)  # Friction velocity (explanation from bomex_model)

    print("hfls: ",hfls,"\n")
    print("hfss: ",hfss,"\n")

    test_scale = FT(2.0)


    problem = AtmosProblem(
        boundaryconditions = (
            AtmosBC(
                momentum = Impenetrable(DragLaw(
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = PrescribedEnergyFlux((state, aux, t) -> (abs(hfls) + abs(hfss)) * test_scale),
                moisture = PrescribedMoistureFlux(
                    (state, aux, t) ->
                        abs(hfls)*test_scale / latent_heat_vapor(param_set, ts),
                ),
            ),
            AtmosBC(),
        ),
        init_state_prognostic = init_cfsites!,
    )

    # Setup Default Source Terms
    source_default = (
        Gravity(),
        LinearSponge(FT, zmax, zmax * 0.85, 1, 4)..., # make sure to expand linear sponge since we added more spects to it
        LargeScaleProcess()...,
        LargeScaleSubsidence()...,
    )

    # Add moisture  |  # make sure to splat the precip sources, see https://github.com/CliMA/ClimateMachine.jl/pull/1782
    if moisture_model == "equilibrium"
        source   = source_default
        moisture = EquilMoist{FT}(; maxiter = 5, tolerance = FT(0.1))
    elseif moisture_model == "nonequilibrium"
        source   = (source_default..., CreateClouds()...) # testing w/ precip?
        moisture = NonEquilMoist()
    else
        @warn @sprintf(""" %s: unrecognized moisture_model in source terms, using the defaults""",
            moisture_model,
        )
        source   = source_default
        moisture = EquilMoist{FT}(; maxiter = 5, tolerance = FT(0.1))
    end
    print("Using " , moisture_model , " moisture model: " , moisture , "\n")

    # Precipitation model and its sources  |  # make sure to splat the precip sources, see https://github.com/CliMA/ClimateMachine.jl/pull/1782
    if precipitation_model == "no_precipitation"
        precipitation = NoPrecipitation()
    elseif precipitation_model == "rain"
        source        = (source..., Rain_1M()...) 
        # precipitation = Rain()
        precipitation = RainModel() # copied from dycoms
    elseif precipitation_model == "remove_precipitation"
        source        = (source..., RemovePrecipitation(true)...)
        precipitation = NoPrecipitation()
    else
        @warn @sprintf(
            """%s: unrecognized precipitation_model in source terms, using the defaults""",
            precipitation_model,
        )
        precipitation = NoPrecipitation()
    end
    print("Using " , precipitation_model , " moisture model: " , precipitation , "\n")

    # Final Setup
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem         = problem,
        turbulence      = Vreman{FT}(0.23),
        source          = source,
        moisture        = moisture,
        precipitation   = precipitation,
        #ZS: hyperdiffusion?
        #hyperdiffusion = DryBiharmonic{FT}(12*3600),
        # lsforcing       = HadGEM(), #????
        # lsforcing       = HadGEMVertical(), # in new branch is HadGEMVertical() but where is this defined\
        lsforcing       = CMIP_cfsite_Vertical(), # my own adapted version to keep the variables I might need
    )

    # Timestepper options
    # -- moved to main() so we can set CFL_direction and CFL at the same time, the resolved solver_type is passed in here since that's what AtmosLESConfiguration wants


    # Configuration
    config = ClimateMachine.AtmosLESConfiguration(
        "sites_"*string(sites), # is just a name, see below (is used in file output so sites_# works)
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_cfsites!,
        #ZS: multi-rate?
        solver_type = solver_type,
        model = model,
    )
    return config
end

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Define the diagnostics configuration (Atmos-Default)
function config_diagnostics(driver_config)
    default_dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        "2500steps",
        driver_config.name,
    )
    core_dgngrp = setup_atmos_core_diagnostics(
        AtmosLESConfigType(),
        "2500steps",
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([
        default_dgngrp,
        core_dgngrp,
    ])
end

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

function main()

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4

    # Provision for custom command line arguments
    # Convenience args for slurm-array launches

    cl_args = parse_commandline(;parse_clargs = true, init_driver=true) # default has fix_rng_seed = true in ClimateMachine.init arg parsing, but idk if that's needed


    data_path      = cl_args["data_path"]
    model          = cl_args["model"]
    exper          = cl_args["exper"]
    rip            = cl_args["rip"]
    years          = cl_args["years"]
    months         = cl_args["months"]
    days           = cl_args["days"]
    hours          = cl_args["hours"]
    minutes        = cl_args["minutes"]
    seconds        = cl_args["seconds"]
    sites          = cl_args["sites"]

    if years === "all"
        nothing
    else
        years = eval(Meta.parse(years))
    end

    if months === "all"
        nothing
    else
        months = eval(Meta.parse(months))
    end

    if days === "all"
        nothing
    else
        days = eval(Meta.parse(days))
    end

    if hours === "all"
        nothing
    else
        hours = eval(Meta.parse(hours))
    end

    if minutes === "all"
        nothing
    else
        minutes = eval(Meta.parse(minutes))
    end

    if seconds === "all"
        nothing
    else
        seconds = eval(Meta.parse(seconds))
    end

    if sites === "all"
        nothing
    else
        sites = eval(Meta.parse(sites))
    end

    # Domain resolutions
    Δh         = cl_args["delta_h"]
    Δv         = cl_args["delta_v"]
    resolution = (Δh, Δh, Δv)
    # Domain extents
    xmax       = cl_args["xmax"]
    ymax       = cl_args["ymax"]
    zmax       = cl_args["zmax"]
    # Simulation time
    t0         = FT(0)
    timeend    = cl_args["tmax"]
    timestep   = cl_args["timestep"]
    

    # Microphysics :: Moisture, and relaxation parameters
    moisture_model      = cl_args["moisture_model"]
    precipitation_model = cl_args["precipitation_model"] #"rain"

    solver_type         = cl_args["solver_type"]

    print("solver_type is: " , solver_type, "\n")

    _τ_cond_evap        = cl_args["tau_cond_evap"]
    _τ_sub_dep          = cl_args["tau_sub_dep"]
    _τ_cond_evap_scale  = cl_args["tau_cond_evap_scale"]
    _τ_sub_dep_scale    = cl_args["tau_sub_dep_scale"]

# ------------------------------------------------------------------- #

    print("Running model with τ_cond_evap: ", τ_cond_evap(param_set), "\n") #  CLIMAParameters.Atmos.Microphysics.τ_cond_evap(param_set) also works
    print("Running model with τ_sub_dep: "  , τ_sub_dep(param_set)  , "\n") #  CLIMAParameters.Atmos.Microphysics.τ_sub_dep(param_set)   also works

    print("Running model with τ_cond_evap: ", τ_cond_evap(param_set.microphys.liq), "\n") #  CLIMAParameters.Atmos.Microphysics.τ_cond_evap(param_set) also works
    print("Running model with τ_sub_dep: "  , τ_sub_dep(param_set.microphys.ice)  , "\n") #  CLIMAParameters.Atmos.Microphysics.τ_sub_dep(param_set)   also works

# ------------------------------------------------------------------- #
  
    # Timestepper options
    # ex_solver   = ClimateMachine.ExplicitSolverType() 
    # imex_solver = ClimateMachine.IMEXSolverType()
    if     solver_type == "ex_solver"
        solver_type   = ClimateMachine.ExplicitSolverType() # Explicit Solver
        CFL_direction = nothing
        CFL           = FT(0.8)
    elseif solver_type == "imex_solver"
        solver_type   = ClimateMachine.IMEXSolverType()     # IMEX Solver Type
        CFL_direction = HorizontalDirection()
        CFL           = FT(0.2)
    elseif solver_type == "mrrk_solver"
        solver_type = ClimateMachine.MultirateSolverType( # Multirate Explicit Solver
        fast_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10)
        CFL_direction = nothing # unsure
        CFL           = FT(0.8)
    elseif solver_type == "LSRK144NiegemannDiehlBusch"
        solver_type = ClimateMachine.ExplicitSolverType( # why no semicolon for keyword arg here?
        solver_method = LSRK144NiegemannDiehlBusch,
        )
        CFL_direction = nothing
        CFL           = FT(0.8)
    else
        @warn @sprintf(""" %s: unrecognized solver type, using the defaults""",
            solver_type
            )
        solver_type   = ClimateMachine.IMEXSolverType()
        CFL_direction = HorizontalDirection()
        CFL           = FT(0.2)
    end
    print("Using ", solver_type, " solver type...","\n" )


    # Execute the get_gcm_info function
    ls_forcing = get_gcm_info(;
                                data_path=data_path,
                                model=model,
                                exper=exper,
                                rip=rip,
                                years=years,
                                months=months,
                                days=days,
                                hours=hours,
                                minutes=minutes,
                                seconds=seconds,
                                sites=sites
                                ) # call on our custom fcn

    # Drop dimensions for compatibility with Dierckx
    z = ls_forcing.zg[:] # i think more consistent than dropdims for 1D, can alter later < old was dropdims(ls_forcing.zg; dims = ) >

    # Create spline objects and pass them into a named tuple
    splines = (
        spl_ta      = Spline1D(z, vec(ls_forcing.ta)), # i think [:], vec more consistent than view with unknown trailing singletons, not sure which is better
        spl_pfull   = Spline1D(z, vec(ls_forcing.pfull)),
        spl_ua      = Spline1D(z, vec(ls_forcing.ua)),
        spl_va      = Spline1D(z, vec(ls_forcing.va)),
        spl_hus     = Spline1D(z, vec(ls_forcing.hus)),
        spl_tntha   = Spline1D(z, vec(ls_forcing.tntha)),
        spl_tntva   = Spline1D(z, vec(ls_forcing.tntva)),
        spl_tntr    = Spline1D(z, vec(ls_forcing.tntr)),
        spl_tnhusha = Spline1D(z, vec(ls_forcing.tnhusha)),
        spl_tnhusva = Spline1D(z, vec(ls_forcing.tnhusva)),
        spl_wap     = Spline1D(z, vec(ls_forcing.wap)),
        spl_alpha   = Spline1D(z, vec(ls_forcing.alpha)),
        spl_cli     = Spline1D(z, vec(ls_forcing.cli)),
        spl_clw     = Spline1D(z, vec(ls_forcing.clw)),
    )

    # Set up driver configuration
    driver_config = config_cfsites(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        hfls,
        hfss,
        ts,
        sites,
        moisture_model,
        precipitation_model,
        solver_type
    )

    # Set up solver configuration
    if !isnothing(CFL_direction)
        solver_config = ClimateMachine.SolverConfiguration(
            t0,
            timeend,
            driver_config,
            splines;
            ode_dt = timestep, # added in for config purposes since default seems to evaluate to nan for some reason (is from config setup, default is nothing)
            init_on_cpu = true,
            Courant_number = CFL,
            CFL_direction = CFL_direction # added from as/gcmforcing-cfsite as per zhaoyi instruction
        )
    else # use default (is there a better way to do this and convert nothing to the default?)
        solver_config = ClimateMachine.SolverConfiguration(
            t0,
            timeend,
            driver_config,
            splines;
            ode_dt = timestep, # added in for config purposes since default seems to evaluate to nan for some reason (is from config setup, default is nothing)
            init_on_cpu = true,
            Courant_number = CFL
        )
    end


   #-----------------bomex-les---------------------------------------#
    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "3000steps", FT(0.0001)),
        ClimateMachine.ConservationCheck("ρe", "3000steps", FT(0.0025)),
    )

    if moisture_model == "equilibrium"
        filter_vars = ("moisture.ρq_tot",)
    elseif moisture_model == "nonequilibrium"
        filter_vars = ("moisture.ρq_tot", "moisture.ρq_liq", "moisture.ρq_ice")
    end

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            filter_vars,
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end
    
    # Set up diagnostic configuration
    dgn_config = config_diagnostics(driver_config)

    #-----------------bomex-les---------------------------------------#


    #ZS: cutoff filter?
    filterorder = 2 * N
    filter = BoydVandevenFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            AtmosFilterPerturbations(driver_config.bl),
            solver_config.dg.grid,
            filter,
        )
        nothing
    end

    cutoff_filter = CutoffFilter(solver_config.dg.grid, N - 1)
    cbcutoff = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(solver_config.Q, 1:6, solver_config.dg.grid, cutoff_filter)
        nothing
    end

    # Invoke solver (calls solve! function for time-integrator)
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        #ZS: only tmar?
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

end
main()
