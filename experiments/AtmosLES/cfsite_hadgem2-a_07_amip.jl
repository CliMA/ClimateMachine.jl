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

# Planet parameters
using CLIMAParameters
using CLIMAParameters.Planet: e_int_v0, grav, day
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
# Physics specific imports
using ClimateMachine.Atmos: altitude, recover_thermo_state
import ClimateMachine.Atmos: source!, atmos_source!, filter_source
import ClimateMachine.BalanceLaws: source, eq_tends

# Citation for problem setup
## CMIP6 Test Dataset - cfsites
## [Webb2017](@cite)

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

function source(s::GCMRelaxation{Mass}, m, args)
    # TODO: write correct tendency
    return 0
end

function source(s::GCMRelaxation{TotalMoisture}, m, args)
    # TODO: write correct tendency
    return 0
end

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

function source(s::LargeScaleProcess{Energy}, m, args)
    @unpack state, aux, diffusive = args
    @unpack ts = args.precomputed
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
    q_tot_tendency = compute_q_tot_tend(m, args)

    return cvm * state.ρ * T_tendency + _e_int_v0 * state.ρ * q_tot_tendency
end

function compute_q_tot_tend(m, args)
    @unpack aux, diffusive = args
    # Establish problem float-type
    FT = eltype(aux)
    k̂ = vertical_unit_vector(m, aux)
    # Establish vertical orientation
    ∂qt∂z = diffusive.lsforcing.∇ᵥhus
    w_s = aux.lsforcing.w_s
    # Compute tendency terms
    return aux.lsforcing.Σqt_tendency + ∂qt∂z * w_s
end

function source(s::LargeScaleProcess{Mass}, m, args)
    @unpack state = args
    q_tot_tendency = compute_q_tot_tend(m, args)
    return state.ρ * q_tot_tendency
end

function source(s::LargeScaleProcess{TotalMoisture}, m, args)
    @unpack state, aux = args
    q_tot_tendency = compute_q_tot_tend(m, args)
    return state.ρ * q_tot_tendency
end

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

function source(s::LargeScaleSubsidence{Mass}, m, args)
    @unpack state, aux, diffusive = args
    # Establish vertical orientation
    k̂ = vertical_unit_vector(m, aux)
    # Establish subsidence velocity
    w_s = aux.lsforcing.w_s
    return -state.ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
end
function source(s::LargeScaleSubsidence{Energy}, m, args)
    @unpack state, aux, diffusive = args
    # Establish vertical orientation
    k̂ = vertical_unit_vector(m, aux)
    # Establish subsidence velocity
    w_s = aux.lsforcing.w_s
    return -state.ρ * w_s * dot(k̂, diffusive.∇h_tot)
end
function source(s::LargeScaleSubsidence{TotalMoisture}, m, args)
    @unpack state, aux, diffusive = args
    # Establish vertical orientation
    k̂ = vertical_unit_vector(m, aux)
    # Establish subsidence velocity
    w_s = aux.lsforcing.w_s
    return -state.ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
end

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
struct LinearSponge{PV <: Momentum, FT} <: TendencyDef{Source, PV}
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Sponge exponent"
    γ::FT
end

LinearSponge(::Type{FT}, args...) where {FT} =
    LinearSponge{Momentum, FT}(args...)

function source(s::LinearSponge{Momentum}, m, args)
    @unpack state, aux = args
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

atmos_source!(s::GCMRelaxation, args...) = nothing
atmos_source!(s::LargeScaleProcess, args...) = nothing
atmos_source!(s::LargeScaleSubsidence, args...) = nothing
atmos_source!(s::LinearSponge, args...) = nothing

filter_source(pv::PV, m, s::GCMRelaxation{PV}) where {PV} = s
filter_source(pv::PV, m, s::LargeScaleProcess{PV}) where {PV} = s
filter_source(pv::PV, m, s::LargeScaleSubsidence{PV}) where {PV} = s
filter_source(pv::PV, m, s::LinearSponge{PV}) where {PV} = s

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

# Define the get_gcm_info function
const forcingfile = "HadGEM2-A_amip.2004-2008.07"
"""
    get_gcm_info(group_id)

For a specific global site, establish and store the GCM state
for each available vertical level. `group_id` refers to the integer
index of the specific global site that we are interested in.
"""
function get_gcm_info(group_id)

    @printf("--------------------------------------------------\n")
    @info @sprintf("""\n
     Experiment: GCM(HadGEM2-A) driven LES(ClimateMachine)
     """)

    @printf("\n")
    @printf("HadGEM2-A_LES = %s\n", group_id)
    @printf("--------------------------------------------------\n")

    lsforcing_dataset = ArtifactWrapper(
        joinpath(@__DIR__, "Artifacts.toml"),
        "lsforcing",
        ArtifactFile[ArtifactFile(
            url = "https://caltech.box.com/shared/static/dszfbqzwgc9a55vhxd43yenvebcb6bcj.nc",
            filename = forcingfile,
        ),],
    )
    lsforcing_dataset_path = get_data_folder(lsforcing_dataset)
    data_file = joinpath(lsforcing_dataset_path, forcingfile)

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
    )
    # Load NETCDF dataset (HadGEM2-A information)
    # Load the NCDataset (currently we assume all time-stamps are
    # in the same NCData file). We store this information in `data`.
    data = NCDataset(data_file)
    # To assist the user / inform them of the data processing step
    # we print out some useful information, such as groupnames
    # and a list of available variables
    @printf("Storing information for group %s ...", group_id)
    for (varname, var) in data.group[group_id]
        for reqvar in req_varnames
            reqvar ≠ varname && continue
            # Get average over time dimension
            var = mean(var, dims = 2)
            if any(varname == y for y in ("hfls", "hfss", "ts"))
                var = mean(var, dims = 1)[1]
            end
            # Assign the variable values to the appropriate converted string
            str2var(varname, var)
        end
        # Store key variables
    end
    @printf("Complete\n")
    @printf("--------------------------------------------------\n")
    @printf("Group data storage complete\n")
    return (
        zg = zg,
        ta = ta,
        hus = hus,
        ua = ua,
        va = va,
        pfull = pfull,
        tntha = tntha,
        tntva = tntva,
        tntr = tntr,
        tnhusha = tnhusha,
        tnhusva = tnhusva,
        wap = wap,
        hfls = hfls,
        hfss = hfss,
        ts = ts,
        alpha = alpha,
    )

end

# Initialise the CFSite experiment :D!
const seed = MersenneTwister(0)
function init_cfsites!(problem, bl, state, aux, localgeo, t, spl)
    FT = eltype(state)
    (x, y, z) = localgeo.coord
    _grav = grav(bl.param_set)

    # Unpack splines, interpolate to z coordinate at
    # present grid index. (Functions are all pointwise)
    ta = FT(spl.spl_ta(z))
    hus = FT(spl.spl_hus(z))
    ua = FT(spl.spl_ua(z))
    va = FT(spl.spl_va(z))
    pfull = FT(spl.spl_pfull(z))
    tntha = FT(spl.spl_tntha(z))
    tntva = FT(spl.spl_tntva(z))
    tntr = FT(spl.spl_tntr(z))
    tnhusha = FT(spl.spl_tnhusha(z))
    tnhusva = FT(spl.spl_tnhusva(z))
    wap = FT(spl.spl_wap(z))
    ρ_gcm = FT(1 / spl.spl_alpha(z))

    # Compute field properties based on interpolated data
    ρ = air_density(bl.param_set, ta, pfull, PhasePartition(hus))
    e_int = internal_energy(bl.param_set, ta, PhasePartition(hus))
    e_kin = (ua^2 + va^2) / 2
    e_pot = _grav * z
    # Assignment of state variables
    state.ρ = ρ
    state.ρu = ρ * SVector(ua, va, 0)
    state.ρe = ρ * (e_kin + e_pot + e_int)
    state.moisture.ρq_tot = ρ * hus
    if z <= FT(400)
        state.ρe += rand(seed) * FT(1 / 100) * (state.ρe)
        state.moisture.ρq_tot +=
            rand(seed) * FT(1 / 100) * (state.moisture.ρq_tot)
    end

    # Assign and store the ref variable for sources
    aux.lsforcing.ta = ta
    aux.lsforcing.hus = hus
    aux.lsforcing.Σtemp_tendency = tntha + tntva + tntr
    aux.lsforcing.ua = ua
    aux.lsforcing.va = va
    aux.lsforcing.Σqt_tendency = tnhusha + tnhusva
    aux.lsforcing.w_s = -wap / ρ_gcm / _grav
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
    group_id,
) where {FT}
    # Boundary Conditions
    u_star = FT(0.28)

    problem = AtmosProblem(
        boundaryconditions = (
            AtmosBC(
                momentum = Impenetrable(DragLaw(
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = PrescribedEnergyFlux((state, aux, t) -> hfls + hfss),
                moisture = PrescribedMoistureFlux(
                    (state, aux, t) ->
                        hfls / latent_heat_vapor(param_set, ts),
                ),
            ),
            AtmosBC(),
        ),
        init_state_prognostic = init_cfsites!,
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        turbulence = Vreman{FT}(0.23),
        source = (
            Gravity(),
            LinearSponge(FT, zmax, zmax * 0.85, 1, 4),
            LargeScaleProcess()...,
            LargeScaleSubsidence()...,
        ),
        moisture = EquilMoist{FT}(; maxiter = 5, tolerance = FT(2)),
        lsforcing = HadGEMVertical(),
    )

    # Timestepper options

    # Explicit Solver
    ex_solver = ClimateMachine.ExplicitSolverType()

    # Multirate Explicit Solver
    mrrk_solver = ClimateMachine.MultirateSolverType(
        fast_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10,
    )

    # IMEX Solver Type
    imex_solver = ClimateMachine.IMEXSolverType()

    # Configuration
    config = ClimateMachine.AtmosLESConfiguration(
        forcingfile * "_$group_id",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_cfsites!,
        #ZS: multi-rate?
        solver_type = imex_solver,
        model = model,
    )
    return config
end

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

function main()

    # Provision for custom command line arguments
    # Convenience args for slurm-array launches
    cfsite_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(cfsite_args, "HadGEM2-A_SiteInfo")
    @add_arg_table! cfsite_args begin
        "--group-id"
        help = "Specify cfSite-ID for forcing data"
        metavar = "site<number>"
        arg_type = String
        default = "site17"
    end
    cl_args = ClimateMachine.init(
        parse_clargs = true,
        custom_clargs = cfsite_args,
        fix_rng_seed = true,
    )
    group_id = cl_args["group_id"]

    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(75)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)
    # Domain extents
    xmax = FT(1800)
    ymax = FT(1800)
    zmax = FT(4000)
    # Simulation time
    t0 = FT(0)
    timeend = FT(600)
    #timeend = FT(3600 * 6)
    # Courant number
    CFL = FT(0.2)

    # Execute the get_gcm_info function
    ls_forcing = get_gcm_info(group_id)

    # Drop dimensions for compatibility with Dierckx
    z = dropdims(ls_forcing.zg; dims = 2)
    # Create spline objects and pass them into a named tuple
    splines = (
        spl_ta = Spline1D(z, view(ls_forcing.ta, :, 1)),
        spl_pfull = Spline1D(z, view(ls_forcing.pfull, :, 1)),
        spl_ua = Spline1D(z, view(ls_forcing.ua, :, 1)),
        spl_va = Spline1D(z, view(ls_forcing.va, :, 1)),
        spl_hus = Spline1D(z, view(ls_forcing.hus, :, 1)),
        spl_tntha = Spline1D(z, view(ls_forcing.tntha, :, 1)),
        spl_tntva = Spline1D(z, view(ls_forcing.tntva, :, 1)),
        spl_tntr = Spline1D(z, view(ls_forcing.tntr, :, 1)),
        spl_tnhusha = Spline1D(z, view(ls_forcing.tnhusha, :, 1)),
        spl_tnhusva = Spline1D(z, view(ls_forcing.tnhusva, :, 1)),
        spl_wap = Spline1D(z, view(ls_forcing.wap, :, 1)),
        spl_alpha = Spline1D(z, view(ls_forcing.alpha, :, 1)),
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
        group_id,
    )
    # Set up solver configuration
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        splines;
        init_on_cpu = true,
        Courant_number = CFL,
        CFL_direction = HorizontalDirection(),
    )
    # Set up diagnostic configuration
    dgn_config = config_diagnostics(driver_config)

    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            ("moisture.ρq_tot",),
            solver_config.dg.grid,
            TMARFilter(),
        )
        nothing
    end

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
