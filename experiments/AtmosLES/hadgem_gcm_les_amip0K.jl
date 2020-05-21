# General Julia modules
using ArgParse
using Dierckx
using Distributions
using DocStringExtensions
using LinearAlgebra
using NCDatasets
using Printf
using Random
using StaticArrays
using Test

# ClimateMachine Modules
using ClimateMachine
using ClimateMachine.Atmos
using ClimateMachine.ConfigTypes
using ClimateMachine.GenericCallbacks
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.Diagnostics
using ClimateMachine.Mesh.Filters
using ClimateMachine.MoistThermodynamics
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.Writers

using CLIMAParameters
using CLIMAParameters.Planet: e_int_v0, grav, day
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()
# Physics specific imports 
import ClimateMachine.DGmethods: vars_state_conservative, vars_state_auxiliary
import ClimateMachine.Atmos: source!, atmos_source!, altitude
import ClimateMachine.Atmos: compute_gradient_flux!, thermo_state

# Citation for problem setup
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

# Reference state setup (GCM reference)
"""
GCMReferenceState{FT} <: ReferenceState
A ReferenceState informed by data from a GCM (Here, HADGEM2)
"""
struct GCMReferenceState{FT} <: ReferenceState end
vars_state_auxiliary(m::GCMReferenceState, FT) = @vars(
    ρ::FT,
    p::FT,
    ta::FT,
    ρe::FT,
    ρq_tot::FT,
    ua::FT,
    va::FT,
    tntha::FT,
    tntva::FT,
    tntr::FT,
    tnhusha::FT,
    tnhusva::FT,
    wap::FT,
    T::FT
)

struct GCMRelaxation{FT} <: Source
    τ_relax::FT
end
function atmos_source!(
    s::GCMRelaxation,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    return nothing
end

# Temperature tendency term, ∂T∂t
"""
    TemperatureTendency <: Source

Temperature tendency for the LES configuration based on quantities 
from a GCM. Quantities are included in standard CMIP naming format. 
Tendencies included here are 

    tntha = temperature tendency due to horizontal advection
    tntva = temperature tendency due to vertical advection
    tntr = temperature tendency due to radiation fluxes
    ∂T∂z = temperature vertical gradient from GCM values
"""
struct TemperatureTendency <: Source end
function atmos_source!(
    s::TemperatureTendency,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction
)
    # Establish problem float-type
    FT = eltype(state)
    _grav = grav(atmos.param_set)
    # Establish vertical orientation
    k̂ = vertical_unit_vector(atmos, aux)
    _e_int_v0 = e_int_v0(atmos.param_set)
    # Unpack vertical gradients
    ∂qt∂z = dot(diffusive.moisture.∇q_tot_gcm, k̂)
    ∂T∂z = dot(diffusive.moisture.∇T_gcm, k̂)
    ρw_adv = -aux.ref_state.wap / _grav
    # Establish thermodynamic state
    TS = thermo_state(atmos, state, aux)
    cvm = cv_m(TS)
    # Compute tendency terms
    # Temperature contribution
    source.ρe += cvm * aux.ref_state.ρ * aux.ref_state.tntha
    source.ρe += cvm * aux.ref_state.ρ * aux.ref_state.tntva
    source.ρe += cvm * aux.ref_state.ρ * aux.ref_state.tntr
    source.ρe += cvm * ∂T∂z * ρw_adv
    # Moisture contribution
    source.ρe += _e_int_v0 * aux.ref_state.tnhusha * aux.ref_state.ρ
    source.ρe += _e_int_v0 * aux.ref_state.tnhusva * aux.ref_state.ρ
    source.ρe += _e_int_v0 * ∂qt∂z * ρw_adv
    # GPU-friendly return nothing
    return nothing
end

# Moisture tendency ∂qt∂t
"""
    MoistureTendency <: Source 

Moisture tendency for the LES configuration based on quantities 
from a GCM. Quantities are included in standard CMIP naming format. 
Tendencies included here are 

    tnhusha = moisture tendency due to horizontal advection
    tnhusva = moisture tendency due to vertical advection
    ∂qt∂z = moisture vertical gradient from GCM values
"""
struct MoistureTendency <: Source end
function atmos_source!(
    s::MoistureTendency,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction
)
    # Establish problem float-type
    FT = eltype(state)
    _grav = grav(atmos.param_set)
    k̂ = vertical_unit_vector(atmos, aux)
    # Establish vertical orientation
    ∂qt∂z = dot(diffusive.moisture.∇q_tot_gcm, k̂)
    ρw_adv = -aux.ref_state.wap / _grav
    # Establish thermodynamic state
    TS = thermo_state(atmos, state, aux)
    cvm = cv_m(TS)
    # Compute tendency terms
    source.moisture.ρq_tot += aux.ref_state.tnhusha * aux.ref_state.ρ
    source.moisture.ρq_tot += aux.ref_state.tnhusva * aux.ref_state.ρ
    source.moisture.ρq_tot += ∂qt∂z * ρw_adv
    # GPU-friendly return nothing
    return nothing
end

# Large-scale subsidence forcing
"""
    Subsidence <: Source 

Subsidence tendency, given a vertical velocity at the large scale, 
obtained from the GCM data. 

    wap = GCM vertical velocity [Pa s⁻¹]. Note the conversion required
"""
struct SubsidenceTendency <: Source end
function atmos_source!(
    s::SubsidenceTendency,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction
)
    _grav = grav(atmos.param_set)
    # Establish vertical orientation
    k̂ = vertical_unit_vector(atmos, aux)
    # Establish subsidence velocity
    w_s = -aux.ref_state.wap / aux.ref_state.ρ / _grav
    # Compute tendency terms
    source.ρe -= state.ρ * w_s * dot(k̂, diffusive.∇h_tot)
    source.moisture.ρq_tot -= state.ρ * w_s * dot(k̂, diffusive.moisture.∇q_tot)
    # GPU-friendly return nothing
    return nothing
end

# Sponge relaxation
"""
  LinearSponge (Source)

Two parameter sponge (α_max, γ) for velocity relaxation to a reference
state.
    α_max = Sponge strength (can be interpreted as timescale)
    γ = Sponge exponent
    z_max = Domain height
    z_sponge = Altitude at which sponge layer starts
"""
struct LinearSponge{FT} <: Source
    "Maximum domain altitude (m)"
    z_max::FT
    "Altitude at with sponge starts (m)"
    z_sponge::FT
    "Sponge Strength 0 ⩽ α_max ⩽ 1"
    α_max::FT
    "Sponge exponent"
    γ::FT
end
function atmos_source!(
    s::LinearSponge,
    atmos::AtmosModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction
)
    #Unpack sponge parameters
    FT = eltype(state)
    z_max = s.z_max
    z_sponge = s.z_sponge
    α_max = s.α_max
    γ = s.γ
    # Establish sponge relaxation velocity
    u_geo = SVector(aux.ref_state.ua, aux.ref_state.va, 0)
    z = altitude(atmos, aux)
    # Accumulate sources
    if z_sponge <= z
        r = (z - z_sponge) / (z_max - z_sponge)
        β_sponge = α_max .* sinpi(r/FT(2)) * sinpi(r/FT(2)) * sinpi(r/ FT(2)) * sinpi(r/FT(2))#.^ γ
        source.ρu -= β_sponge * (state.ρu .- state.ρ * u_geo)
    end
    # GPU-friendly return nothing
    return nothing
end

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

parsed_args = parse_commandline(nothing)
const groupid = parsed_args["group-id"]

# Define the get_gcm_info function
"""
    get_gcm_info(groupid)

For a specific global site, establish and store the GCM state
for each available vertical level. `groupid` refers to the integer
index of the specific global site that we are interested in.
"""
const forcingfile = "HadGEM2-A_amip.2004-2008.07"
function get_gcm_info(groupid)

    @printf("--------------------------------------------------\n")
    @info @sprintf("""\n
       ____ _     ___ __  __    _                                  
      / ___| |   |_ _|  \\/  |  / \\                                 
     | |   | |    | || |\\/| | / _ \\                                
     | |___| |___ | || |  | |/ ___ \\                               
      \\____|_____|___|_| _|_/_/___\\_\\_  __       _     _____ ____  
     | | | | __ _  __| |/ ___| ____|  \\/  |     | |   | ____/ ___| 
     | |_| |/ _` |/ _` | |  _|  _| | |\\/| |_____| |   |  _| \\___ \\ 
     |  _  | (_| | (_| | |_| | |___| |  | |_____| |___| |___ ___) |
     |_| |_|\\__,_|\\__,_|\\____|_____|_|  |_|     |_____|_____|____/ 
     """)

    @printf("\n")
    @printf("Had_GCM_LES = %s\n", groupid)
    @printf("--------------------------------------------------\n")
    filename = "/home/asridhar/CLIMA/datasets/"*forcingfile*".nc"
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
        "alpha",
    )
    # Load NETCDF dataset (HadGEM information)
    # Load the NCDataset (currently we assume all time-stamps are 
    # in the same NCData file). We store this information in `data`. 
    data = NCDataset(filename)
    # To assist the user / inform them of the data processing step
    # we print out some useful information, such as groupnames 
    # and a list of available variables
    @printf("Storing information for group %s ...", groupid)
    for (varname, var) in data.group[groupid]
        for reqvar in req_varnames
            if reqvar == varname
                # Get average over time dimension
                var = mean(var, dims = 2)
                if varname == "hfls" || varname == "hfss"
                    var = mean(var, dims = 1)[1]
                end
                # Assign the variable values to the appropriate converted string
                str2var(varname, var)
            end
        end
        # Store key variables
    end
    @printf("Complete\n")
    @printf("--------------------------------------------------\n")
    @printf("Group data storage complete\n")
    return (
        zg,
        ta,
        hus,
        ua,
        va,
        pfull,
        tntha,
        tntva,
        tntr,
        tnhusha,
        tnhusva,
        wap,
        hfls,
        hfss,
        ta[1],
        alpha,
    )

end

# Initialise the CFSite experiment :D! 
const seed = MersenneTwister(0)
function init_cfsites!(bl, state, aux, (x, y, z), t, spl)
    FT = eltype(state)
    _grav = grav(bl.param_set)

    # Unpack splines, interpolate to z coordinate at 
    # present grid index. (Functions are all pointwise)
    ta = FT(spl.spl_ta(z))
    q_tot = FT(spl.spl_hus(z))
    ua = FT(spl.spl_ua(z))
    va = FT(spl.spl_va(z))
    P = FT(spl.spl_pfull(z))
    tntha = FT(spl.spl_tntha(z))
    tntva = FT(spl.spl_tntva(z))
    tntr = FT(spl.spl_tntr(z))
    tnhusha = FT(spl.spl_tnhusha(z))
    tnhusva = FT(spl.spl_tnhusva(z))
    wap = FT(spl.spl_wap(z))
    ρ_gcm = FT(1 / spl.spl_alpha(z))

    # Compute field properties based on interpolated data
    ρ = air_density(bl.param_set, ta, P, PhasePartition(q_tot))
    e_int = internal_energy(bl.param_set, ta, PhasePartition(q_tot))
    e_kin = (ua^2 + va^2) / 2
    e_pot = _grav * z
    # Assignment of state variables
    state.ρ = ρ
    state.ρu = ρ * SVector(ua, va, 0)
    state.ρe = ρ * (e_kin + e_pot + e_int)
    state.moisture.ρq_tot = ρ * q_tot
    if z <= FT(400)
        state.ρe += rand(seed) * FT(1 / 100) * (state.ρe)
        state.moisture.ρq_tot +=
            rand(seed) * FT(1 / 100) * (state.moisture.ρq_tot)
    end

    # Assign and store the ref variable for sources
    aux.ref_state.ρ = ρ_gcm
    aux.ref_state.p = P
    aux.ref_state.ta = ta
    aux.ref_state.ρe = ρ_gcm * (e_kin + e_pot + e_int)
    aux.ref_state.ρq_tot = ρ_gcm * q_tot
    aux.ref_state.tntha = tntha
    aux.ref_state.tntva = tntva
    aux.ref_state.ua = ua
    aux.ref_state.va = va
    aux.ref_state.tntr = tntr
    aux.ref_state.tnhusha = tnhusha
    aux.ref_state.tnhusva = tnhusva
    aux.ref_state.wap = wap
    aux.ref_state.T = ta

end

function config_cfsites(FT, N, resolution, xmax, ymax, zmax, hfls, hfss, T_sfc)
    # Boundary Conditions
    u_star = FT(0.28)
    zeroflux = FT(0)
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        ref_state = GCMReferenceState{FT}(),
        turbulence = Vreman{FT}(0.23),
        hyperdiffusion = StandardHyperDiffusion(3600),
        source = (
            Gravity(),
            GCMRelaxation{FT}(3600),
            LinearSponge{FT}(zmax, zmax * 0.85, 1, 4),
            MoistureTendency(),
            TemperatureTendency(),
            SubsidenceTendency(),
        ),
        boundarycondition = (
            AtmosBC(
                momentum = Impenetrable(DragLaw(
                    (state, aux, t, normPu_int) -> (u_star / normPu_int)^2,
                )),
                energy = PrescribedEnergyFlux((state, aux, t) -> hfls + hfss),
                moisture = PrescribedMoistureFlux(
                    (state, aux, t) -> hfls / latent_heat_vapor(param_set,T_sfc),
                ),
            ),
            AtmosBC(),
        ),
        moisture = EquilMoist{FT}(; maxiter = 5, tolerance = FT(2)),
        init_state_conservative = init_cfsites!,
    )
    mrrk_solver = ClimateMachine.MultirateSolverType(
        linear_model = AtmosAcousticGravityLinearModel,
        slow_method = LSRK144NiegemannDiehlBusch,
        fast_method = LSRK144NiegemannDiehlBusch,
        timestep_ratio = 10,
    )
    config = ClimateMachine.AtmosLESConfiguration(
        forcingfile*"_$groupid",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_cfsites!,
        solver_type = mrrk_solver,
        model = model,
    )
    return config
end

# Define the diagnostics configuration (Atmos-Default)
function config_diagnostics(driver_config)
    interval = ""
    writer = NetCDFWriter()
    dgngrp = setup_atmos_default_diagnostics(
                interval, driver_config.name; 
                writer=writer
             )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()

    ClimateMachine.init()

    # Working precision
    FT = Float32
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δh = FT(75)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)
    # Domain extents
    xmax = FT(3600)
    ymax = FT(3600)
    zmax = FT(4000)
    # Simulation time
    t0 = FT(0)
    timeend = FT(3600 * 6)
    # Courant number
    CFL = FT(10)

    # Execute the get_gcm_info function
    (
        z,
        ta,
        hus,
        ua,
        va,
        pfull,
        tntha,
        tntva,
        tntr,
        tnhusha,
        tnhusva,
        wap,
        hfls,
        hfss,
        T_sfc,
        alpha,
    ) = get_gcm_info(groupid)

    # Drop dimensions for compatibility with Dierckx
    z = dropdims(z; dims = 2)
    # Create spline objects and pass them into a named tuple
    splines = (
        spl_ta = Spline1D(z, view(ta, :, 1)),
        spl_pfull = Spline1D(z, view(pfull, :, 1)),
        spl_ua = Spline1D(z, view(ua, :, 1)),
        spl_va = Spline1D(z, view(va, :, 1)),
        spl_hus = Spline1D(z, view(hus, :, 1)),
        spl_tntha = Spline1D(z, view(tntha, :, 1)),
        spl_tntva = Spline1D(z, view(tntva, :, 1)),
        spl_tntr = Spline1D(z, view(tntr, :, 1)),
        spl_tnhusha = Spline1D(z, view(tnhusha, :, 1)),
        spl_tnhusva = Spline1D(z, view(tnhusva, :, 1)),
        spl_wap = Spline1D(z, view(wap, :, 1)),
        spl_alpha = Spline1D(z, view(alpha, :, 1)),
    )

    # Set up driver configuration
    driver_config =
        config_cfsites(FT, N, resolution, xmax, ymax, zmax, hfls, hfss, T_sfc)
    # Set up solver configuration
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        splines;
        init_on_cpu = true,
        Courant_number = CFL,
    )
    # Set up diagnostic configuration
    dgn_config = config_diagnostics(driver_config)

    # User defined filter (TMAR positivity preserving filter)
    cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init = false)
        Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
        nothing
    end
    
    filterorder = N
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
            solver_config.Q,
            1:size(solver_config.Q, 2),
            solver_config.dg.grid,
            filter,
        )
        nothing
    end

    # Invoke solver (calls solve! function for time-integrator)
    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )

end
main()
