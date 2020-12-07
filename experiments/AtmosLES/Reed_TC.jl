#!/usr/bin/env julia --project
using NCDatasets
using DelimitedFiles
using ClimateMachine
using ClimateMachine.ArtifactWrappers
using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.TemperatureProfiles
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates
using ClimateMachine.BalanceLaws:
    AbstractStateType, Auxiliary, UpwardIntegrals, DownwardIntegrals

using ArgParse
using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using Printf

using CLIMAParameters
using CLIMAParameters.Planet:
    planet_radius, R_d, Omega, cp_d, MSLP, grav, LH_v0, cv_d

using CLIMAParameters.Atmos.Microphysics

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet <: AbstractIceParameterSet end
struct RainParameterSet <: AbstractRainParameterSet end
struct SnowParameterSet <: AbstractSnowParameterSet end

struct MicropysicsParameterSet{L, I, R, S} <: AbstractMicrophysicsParameterSet
    liq::L
    ice::I
    rai::R
    sno::S
end

struct EarthParameterSet{M} <: AbstractEarthParameterSet
    microphys::M
end

microphys = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)

const param_set = EarthParameterSet(microphys)


"""
  Initial Condition for DYCOMS_RF01 LES

## References
 - [Stevens2005](@cite)
"""
function init_ReedTC!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)
    (x, y, z) = localgeo.coord

    height = altitude(bl, aux)

    if bl.orientation isa SphericalOrientation
        r = sqrt(x^2 + y^2 + z^2)
        lat = asin(z / r)
        lon = atan(y, x + 1e-10)
        cen_lat = FT(10)
        cen_lon = FT(180)
    else
        lat = x
        lon = y
        cen_lat = FT(0)
        cen_lon = FT(0)
    end
    # These constants are those used by Stevens et al. (2005)
    q0 = FT(0.021)
    qtrop = 1e-11
    zq1 = FT(3000)
    zq2 = FT(8000)
    theta_0 = FT(1)
    sigma_r = FT(20000)
    sigma_z = FT(2000)
    r_b = FT(40000)
    z_b = FT(5000)
    Ts0 = FT(302.15)
    constTv = FT(0.608)
    _MSLP = FT(MSLP(bl.param_set))
    _grav = FT(grav(bl.param_set))
    rp = FT(282000)
    convert = FT(180) / pi
    _Omega::FT = Omega(bl.param_set)
    _R_d::FT = R_d(bl.param_set)
    _cp::FT = cp_d(bl.param_set)
    _cv::FT = cv_d(bl.param_set)
    γ = FT(0.007)
    _PR = planet_radius(bl.param_set)
    ztrop = FT(15000)
    exppr = FT(1.5)
    zp = FT(7000)
    exppz = FT(2)
    dp = FT(1115)
    exponent = _R_d * γ / _grav
    T0 = Ts0 * (FT(1) + constTv * q0)
    Ttrop = Ts0 - γ * ztrop
    ptrop = _MSLP * (Ttrop / T0)^(FT(1) / exponent)

    if bl.orientation isa SphericalOrientation
        f = FT(2) * _Omega * sin(cen_lat / convert)
        gr =
            _PR * acos(
                sin(cen_lat / convert) * sin(lat) + (
                    cos(cen_lat / convert) *
                    cos(lat) *
                    cos(lon - cen_lon / convert)
                ),
            )
    else
        gr = sqrt((cen_lat - lat)^2 + (cen_lon - lon)^2)
        f = FT(5e-5)
    end
    ps = _MSLP - dp * exp(-(gr / rp)^exppr)
    if (height > ztrop)
        p = ptrop * exp(-(_grav * (height - ztrop)) / (_R_d * Ttrop))
        pb = p
    else
        p =
            (_MSLP - dp * exp(-(gr / rp)^exppr) * exp(-(height / zp)^exppz)) *
            ((T0 - γ * height) / T0)^(1 / exponent)
        pb = _MSLP * ((T0 - γ * height) / T0)^(1 / exponent)
    end
    if bl.orientation isa SphericalOrientation
        d1 =
            sin(cen_lat / convert) * cos(lat) -
            cos(cen_lat / convert) * sin(lat) * cos(lon - cen_lon / convert)
        d2 = cos(cen_lat / convert) * sin(lon - cen_lon / convert)
        d = max(1e-25, sqrt(d1^2 + d2^2))
        ufac = d1 / d
        vfac = d2 / d
    else
        angle = atan(lon - cen_lon, lat - cen_lat)
        theta = pi / 2 + angle
        radial_decay =
            exp(-height * height / (2 * 5823 * 5823)) * exp(-(gr / 200000)^6)
        ufac = cos(theta) * radial_decay
        vfac = sin(theta) * radial_decay
    end
    if (height > ztrop)
        us = FT(0)
        vs = FT(0)
    else
        vs =
            vfac * (
                -f * gr / 2 + sqrt(
                    (f * gr / 2)^2 -
                    exppr * (gr / rp)^exppr * _R_d * (T0 - γ * height) / (
                        exppz * height * _R_d * (T0 - γ * height) /
                        (_grav * zp^exppz) + (
                            FT(1) -
                            _MSLP / dp *
                            exp((gr / rp)^exppr) *
                            exp((height / zp)^exppz)
                        )
                    ),
                )
            )
        us =
            ufac * (
                -f * gr / 2 + sqrt(
                    (f * gr / 2)^2 -
                    exppr * (gr / rp)^exppr * _R_d * (T0 - γ * height) / (
                        exppz * height * _R_d * (T0 - γ * height) /
                        (_grav * zp^exppz) + (
                            FT(1) -
                            _MSLP / dp *
                            exp((gr / rp)^exppr) *
                            exp((height / zp)^exppz)
                        )
                    ),
                )
            )
    end
    u = FT(us)
    v = FT(vs)
    w = FT(0)
    if bl.orientation isa SphericalOrientation
        u = -us * sin(lon) - vs * sin(lat) * cos(lon)
        v = us * cos(lon) - vs * sin(lat) * cos(lon)
        w = vs * cos(lat)
    end
    if (height > ztrop)
        q = qtrop
    else
        q = q0 * exp(-height / zq1) * exp(-(height / zq2)^exppz)
    end
    qb = q
    if (height > ztrop)
        T = Ttrop
        Tb = T
    else
        T =
            (T0 - γ * height) / (FT(1) + constTv * q) / (
                FT(1) +
                exppz * _R_d * (T0 - γ * height) * height / (
                    _grav *
                    zp^exppz *
                    (
                        FT(1) -
                        _MSLP / dp *
                        exp((gr / rp)^exppr) *
                        exp((height / zp)^exppz)
                    )
                )
            )
        Tb = (T0 - γ * height) / (FT(1) + constTv * q)
    end
    wavenumber = 3
    angle = atan(lat - cen_lat, lon - cen_lon)
    wave = FT(real.(exp(complex(0, 1) * (wavenumber * (pi / 2 - angle)))))
   pert =
        wave *
        theta_0 *
        exp(-((gr - r_b) / sigma_r)^2 - ((height - z_b) / sigma_z)^2)
    T = T + pert
    ρ = p / (_R_d * T * (FT(1) + constTv * q))

    ts = PhaseEquil_ρTq(bl.param_set, ρ, T, q)

    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.ρe = E

    state.moisture.ρq_tot = ρ * q

    if bl.moisture isa NonEquilMoist
        q_init = PhasePartition(ts)
        state.moisture.ρq_liq = q_init.liq
        state.moisture.ρq_ice = q_init.ice
    end
    if bl.precipitation isa Rain
        state.precipitation.ρq_rai = FT(0)
    end
    pi_b = (p / _MSLP)^(_R_d / _cp)
    theta_b = Tb / pi_b
    ρ_b = _MSLP / (_R_d * theta_b) * pi_b^(_cv / _R_d)
    tsb = PhaseEquil_ρTq(bl.param_set, ρ_b, Tb, q)
    E_b = ρ_b * total_energy(FT(0), e_pot, tsb)
    #aux.ref_state.ρ = ρ_b
    #aux.ref_state.T = Tb
    #aux.ref_state.p = pb
    #aux.ref_state.ρe = E_b
    return nothing
end


function read_sounding()
    #read in the original squal sounding
    soundings_dataset = ArtifactWrapper(joinpath(@__DIR__, "Artifacts.toml"),
        "soundings",
	ArtifactFile[ArtifactFile(url = "https://caltech.box.com/shared/static/rjnvt2dlw7etm1c7mmdfrkw5gnfds5lx.nc",filename = "sounding_gabersek.nc",),],
    )
    data_folder = get_data_folder(soundings_dataset)
    fsounding = joinpath(data_folder, "sounding_gabersek.nc")
    sounding = Dataset(fsounding, "r")
    height = sounding["height [m]"][:]
    θ = sounding["theta [K]"][:]
    q_vap = sounding["qv [g kg⁻¹]"][:]
    u = sounding["u [m s⁻¹]"][:]
    v = sounding["v [m s⁻¹]"][:]
    p = sounding["pressure [Pa]"][:]
    close(sounding)
    return (height = height, θ = θ, q_vap = q_vap, u = u, v = v, p = p)
end


function config_ReedTC(
    FT,
    N,
    resolution,
    xmax,
    ymax,
    zmax,
    moisture_model = "equilibrium",
    precipitation_model = "noprecipitation",
)
    # Reference state
    nt = read_sounding()

    zinit = nt.height
    tinit = nt.θ
    qinit = nt.q_vap .* 0.001
    u_init = nt.u
    v_init = nt.v
    p_init = nt.p

    maxz = length(zinit)
    thinit = zeros(maxz)
    piinit = zeros(maxz)
    thinit[1] = tinit[1] / (1 + 0.61 * qinit[1])
    piinit[1] = 1
    for k in 2:maxz
        thinit[k] = tinit[k] / (1 + 0.61 * qinit[k])
        piinit[k] =
            piinit[k - 1] -
            9.81 / (1004 * 0.5 * (tinit[k] + tinit[k - 1])) *
            (zinit[k] - zinit[k - 1])
    end
    T_min = FT(thinit[maxz] * piinit[maxz])
    T_s = FT(thinit[1] * piinit[1])
    Γ_lapse = FT(9.81 / 1004)
    tvmax = 302 * (1 + 0.61 * 0.022)
    deltatv = -(T_min - tvmax)
    tvmin = 197 * (1 + 0.61 * 1e-11)
    @info deltatv
    htv = 8000.0
    #T = DecayingTemperatureProfile(T_min, T_s, Γ_lapse)
    Tv = DecayingTemperatureProfile{FT}(param_set, tvmax, tvmin, htv)
    rel_hum = FT(0)
    ref_state = HydrostaticState(Tv, rel_hum)

    if moisture_model == "equilibrium"
        equilibrium_moisture_model = true
    else
        equilibrium_moisture_model = false
    end

    # Sources
    w_ref = FT(0)
    u_relaxation = SVector(w_ref, w_ref, w_ref)
    # Sponge
    c_sponge = FT(0.75)
    # Rayleigh damping
    zsponge = FT(15000.0)
    rayleigh_sponge =
        RayleighSponge(FT, zmax, zsponge, c_sponge, u_relaxation, 2)
    # Geostrophic forcing
    energy_sponge =
        EnergySponge(FT, zmax, zsponge, c_sponge, u_relaxation, 2)
    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)

    source = (Gravity(), Coriolis(),rayleigh_sponge, energy_sponge, RemovePrecipitation(false))

    # moisture model and its sources
    if moisture_model == "equilibrium"
        moisture = EquilMoist{FT}(; maxiter = 4, tolerance = FT(1))
    elseif moisture_model == "nonequilibrium"
        source = (source..., CreateClouds()...)
        moisture = NonEquilMoist()
    else
        @warn @sprintf(
            """
%s: unrecognized moisture_model in source terms, using the defaults""",
            moisture_model,
        )
        moisture = EquilMoist{FT}(; maxiter = 4, tolerance = FT(1))
    end

    # precipitation model and its sources
    if precipitation_model == "noprecipitation"
        precipitation = NoPrecipitation()
    elseif precipitation_model == "rain"
        source = (source..., Rain_1M())
        precipitation = Rain()
    else
        @warn @sprintf(
            """
%s: unrecognized precipitation_model in source terms, using the defaults""",
            precipitation_model,
        )
        precipitation = NoPrecipitation()
    end

    problem = AtmosProblem(
        boundaryconditions = (
            AtmosBC(
                momentum = (Impenetrable(DragLaw(
                    (state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu,
                ))),
                energy = BulkFormulaEnergy(
                    (atmos, state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu,
                    (atmos, state, aux, t) -> (aux.moisture.temperature, state.moisture.ρq_tot),
                ),
                moisture = BulkFormulaMoisture(
                    (state, aux, t, normPu) -> C_drag + 4 * 1e-5 * normPu,
                    (state, aux, t) -> state.moisture.ρq_tot,
                ),
            ),
            AtmosBC(),
            #AtmosBC(energy = PrescribedTemperature((state⁻, aux⁻, t)-> 218),)
        ),
        init_state_prognostic = init_ReedTC!,
    )


    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        problem = problem,
        ref_state = ref_state,
        turbulence = Vreman{FT}(C_smag),
        moisture = moisture,
        precipitation = precipitation,
        source = source,
    )

    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "ReedTC",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_ReedTC!,
        xmin = -xmax,
        ymin = -ymax,
        solver_type = ode_solver,
        model = model,
    )
    return config
end

function config_diagnostics(driver_config)
    interval = "10000steps"
    dgngrp = setup_atmos_default_diagnostics(
        AtmosLESConfigType(),
        interval,
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([dgngrp])
end

function main()
    # add a command line argument to specify the kind of
    # moisture and precipitation model you want
    # TODO: this will move to the future namelist functionality
    dycoms_args = ArgParseSettings(autofix_names = true)
    add_arg_group!(dycoms_args, "DYCOMS")
    @add_arg_table! dycoms_args begin
        "--moisture-model"
        help = "specify cloud condensate model"
        metavar = "equilibrium|nonequilibrium"
        arg_type = String
        default = "equilibrium"
        "--precipitation-model"
        help = "specify precipitation model"
        metavar = "noprecipitation|rain"
        arg_type = String
        default = "noprecipitation"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = dycoms_args)
    moisture_model = cl_args["moisture_model"]
    precipitation_model = cl_args["precipitation_model"]

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(10000)
    Δv = FT(400)
    resolution = (Δh, Δh, Δv)

    xmax = FT(1000000)
    ymax = FT(1000000)
    zmax = FT(24000)

    t0 = FT(0)
    timeend = FT(86400) #FT(4 * 60 * 60)
    Cmax = FT(0.4)     # use this for single-rate explicit LSRK144

    driver_config = config_ReedTC(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        moisture_model,
        precipitation_model,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = Cmax,
    )
    dgn_config = config_diagnostics(driver_config)

    if moisture_model == "equilibrium"
        filter_vars = ("moisture.ρq_tot",)
    elseif moisture_model == "nonequilibrium"
        filter_vars = ("moisture.ρq_tot", "moisture.ρq_liq", "moisture.ρq_ice")
    end
    if precipitation_model == "rain"
        filter_vars = (filter_vars..., "precipitation.ρq_rai")
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

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        user_callbacks = (cbtmarfilter,),
        check_euclidean_distance = true,
    )
end

main()
