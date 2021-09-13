#!/usr/bin/env julia --project
using ClimateMachine

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using Thermodynamics.TemperatureProfiles
using Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using ClimateMachine.BalanceLaws
import ClimateMachine.BalanceLaws:
    vars_state,
    prognostic_vars,
    flux,
    eq_tends,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

using ArgParse
using UnPack
using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra
using Printf

using CLIMAParameters
using CLIMAParameters.Planet: cp_d, MSLP, grav, LH_v0

using CLIMAParameters.Atmos.Microphysics

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

# ------------------------ Begin Radiation Model ---------------------- #

"""
    DYCOMSRadiationModel <: RadiationModel

## References
 - [Stevens2005](@cite)
"""
struct DYCOMSRadiationModel{FT} <: RadiationModel
    "mass absorption coefficient `[m^2/kg]`"
    κ::FT
    "Troposphere cooling parameter `[m^(-4/3)]`"
    α_z::FT
    "Inversion height `[m]`"
    z_i::FT
    "Density"
    ρ_i::FT
    "Large scale divergence `[s^(-1)]`"
    D_subsidence::FT
    "Radiative flux parameter `[W/m^2]`"
    F_0::FT
    "Radiative flux parameter `[W/m^2]`"
    F_1::FT
    "is AtmosLES moisture model an equilibrium model"
    equilibrium_moisture_model::Bool
end

struct DYCOMSRadiation <: TendencyDef{Flux{FirstOrder}} end

eq_tends(pv::Energy, ::DYCOMSRadiationModel, ::Flux{FirstOrder}) =
    (DYCOMSRadiation(),)

function flux(::Energy, ::DYCOMSRadiation, atmos, args)
    @unpack state, aux = args
    m = radiation_model(atmos)
    FT = eltype(state)
    z = altitude(atmos, aux)
    Δz_i = max(z - m.z_i, -zero(FT))
    # Constants
    upward_flux_from_cloud = m.F_0 * exp(-aux.∫dnz.radiation.attenuation_coeff)
    upward_flux_from_sfc = m.F_1 * exp(-aux.∫dz.radiation.attenuation_coeff)
    param_set = parameter_set(atmos)
    free_troposphere_flux =
        m.ρ_i *
        FT(cp_d(param_set)) *
        m.D_subsidence *
        m.α_z *
        cbrt(Δz_i) *
        (Δz_i / 4 + m.z_i)
    F_rad =
        upward_flux_from_sfc + upward_flux_from_cloud + free_troposphere_flux
    ẑ = vertical_unit_vector(atmos, aux)
    return F_rad * ẑ
end

vars_state(m::DYCOMSRadiationModel, ::Auxiliary, FT) = @vars(Rad_flux::FT)

vars_state(m::DYCOMSRadiationModel, ::UpwardIntegrals, FT) =
    @vars(attenuation_coeff::FT)
function integral_load_auxiliary_state!(
    m::DYCOMSRadiationModel,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    FT = eltype(state)

    if m.equilibrium_moisture_model
        integrand.radiation.attenuation_coeff =
            state.ρ * m.κ * aux.moisture.q_liq
    else
        integrand.radiation.attenuation_coeff = m.κ * state.moisture.ρq_liq
    end
end
function integral_set_auxiliary_state!(
    m::DYCOMSRadiationModel,
    aux::Vars,
    integral::Vars,
)
    integral = integral.radiation.attenuation_coeff
    aux.∫dz.radiation.attenuation_coeff = integral
end

vars_state(m::DYCOMSRadiationModel, ::DownwardIntegrals, FT) =
    @vars(attenuation_coeff::FT)
function reverse_integral_load_auxiliary_state!(
    m::DYCOMSRadiationModel,
    integrand::Vars,
    state::Vars,
    aux::Vars,
)
    FT = eltype(state)
    integrand.radiation.attenuation_coeff = aux.∫dz.radiation.attenuation_coeff
end
function reverse_integral_set_auxiliary_state!(
    m::DYCOMSRadiationModel,
    aux::Vars,
    integral::Vars,
)
    aux.∫dnz.radiation.attenuation_coeff = integral.radiation.attenuation_coeff
end

# -------------------------- End Radiation Model ------------------------ #

"""
  Initial Condition for DYCOMS_RF01 LES

## References
 - [Stevens2005](@cite)
"""
function init_dycoms!(problem, bl, state, aux, localgeo, t)
    FT = eltype(state)

    (x, y, z) = localgeo.coord
    param_set = parameter_set(bl)

    z = altitude(bl, aux)

    # These constants are those used by Stevens et al. (2005)
    qref = FT(9.0e-3)
    q_pt_sfc = PhasePartition(qref)
    Rm_sfc = gas_constant_air(param_set, q_pt_sfc)
    T_sfc = FT(290.4)
    _MSLP = FT(MSLP(param_set))
    _grav = FT(grav(param_set))

    # Specify moisture profiles
    q_liq = FT(0)
    q_ice = FT(0)
    zb = FT(600)         # initial cloud bottom
    zi = FT(840)         # initial cloud top

    if z <= zi
        θ_liq = FT(289.0)
        q_tot = qref
    else
        θ_liq = FT(297.0) + (z - zi)^(FT(1 / 3))
        q_tot = FT(1.5e-3)
    end

    ugeo = FT(7)
    vgeo = FT(-5.5)
    u, v, w = ugeo, vgeo, FT(0)

    # Perturb initial state to break symmetry and trigger turbulent convection
    r1 = FT(rand(Uniform(-0.001, 0.001)))
    if z <= 200.0
        θ_liq += r1 * θ_liq
    end

    # Pressure
    H = Rm_sfc * T_sfc / _grav
    p = _MSLP * exp(-z / H)

    # Density, Temperature

    ts = PhaseEquil_pθq(param_set, p, θ_liq, q_tot)
    ρ = air_density(ts)

    e_kin = FT(1 / 2) * FT((u^2 + v^2 + w^2))
    e_pot = gravitational_potential(bl.orientation, aux)
    E = ρ * total_energy(e_kin, e_pot, ts)

    state.ρ = ρ
    state.ρu = SVector(ρ * u, ρ * v, ρ * w)
    state.energy.ρe = E

    state.moisture.ρq_tot = ρ * q_tot

    if moisture_model(bl) isa NonEquilMoist
        q_init = PhasePartition(ts)
        state.moisture.ρq_liq = q_init.liq
        state.moisture.ρq_ice = q_init.ice
    end
    if precipitation_model(bl) isa RainModel
        state.precipitation.ρq_rai = FT(0)
    end

    return nothing
end

function config_dycoms(
    ::Type{FT},
    N,
    resolution,
    xmax,
    ymax,
    zmax,
    moisture_model = "equilibrium",
    precipitation_model = "noprecipitation",
) where {FT}
    # Reference state
    T_profile = DecayingTemperatureProfile{FT}(param_set)
    ref_state = HydrostaticState(T_profile)

    # Radiation model
    κ = FT(85)
    α_z = FT(1)
    z_i = FT(840)
    ρ_i = FT(1.13)

    D_subsidence = FT(3.75e-6)

    F_0 = FT(70)
    F_1 = FT(22)
    if moisture_model == "equilibrium"
        equilibrium_moisture_model = true
    else
        equilibrium_moisture_model = false
    end
    radiation = DYCOMSRadiationModel{FT}(
        κ,
        α_z,
        z_i,
        ρ_i,
        D_subsidence,
        F_0,
        F_1,
        equilibrium_moisture_model,
    )

    # Sources
    f_coriolis = FT(0.762e-4)
    u_geostrophic = FT(7.0)
    v_geostrophic = FT(-5.5)
    w_ref = FT(0)
    u_relaxation = SVector(u_geostrophic, v_geostrophic, w_ref)
    # Sponge
    c_sponge = 1
    # Rayleigh damping
    zsponge = FT(1000.0)
    rayleigh_sponge =
        RayleighSponge{FT}(zmax, zsponge, c_sponge, u_relaxation, 2)
    # Geostrophic forcing
    geostrophic_forcing =
        GeostrophicForcing{FT}(f_coriolis, u_geostrophic, v_geostrophic)

    # Boundary conditions
    # SGS Filter constants
    C_smag = FT(0.21) # 0.21 for stable testing, 0.18 in practice
    C_drag = FT(0.0011)
    LHF = FT(115)
    SHF = FT(15)
    moisture_flux = LHF / FT(LH_v0(param_set))

    source = (
        Gravity(),
        rayleigh_sponge,
        Subsidence{FT}(D_subsidence),
        geostrophic_forcing,
    )

    # moisture model and its sources
    if moisture_model == "equilibrium"
        moisture = EquilMoist(; maxiter = 4, tolerance = FT(1))
    elseif moisture_model == "nonequilibrium"
        source = (source..., CreateClouds())
        moisture = NonEquilMoist()
    else
        @warn @sprintf(
            """
%s: unrecognized moisture_model in source terms, using the defaults""",
            moisture_model,
        )
        moisture = EquilMoist(; maxiter = 4, tolerance = FT(1))
    end

    # precipitation model and its sources
    if precipitation_model == "noprecipitation"
        precipitation = NoPrecipitation()
    elseif precipitation_model == "rain"
        source = (source..., WarmRain_1M())
        precipitation = RainModel()
    else
        @warn @sprintf(
            """
%s: unrecognized precipitation_model in source terms, using the defaults""",
            precipitation_model,
        )
        precipitation = NoPrecipitation()
    end

    physics = AtmosPhysics{FT}(
        param_set;
        ref_state = ref_state,
        turbulence = Vreman{FT}(C_smag),
        moisture = moisture,
        precipitation = precipitation,
        radiation = radiation,
    )

    problem = AtmosProblem(
        boundaryconditions = (
            AtmosBC(
                physics;
                momentum = Impenetrable(DragLaw(
                    (state, aux, t, normPu) -> C_drag,
                )),
                energy = PrescribedEnergyFlux((state, aux, t) -> LHF + SHF),
                moisture = PrescribedMoistureFlux(
                    (state, aux, t) -> moisture_flux,
                ),
            ),
            AtmosBC(physics;),
        ),
        init_state_prognostic = init_dycoms!,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        physics;
        problem = problem,
        source = source,
    )

    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "DYCOMS",
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        param_set,
        init_dycoms!,
        model = model,
    )
    return config, ode_solver
end

function config_diagnostics(driver_config, timeend)
    ssecs = cld(timeend, 2) + 10
    interval = "$(ssecs)ssecs"
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
        "--check-asserts"
        help = "should asserts be checked at the end of the simulation"
        metavar = "yes|no"
        arg_type = String
        default = "no"
    end

    cl_args =
        ClimateMachine.init(parse_clargs = true, custom_clargs = dycoms_args)
    moisture_model = cl_args["moisture_model"]
    precipitation_model = cl_args["precipitation_model"]
    check_asserts = cl_args["check_asserts"]

    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution and size
    Δh = FT(40)
    Δv = FT(20)
    resolution = (Δh, Δh, Δv)

    xmax = FT(1000)
    ymax = FT(1000)
    zmax = FT(1500)

    t0 = FT(0)
    timeend = FT(100) #FT(4 * 60 * 60)
    Cmax = FT(1.7)     # use this for single-rate explicit LSRK144

    driver_config, ode_solver_type = config_dycoms(
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
        ode_solver_type = ode_solver_type,
        init_on_cpu = true,
        Courant_number = Cmax,
    )
    dgn_config = config_diagnostics(driver_config, timeend)

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

    # some simple checks to ensure that rain and clouds exist in the CI runs
    if check_asserts == "yes"

        m = driver_config.bl
        Q = solver_config.Q
        ρ_ind = varsindex(vars_state(m, Prognostic(), FT), :ρ)

        if moisture_model == "nonequilibrium"

            ρq_liq_ind =
                varsindex(vars_state(m, Prognostic(), FT), :moisture, :ρq_liq)
            ρq_ice_ind =
                varsindex(vars_state(m, Prognostic(), FT), :moisture, :ρq_ice)

            min_q_liq = minimum(abs.(
                Array(Q[:, ρq_liq_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_liq = maximum(abs.(
                Array(Q[:, ρq_liq_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))

            min_q_ice = minimum(abs.(
                Array(Q[:, ρq_ice_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_ice = maximum(abs.(
                Array(Q[:, ρq_ice_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))

            @info(min_q_liq, max_q_liq)
            @info(min_q_ice, max_q_ice)

            # test that cloud condensate variables exist and are not NaN
            @test !isnan(max_q_liq)
            @test !isnan(max_q_ice)

            # test that there is reasonable amount of cloud water...
            @test abs(max_q_liq) > FT(5e-4)

            # ...and that there is no cloud ice
            @test isequal(min_q_ice, FT(0))
            @test isequal(max_q_ice, FT(0))


        end
        if precipitation_model == "rain"
            ρq_rai_ind = varsindex(
                vars_state(m, Prognostic(), FT),
                :precipitation,
                :ρq_rai,
            )

            min_q_rai = minimum(abs.(
                Array(Q[:, ρq_rai_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))
            max_q_rai = maximum(abs.(
                Array(Q[:, ρq_rai_ind, :]) ./ Array(Q[:, ρ_ind, :]),
            ))

            @info(min_q_rai, max_q_rai)

            # test that rain variable exists and is not NaN
            @test !isnan(max_q_rai)

            # test that there is reasonable amount of rain water...
            @test abs(max_q_rai) > FT(1e-6)
        end
    end
end

main()
