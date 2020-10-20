#!/usr/bin/env julia --project

using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.Atmos
using ClimateMachine.Orientations
using ClimateMachine.ConfigTypes
using ClimateMachine.Diagnostics
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.Mesh.Filters
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VariableTemplates

using Distributions
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMAParameters
using CLIMAParameters.Planet: R_d, cv_d, cp_d, MSLP, grav, LH_v0
using CLIMAParameters.Atmos.SubgridScale: C_smag
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.BalanceLaws:
    vars_state,
    indefinite_stack_integral!,
    reverse_indefinite_stack_integral!,
    integral_load_auxiliary_state!,
    integral_set_auxiliary_state!,
    reverse_integral_load_auxiliary_state!,
    reverse_integral_set_auxiliary_state!

import ClimateMachine.BalanceLaws: boundary_state!
import ClimateMachine.Atmos: flux_second_order!

"""
    Initial Condition for Taylor-Green vortex (LES)

@article{taylorGreen1937,
author = {Taylor, G. I. and Green, A. E.},
title = {Mechanisms of production of small eddies from large ones},
journal = {Proc. Roy. Soc. A},
volume = {158}
number = {895},
year = {1937},
doi={doi.org/10.1098/rspa.1937.0036},
}
@article{rafeiEtAl2018,
author = {Rafei, M.E. and K\"on\"oszy, L. and Rana, Z.},
title = {Investigation of Numerical Dissipation in Classicaland Implicit Large Eddy Simulations,
journal = {Aerospace},
year = {2018},
}
@article{bullJameson2014,
author = {Bull, J.R. and Jameson, A.},
title = {Simulation of the Compressible {Taylor Green} Vortex
using High-Order Flux Reconstruction Schemes},
journal = {AIAA Aviation 7th AIAA theoretical fluid mechanics conference},
year = {2014},
}
"""
function init_greenvortex!(problem, bl, state, aux, localgeo, t)
    (x, y, z) = localgeo.coord

    # Problem float-type
    FT = eltype(state)

    # Unpack constant parameters
    R_gas::FT = R_d(bl.param_set)
    c_p::FT = cp_d(bl.param_set)
    c_v::FT = cv_d(bl.param_set)
    p0::FT = MSLP(bl.param_set)
    _grav::FT = grav(bl.param_set)
    γ::FT = c_p / c_v

    # Compute perturbed thermodynamic state:
    ρ = FT(1.178)      # density
    ρu = SVector(FT(0), FT(0), FT(0))  # momentum
    #State (prognostic) variable assignment
    e_pot = FT(0)# potential energy
    Pinf = 101325
    Uzero = FT(100)
    p = Pinf + (ρ * Uzero / 16) * (2 + cos(z)) * (cos(x) + cos(y))
    u = Uzero * sin(x) * cos(y) * cos(z)
    v = -Uzero * cos(x) * sin(y) * cos(z)
    e_kin = 0.5 * (u^2 + v^2)
    T = p / (ρ * R_gas)
    e_int = internal_energy(bl.param_set, T, PhasePartition(FT(0)))
    ρe_tot = ρ * (e_kin + e_pot + e_int)
    # Assign State Variables
    state.ρ = ρ
    state.ρu = SVector(FT(ρ * u), FT(ρ * v), FT(0))
    state.ρe = ρe_tot
end

# Set up AtmosLESConfiguration for the experiment
function config_greenvortex(
    ::Type{FT},
    N,
    (xmin, xmax, ymin, ymax, zmin, zmax),
    resolution,
) where {FT}
    ode_solver = ClimateMachine.ExplicitSolverType(
        solver_method = LSRK144NiegemannDiehlBusch,
    )

    _C_smag = FT(C_smag(param_set))
    model = AtmosModel{FT}(
        AtmosLESConfigType,                 # Flow in a box, requires the AtmosLESConfigType
        param_set;                          # Parameter set corresponding to earth parameters
        init_state_prognostic = init_greenvortex!,
        ref_state = NoReferenceState(),
        orientation = NoOrientation(),
        turbulence = Vreman(_C_smag),       # Turbulence closure model
        moisture = DryModel(),
        source = (),
    )

    config = ClimateMachine.AtmosLESConfiguration(
        "GreenVortex",          # Problem title [String]
        N,                      # Polynomial order [Int]
        resolution,             # (Δx, Δy, Δz) effective resolution [m]
        xmax,                   # Domain maximum size [m]
        ymax,                   # Domain maximum size [m]
        zmax,                   # Domain maximum size [m]
        param_set,              # Parameter set.
        init_greenvortex!,      # Function specifying initial condition
        boundary = ((0, 0), (0, 0), (0, 0)),
        periodicity = (true, true, true),
        xmin = xmin,
        ymin = ymin,
        zmin = zmin,
        solver_type = ode_solver,       # Time-integrator type
        model = model,                  # Model type
    )
    return config
end

# Define the diagnostics configuration for this experiment
function config_diagnostics(
    driver_config,
    (xmin, xmax, ymin, ymax, zmin, zmax),
    resolution,
    tnor,
    titer,
    snor,
)
    ts_dgngrp = setup_atmos_turbulence_stats(
        AtmosLESConfigType(),
        "360steps",
        driver_config.name,
        tnor,
        titer,
    )

    boundaries = [
        xmin ymin zmin
        xmax ymax zmax
    ]
    interpol = ClimateMachine.InterpolationConfiguration(
        driver_config,
        boundaries,
        resolution,
    )
    ds_dgngrp = setup_dump_spectra_diagnostics(
        AtmosLESConfigType(),
        "0.06ssecs",
        driver_config.name,
        interpol = interpol,
        snor,
    )
    me_dgngrp = setup_atmos_mass_energy_loss(
        AtmosLESConfigType(),
        "0.02ssecs",
        driver_config.name,
    )
    return ClimateMachine.DiagnosticsConfiguration([
        ts_dgngrp,
        ds_dgngrp,
        me_dgngrp,
    ],)
end

# Entry point
function main()
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Ncellsx = 64
    Ncellsy = 64
    Ncellsz = 64
    Δx = FT(2 * pi / Ncellsx)
    Δy = Δx
    Δz = Δx
    resolution = (Δx, Δy, Δz)
    xmin = FT(-pi)
    xmax = FT(pi)
    ymin = FT(-pi)
    ymax = FT(pi)
    zmin = FT(-pi)
    zmax = FT(pi)
    # Simulation time
    t0 = FT(0)
    timeend = FT(0.1)
    CFL = FT(1.8)

    driver_config = config_greenvortex(
        FT,
        N,
        (xmin, xmax, ymin, ymax, zmin, zmax),
        resolution,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        init_on_cpu = true,
        Courant_number = CFL,
    )

    tnor = FT(100)
    titer = FT(0.01)
    snor = FT(10000.0)
    dgn_config = config_diagnostics(
        driver_config,
        (xmin, xmax, ymin, ymax, zmin, zmax),
        resolution,
        tnor,
        titer,
        snor,
    )

    check_cons = (
        ClimateMachine.ConservationCheck("ρ", "100steps", FT(0.0001)),
        ClimateMachine.ConservationCheck("ρe", "100steps", FT(0.0025)),
    )

    result = ClimateMachine.invoke!(
        solver_config;
        diagnostics_config = dgn_config,
        check_cons = check_cons,
        check_euclidean_distance = true,
    )

end

main()
