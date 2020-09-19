using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d
using ClimateMachine.Mesh.Grids:
    DiscontinuousSpectralElementGrid, VerticalDirection, min_node_distance
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.DGMethods.NumericalFluxes:
    RusanovNumericalFlux,
    CentralNumericalFluxGradient,
    CentralNumericalFluxSecondOrder
using ClimateMachine.ODESolvers
using ClimateMachine.SystemSolvers
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.Thermodynamics:
    air_density,
    soundspeed_air,
    internal_energy,
    PhaseDry_given_pT,
    PhasePartition
using ClimateMachine.TemperatureProfiles: DecayingTemperatureProfile
using ClimateMachine.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: day, grav, R_d, cp_d, cv_d, Omega, planet_radius, MSLP
import CLIMAParameters

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test
using CUDA

const output_vtk = false

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();
const X = 20
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371e6 / X
CLIMAParameters.Planet.Omega(::EarthParameterSet) = 7.2921159e-5 * X

include("DryAtmos.jl")

function sphr_to_cart_vec(
    vec, lat, lon
)
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

struct HeldSuarez <: AbstractDryAtmosProblem end

# Held Suarez needs coordinates
vars_state_auxiliary(::DryAtmosModel, ::HeldSuarez, FT) = @vars(coord::SVector{3, FT})
function init_state_auxiliary!(
    m::DryAtmosModel,
    ::HeldSuarez,
    state_auxiliary,
    geom,
)
  state_auxiliary.problem.coord = geom.coord
end

function init_state_conservative!(bl::DryAtmosModel,
                                  ::HeldSuarez,
                                  state, aux, coord, t)
    FT = eltype(state)

    # parameters 
    _a::FT = planet_radius(param_set)

    z_t::FT = 15e3
    λ_c::FT = π / 9
    φ_c::FT = 2 * π / 9
    d_0::FT = _a / 6
    V_p::FT = 10

    # grid
    λ = @inbounds atan(coord[2], coord[1])
    φ =  @inbounds asin(coord[3] / norm(coord, 2))
    z =  norm(coord) - _a

    # deterministic velocity perturbation
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
    u_sphere = SVector{3, FT}(u′, v′, w′)
    u_cart = sphr_to_cart_vec(u_sphere, φ, λ)

    ## potential & kinetic energy
    e_kin::FT = 0.5 * u_cart' * u_cart

    ## Assign state variables
    state.ρ = aux.ref_state.ρ
    state.ρu = state.ρ * u_cart
    state.ρe = aux.ref_state.ρe + state.ρ * e_kin
    nothing
end

struct HeldSuarezForcing end

function source!(
    m::DryAtmosModel,
    ::HeldSuarezForcing,
    source,
    state,
    aux,
)
    FT = eltype(state)
    
    _R_d = FT(R_d(param_set))
    _day = FT(day(param_set))
    _grav = FT(grav(param_set))
    _cp_d = FT(cp_d(param_set))
    _cv_d = FT(cv_d(param_set))
    _p0 = FT(MSLP(param_set))

    # Parameters
    T_ref = FT(255)

    # Extract the state
    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ = aux.Φ

    coord = aux.problem.coord
    
    p = pressure(ρ, ρu, ρe, Φ)
    T = p / (ρ * _R_d)

    # Held-Suarez parameters
    k_a = FT(1 / (40 * _day))
    k_f = FT(1 / _day)
    k_s = FT(1 / (4 * _day))
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)
    σ_b = FT(7 / 10)

    # Held-Suarez forcing
    φ = @inbounds asin(coord[3] / norm(coord, 2))

    #TODO: replace _p0 with dynamic surfce pressure in Δσ calculations to account
    #for topography, but leave unchanged for calculations of σ involved in T_equil
    σ = p / _p0
    exner_p = σ^(_R_d / _cp_d)
    Δσ = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil = (T_equator - ΔT_y * sin(φ)^2 - Δθ_z * log(σ) * cos(φ)^2) * exner_p
    T_equil = max(T_min, T_equil)
    k_T = k_a + (k_s - k_a) * height_factor * cos(φ)^4
    k_v = k_f * height_factor

    # horizontal projection
    k = coord / norm(coord)
    P = I - k * k'

    # Apply Held-Suarez forcing
    source.ρu -= k_v * P * ρu
    source.ρe -= k_T * ρ * _cv_d * (T - T_equil)
    return nothing
end


function main()
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 5
    numelem_vert = 5

    timeend = 5 * 43200
    # timeend = 33 * 60 * 60 # Full simulation
    outputtime = 600


    FT = Float64
    result = run(
        mpicomm,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        timeend,
        outputtime,
        ArrayType,
        FT,
    )
end

function run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
)
    _planet_radius::FT = planet_radius(param_set)
    domain_height = FT(30e3)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + domain_height),
        nelem = numelem_vert,
    )
    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = cubedshellwarp,
    )

    T_profile = DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))

    model = DryAtmosModel{FT}(
      SphericalOrientation(),
      HeldSuarez(),
      ref_state = DryReferenceState(T_profile),
      sources = (HeldSuarezForcing(), Coriolis(),)
    )

    esdg = ESDGModel(
        model,
        grid,
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = EntropyConservative(),
    )

    # determine the time step
    element_size = (domain_height / numelem_vert)
    acoustic_speed = soundspeed_air(param_set, FT(330))
    #dt_factor = 1
    #dt = dt_factor * element_size / acoustic_speed / polynomialorder^2
    dx = min_node_distance(grid)
    cfl = 1.0
    dt = cfl * dx / acoustic_speed

    Q = init_ode_state(esdg, FT(0))

    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)

    #filterorder = 18
    #filter = ExponentialFilter(grid, 0, filterorder)
    #cbfilter = EveryXSimulationSteps(1) do
    #    Filters.apply!(Q, :, grid, filter, direction = VerticalDirection())
    #    nothing
    #end

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end
    callbacks = (cbinfo,)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_esdg_heldsuarez" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end

    solve!(
        Q,
        odesolver;
        timeend = timeend,
        adjustfinalstep = false,
        callbacks = callbacks,
    )

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end


function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "heldsuarez",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state_conservative(model, eltype(Q)))
    auxnames = flattenednames(vars_state_auxiliary(model, eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...), eltype(Q))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
