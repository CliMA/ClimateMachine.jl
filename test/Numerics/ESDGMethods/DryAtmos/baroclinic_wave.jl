using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies: StackedCubedSphereTopology, grid1d
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

include("DryAtmos.jl")

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

struct BaroclinicWave <: AbstractDryAtmosProblem end

function init_state_prognostic!(
    bl::DryAtmosModel,
    ::BaroclinicWave,
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


function main()
    ClimateMachine.init(parse_clargs = true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 3
    numelem_horz = 8
    numelem_vert = 5

    timeend = 10 * 24 * 3600
    outputtime = 24 * 3600

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

    T_profile =
        DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))


    if total_energy
        sources = (Coriolis(),)
    else
        sources = (Coriolis(), Gravity())
    end
    problem = BaroclinicWave()
    model = DryAtmosModel{FT}(
        SphericalOrientation(),
        problem,
        ref_state = DryReferenceState(T_profile),
        sources = sources,
    )

    esdg = ESDGModel(
        model,
        grid,
        #volume_numerical_flux_first_order = CentralVolumeFlux(),
        #volume_numerical_flux_first_order = EntropyConservative(),
        volume_numerical_flux_first_order = KGVolumeFlux(),
        #surface_numerical_flux_first_order = MatrixFlux(),
        surface_numerical_flux_first_order = RusanovNumericalFlux(),
    )

    linearmodel = DryAtmosAcousticGravityLinearModel(model)
    lineardg = DGModel(
        linearmodel,
        grid,
        RusanovNumericalFlux(),
        #CentralNumericalFluxFirstOrder(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        state_auxiliary = esdg.state_auxiliary,
    )

    # determine the time step
    element_size = (domain_height / numelem_vert)
    acoustic_speed = soundspeed_air(param_set, FT(330))

    dx = min_node_distance(grid)
    cfl = 3
    dt = cfl * dx / acoustic_speed

    Q = init_ode_state(esdg, FT(0))

    #odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)

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
    cbcfl = EveryXSimulationSteps(100) do
        simtime = gettime(odesolver)

        @views begin
            ρ = Array(Q.data[:, 1, :])
            ρu = Array(Q.data[:, 2, :])
            ρv = Array(Q.data[:, 3, :])
            ρw = Array(Q.data[:, 4, :])
        end

        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ

        ue = extrema(u)
        ve = extrema(v)
        we = extrema(w)

        @info @sprintf """CFL
                simtime = %.16e
                u = (%.4e, %.4e)
                v = (%.4e, %.4e)
                w = (%.4e, %.4e)
                """ simtime ue... ve... we...
    end
    callbacks = (cbinfo, cbcfl)


    #filterorder = 32
    #filter = ExponentialFilter(grid, 0, filterorder)
    #cbfilter = EveryXSimulationSteps(1) do
    #    Filters.apply!(
    #        Q,
    #        #AtmosFilterPerturbations(model),
    #        :,
    #        grid,
    #        filter,
    #       # state_auxiliary = esdg.state_auxiliary,
    #    )
    #    nothing
    #end
    #callbacks = (callbacks..., cbfilter)

    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_esdg_total_KG_ncg_hires_baroclinic" *
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

    ## Create a callback to report state statistics for main MPIStateArrays
    ## every ntFreq timesteps.
    nt_freq = floor(Int, 1 // 10 * timeend / dt)
    cbsc =
        ClimateMachine.StateCheck.sccreate([(Q, "state")], nt_freq; prec = 12)
    callbacks = (callbacks..., cbsc)

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

    ## Check results against reference if present
    ClimateMachine.StateCheck.scprintref(cbsc)
    #! format: off
    refDat = (
        [
            [ "state",     "ρ",   1.26085994139264381125e-02,  1.51502814805562069367e+00,  3.46330071392505767225e-01,  3.42221017037761976454e-01 ],
            [ "state", "ρu[1]",  -1.26747868050267172180e+02,  1.22735648852872344605e+02, -6.56484249582622303443e-02,  1.07588365672914481053e+01 ],
            [ "state", "ρu[2]",  -1.44635478251794808102e+02,  1.11383888659731638882e+02, -1.53109264073002888581e-03,  1.06376480052955262323e+01 ],
            [ "state", "ρu[3]",  -1.50479775987282266669e+02,  1.62284843398145170568e+02,  5.84596035289077428643e-02,  9.73728925076320095400e+00 ],
            [ "state",    "ρe",   1.73490214503025617887e+03,  2.70352534694924892392e+05,  6.42401799036320589948e+04,  7.10054852130306971958e+04 ],
        ],
        [
            [ "state",     "ρ",    12,    12,    12,    12 ],
            [ "state", "ρu[1]",    12,    12,    12,    12 ],
            [ "state", "ρu[2]",    12,    12,    12,    12 ],
            [ "state", "ρu[3]",    12,    12,    12,    12 ],
            [ "state",    "ρe",    12,    12,    12,    12 ],
        ],
    )
    #! format: on
    if length(refDat) > 0
        @test ClimateMachine.StateCheck.scdocheck(cbsc, refDat)
    end
end


function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "baroclinicwave",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
    auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
    writevtk(
        filename,
        Q,
        dg,
        statenames,
        dg.state_auxiliary,
        auxnames;
        number_sample_points = 10,
    )

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

@testset "$(@__FILE__)" begin
    tic = Base.time()

    main()

    toc = Base.time()
    time = toc - tic
    println(time)
end
