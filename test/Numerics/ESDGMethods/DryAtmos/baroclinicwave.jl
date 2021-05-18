using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
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

const output_vtk = true

struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet();
#const X = 20
const X = 1
CLIMAParameters.Planet.planet_radius(::EarthParameterSet) = 6.371229e6 / X
CLIMAParameters.Planet.Omega(::EarthParameterSet) = 7.29212e-5 * X
CLIMAParameters.Planet.MSLP(::EarthParameterSet) = 1e5
CLIMAParameters.Planet.grav(::EarthParameterSet) = 9.80616

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

struct BaroclinicWave <: AbstractDryAtmosProblem end

vars_state(::DryAtmosModel, ::BaroclinicWave, ::Auxiliary, FT) = @vars(ωk::FT)

function init_state_prognostic!(bl::DryAtmosModel,
                                ::BaroclinicWave,
                                state, aux, localgeo, t)
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
    φ =  @inbounds asin(coords[3] / norm(coords, 2))
    z =  norm(coords) - _a

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
    state.ρe = ρ * (e_int + e_kin + e_pot)
    nothing
end


function main()
    ClimateMachine.init(parse_clargs=true)
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 3

    numelem_horz = 16
    numelem_vert = 8

    timeend = 15 * 24 * 3600
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
    N,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
)

    Nq = N + 1
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
        polynomialorder = N,
        meshwarp = cubedshellwarp,
    )

    T_profile = DecayingTemperatureProfile{FT}(param_set, FT(290), FT(220), FT(8e3))


    problem = BaroclinicWave()
    model = DryAtmosModel{FT}(SphericalOrientation(),
                              problem,
                              ref_state = DryReferenceState(T_profile),
                              sources = (Coriolis(),)
    )

    esdg = ESDGModel(
        model,
        grid,
        volume_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = MatrixFlux(),
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
    #dt_factor = 1
    #dt = dt_factor * element_size / acoustic_speed / polynomialorder^2
    dx = min_node_distance(grid)
    cfl = 3.0
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
                      """ "$ArrayType" "$FT" N numelem_horz numelem_vert dt eng0

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
            c_v = courant(
                nondiffusive_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                VerticalDirection(),
            )
            c_h = courant(
                nondiffusive_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                HorizontalDirection(),
            )
            ca_v = courant(
                advective_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                VerticalDirection(),
            )
            ca_h = courant(
                advective_courant,
                esdg,
                model,
                Q,
                dt,
                simtime,
                HorizontalDirection(),
            )

            @info @sprintf """CFL
                              simtime = %.16e
                              Acoustic (vertical) Courant number    = %.2g
                              Acoustic (horizontal) Courant number  = %.2g
                              Advection (vertical) Courant number   = %.2g
                              Advection (horizontal) Courant number = %.2g
                              """ simtime c_v c_h ca_v ca_h
    end

    times = FT[]
    pmin = FT[]
    vmax = FT[]

    cb_vel_p = EveryXSimulationSteps(100) do
            γ = FT(gamma(param_set))
            simtime = gettime(odesolver)
            push!(times, simtime)
            
            ρ = Array(view(Q.data, :, 1, :))
            ρu = Array(view(Q.data, :, 2, :))
            ρv = Array(view(Q.data, :, 3, :))
            ρw = Array(view(Q.data, :, 4, :))
            ρe = Array(view(Q.data, :, 5, :))
            Φ = Array(view(esdg.state_auxiliary.data, :, 1, :))
            
            vel = @. sqrt((ρu ^ 2 + ρv ^ 2 + ρw ^ 2) / ρ ^ 2)
            push!(vmax, maximum(vel))

            p = @. (γ - 1) * (ρe - (ρu ^ 2 + ρv ^ 2 + ρw ^ 2) / (2 * ρ) - ρ * Φ)
            psurf = @view p[1:Nq^2, 1:numelem_vert:end]
            push!(pmin, minimum(psurf))
    end
    callbacks = (cbinfo, cbcfl, cb_vel_p)

    if output_vtk
        # vorticity stuff
        ω = similar(Q; vars = @vars(ω::SVector{3, FT}), nstate = 3)
        vort_model = VorticityModel()
        vort_dg = DGModel(
            vort_model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )
        vortQ = init_ode_state(vort_dg, FT(0))
        ∇Φ1 = view(esdg.state_auxiliary.data, :, 2, :)
        ∇Φ2 = view(esdg.state_auxiliary.data, :, 3, :)
        ∇Φ3 = view(esdg.state_auxiliary.data, :, 4, :)
        
        # create vtk dir
        vtkdir =
            "vtk_esdg_baroclinic" *
            "_poly$(N)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, Nq)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1

            vort_dg.state_auxiliary.data .= @view Q.data[:, 1:4, :]
            vort_dg(ω, vortQ, nothing, FT(0))
            
            _grav = FT(grav(param_set))
            ω1 = view(ω.data, :, 1, :)
            ω2 = view(ω.data, :, 2, :)
            ω3 = view(ω.data, :, 3, :)
            esdg.state_auxiliary.data[:, end, :]  .= @. (∇Φ1 * ω1 + ∇Φ2 * ω2 + ∇Φ3 * ω3) / _grav

            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, Nq)
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
    open(joinpath(vtkdir, "timeseries.txt"), "w") do f
      msg = ""
      for i in 1:length(times)
        msg *= @sprintf("%.16e %.16e %.16e\n", times[i], pmin[i], vmax[i])
      end
      write(f, msg)
    end


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
    Nq,
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
    writevtk(filename, Q, dg, statenames, dg.state_auxiliary, auxnames;
             number_sample_points=2 * Nq)

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
