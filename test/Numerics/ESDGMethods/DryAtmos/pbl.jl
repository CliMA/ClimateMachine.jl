using MPI
using ClimateMachine
using Logging
using ClimateMachine.DGMethods: ESDGModel, init_ode_state
using ClimateMachine.Mesh.Topologies: StackedBrickTopology
using ClimateMachine.Mesh.Grids: DiscontinuousSpectralElementGrid, min_node_distance
using ClimateMachine.Thermodynamics
using LinearAlgebra
using Printf
using Dates
using ClimateMachine.GenericCallbacks:
    EveryXWallTimeSeconds, EveryXSimulationSteps
using ClimateMachine.VTK: writevtk, writepvtu
using ClimateMachine.VariableTemplates
using ClimateMachine.ODESolvers
using StaticArrays: @SVector
using LazyArrays

using DoubleFloats
using GaussQuadrature
GaussQuadrature.maxiterations[Double64] = 40

using ClimateMachine.TemperatureProfiles: DryAdiabaticProfile

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

const output = parse(Bool, lowercase(get(ENV, "JULIA_CLIMA_OUTPUT", "false")))

include("DryAtmos.jl")
include("../diagnostics.jl")
include("pbl_diagnostics.jl")

import CLIMAParameters
CLIMAParameters.Planet.grav(::EarthParameterSet) = 10

struct HeatFlux end
function source!(
    m::DryAtmosModel,
    ::HeatFlux,
    source,
    state,
    aux,
)
    FT = eltype(aux)

    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)
    _MSLP::FT = MSLP(param_set)

    ρ = state.ρ
    ρu = state.ρu
    ρe = state.ρe
    Φ = aux.Φ

    z = Φ / _grav

    p = pressure(ρ, ρu, ρe, Φ)

    h0 = FT(1 / 100)
    hscale = FT(25)
    hflux = h0 * exp(-z / hscale) / hscale
    exner = (p / _MSLP) ^ (_R_d / _cp_d)
    source.ρe += _cp_d * ρ * hflux * exner
    
    return nothing
end

struct Absorber{FT}
  tau::FT
  zabs::FT
  ztop::FT
end
function source!(
    m::DryAtmosModel,
    absorber::Absorber,
    source,
    state,
    aux,
)
    FT = eltype(aux)
    _grav::FT = grav(param_set)
    
    z = aux.Φ / _grav

    tau = absorber.tau
    zabs = absorber.zabs
    ztop = absorber.ztop

    zeps = FT(1e-4)
    if z >= (zabs + zeps)
      α = (z - zabs) / (ztop - zabs) / tau
    else
      α = FT(0)
    end
    source.ρu += -α * state.ρu
    source.ρe += -α * (state.ρe - aux.ref_state.ρe)
    return nothing
end

struct Drag end
function drag_source!(m::DryAtmosModel, ::Drag,
                      source::Vars, state::Vars, state_bot::Vars, aux::Vars)
    FT = eltype(source)
    _grav::FT = grav(param_set)
    Φ = aux.Φ
    z = Φ / _grav

    c0 = FT(1 / 10)
    hscale = FT(25)
    u0 = state_bot.ρu / state_bot.ρ
    v0 = @inbounds sqrt(u0[1] ^ 2 + u0[2] ^ 2)
    ρu = state.ρu
    ρu_drag = @inbounds SVector(ρu[1], ρu[2], 0)
    u = ρu / state.ρ

    S_ρu = -c0 * v0 * ρu_drag * exp(-z / hscale) / hscale
    source.ρu +=  S_ρu
    source.ρe += u' * S_ρu
end


Base.@kwdef struct PBL{FT} <: AbstractDryAtmosProblem
    domain_length::FT = 3200
    domain_height::FT = 1500
    θ0::FT = 300
    zm::FT = 500
    st::FT = 1e-4 / grav(param_set)
end

function init_state_prognostic!(bl::DryAtmosModel, 
                                setup::PBL,
                                state, aux, localgeo, t)

    FT = eltype(state)
    (x, y, z) = localgeo.coord
    
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _cp_d::FT = cp_d(param_set)

    θ0 = setup.θ0
    zm = setup.zm
    st = setup.st
    
    ρ = aux.ref_state.ρ
    p = aux.ref_state.p

    rnd = rand() - FT(0.5)
    zeps = FT(1e-4)
    if z <= zm - zeps
      fac = rnd * (1 - z / zm)
    else
      fac = FT(0)
    end
    δθ = FT(0.001) * fac
    δT = δθ * (p / _MSLP) ^ (_R_d / _cp_d)
    δw = FT(0.2) * fac
    δe = δw ^ 2 / 2 + _cv_d * δT

    state.ρ = ρ
    state.ρu = ρ * SVector(0, 0, δw)
    state.ρe = aux.ref_state.ρe + ρ * δe
end

struct PBLProfile{S}
  setup::S
end
function (prof::PBLProfile)(param_set, z)
     FT = typeof(z)
     setup = prof.setup
    _MSLP::FT = MSLP(param_set)
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _cp_d::FT = cp_d(param_set)
    _grav::FT = grav(param_set)

    θ0 = setup.θ0
    zm = setup.zm
    st = setup.st
  
    α = _grav / (_cp_d * θ0)
    β = _cp_d / _R_d
    θ = θ0 * (z <= zm ? 1 : 1 + (z - zm) * st)
    zeps = FT(1e-4)
    if z <= zm - zeps
      p = _MSLP * (1 - α * z) ^ β
    else
      p = _MSLP * (1 - α * (zm + log(1 + st * (z - zm)) / st)) ^ β
    end
    T = θ * (p / _MSLP) ^ (_R_d / _cp_d)
    
    T, p
end


function main()
    ClimateMachine.init()
    mpicomm = MPI.COMM_WORLD
    ArrayType = ClimateMachine.array_type()
    FT = Float64

    polynomialorder = 5
    numelem_horz = 6
    numelem_vert = 6

    timeend = 15000
    outputtime = 200

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
    setup = PBL{FT}()

    horz_range = range(FT(0),
                       stop = setup.domain_length,
                       length = numelem_horz+1)
    vert_range = range(FT(0),
                       stop = setup.domain_height,
                       length = numelem_vert+1)
    brickrange = (horz_range, horz_range, vert_range)
    topology = StackedBrickTopology(mpicomm,
                                    brickrange,
                                    periodicity=(true, true, false))

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
    )

    dim = 3

    absorber = Absorber(FT(1020), FT(1000), FT(setup.domain_height))
    model = DryAtmosModel{dim}(
        FlatOrientation(),
        setup,
        sources = (HeatFlux(), absorber),
        drag_source=Drag(),
        ref_state=DryReferenceState(PBLProfile(setup))
    )

    esdg = ESDGModel(
        model,
        grid;
        volume_numerical_flux_first_order = EntropyConservative(),
        #surface_numerical_flux_first_order = EntropyConservative(),
        surface_numerical_flux_first_order = MatrixFlux(),
    )

    # determine the time step
    dx = min_node_distance(grid)
    acoustic_speed = soundspeed_air(param_set, FT(setup.θ0))
    cfl = FT(1.5)
    dt = cfl * dx / acoustic_speed

    Q = init_ode_state(esdg, FT(0); init_on_cpu=true)

    η = similar(Q, vars = @vars(η::FT), nstate=1)

    ∫η0 = entropy_integral(esdg, η, Q)

    #η_int = function(dg, Q1)
    #  entropy_integral(dg, η, Q1)
    #end
    #η_prod = function(dg, Q1, Q2)
    #  entropy_product(dg, η, Q1, Q2)
    #end
    #odesolver = RLSRK144NiegemannDiehlBusch(esdg, η_int, η_prod, Q; dt = dt, t0 = 0)

    odesolver = LSRK144NiegemannDiehlBusch(esdg, Q; dt = dt, t0 = 0)
    

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      ∫η              = %.16e
                      """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0 ∫η0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = EveryXSimulationSteps(1000) do (s = false)
        if s
            starttime[] = now()
        else
            ∫η = entropy_integral(esdg, η, Q)
            dη = (∫η - ∫η0) / abs(∫η0)
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime            = %.16e
                              runtime            = %s
                              norm(Q)            = %.16e
                              ∫η                 = %.16e
                              (∫η - ∫η0) / |∫η0| = %.16e 
                              """ gettime(odesolver) runtime energy ∫η dη
        end
    end
    cbcheck = EveryXSimulationSteps(1000) do
        @views begin
          ρ = Array(Q.data[:, 1, :])
          ρu = Array(Q.data[:, 2, :])
          ρv = Array(Q.data[:, 3, :])
          ρw = Array(Q.data[:, 4, :])
        end

        u = ρu ./ ρ
        v = ρv ./ ρ
        w = ρw ./ ρ

        ω = Array(Q.weights)
        ekin = (u .^ 2 + v .^ 2 + w .^ 2) ./ 2

        ∫ekin = sum(ω .* ekin)

        @info "u = $(extrema(u))"
        @info "v = $(extrema(v))"
        @info "w = $(extrema(w))"
        @info "∫ekin = $(∫ekin)"
        nothing
    end

    callbacks = (cbinfo, cbcheck)

    output_vtk = false
    if output_vtk
        # create vtk dir
        Nelem = Ne[1]
        vtkdir =
            "test_RTB" *
            "_poly$(polynomialorder)_dims$(dim)_$(ArrayType)_$(FT)_nelem$(Nelem)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)

        # setup the output callback
        outputtime = 50
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model)
        end
        callbacks = (callbacks..., cbvtk)
    end


    nsteps = ceil(Int, timeend / dt)
    prof_steps = 0
    mkpath("esdg_pbl_profs")
    diagnostic_vars = pbl_diagnostic_vars(FT)
    state_diagnostic = similar(Q;
                               vars = pbl_diagnostic_vars(FT),
                               nstate=varsize(diagnostic_vars))
    cbdiagnostics = EveryXSimulationSteps(1) do
      step = getsteps(odesolver)
      if mod(step, 10000) == 0 || step > (nsteps - 50)
        prof_steps += 1
        nodal_diagnostics!(pbl_diagnostics!, diagnostic_vars, 
                           esdg, state_diagnostic, Q)
        variance_pairs = ((:θ, :θ), (:w, :θ), (:w, :w))
        z, profs, variances = profiles(diagnostic_vars, variance_pairs, esdg, state_diagnostic)

        s = @sprintf "z θ w θxθ wxθ, wxw\n"
        for k in 1:length(profs.θ)
          s *= @sprintf("%.16e %.16e %.16e %.16e %.16e %.16e\n",
                        z[k],
                        profs.θ[k],
                        profs.w[k],
                        variances.θxθ[k],
                        variances.wxθ[k],
                        variances.wxw[k])
        end
        open("esdg_pbl_profs/pbl_profiles_$step.txt", "w") do f
          write(f, s)
        end
      end
    end
    callbacks = (callbacks..., cbdiagnostics)

    solve!(Q, odesolver; callbacks = callbacks, timeend = timeend)

    # final statistics
    engf = norm(Q)
    ∫ηf = entropy_integral(esdg, η, Q)
    dηf = (∫ηf - ∫η0) / abs(∫η0)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    ∫η                      = %.16e
    (∫η - ∫η0) / |∫η0|      = %.16e 
    """ engf engf / eng0 engf - eng0 ∫ηf dηf
    engf
end

function do_output(mpicomm, vtkdir, vtkstep, esdg, Q, model, testname = "RTB")
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

    writevtk(filename, Q, esdg, statenames, esdg.state_auxiliary, auxnames)#; number_sample_points = 10)

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
