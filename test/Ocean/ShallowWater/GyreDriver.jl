include("GyreInABox.jl")

using Test

if !isempty(ARGS)
    stommel = Bool(parse(Int, ARGS[1]))
    linear = Bool(parse(Int, ARGS[2]))
    test = parse(Int, ARGS[3])
else
    stommel = true
    linear = true
    test = 1
end

###################
# PARAM SELECTION #
###################
const FT = Float64

const τₒ = 2e-4 # value includes τₒ, g, and ρ
const fₒ = 1e-4
const β = 1e-11
const λ = 1e-6
const ν = 1e4

const Lˣ = 1e6
const Lʸ = 1e6
const timeend = 100 * 24 * 60 * 60
const H = 1000
const c = sqrt(grav(param_set) * H)

if stommel
    gyre = "stommel"
else
    gyre = "munk"
end

if linear
    advec = "linear"
else
    advec = "nonlinear"
end

outname = "vtk_new_dt_" * gyre * "_" * advec

function setup_model(FT, stommel, linear, τₒ, fₒ, β, γ, ν, Lˣ, Lʸ, H)
    problem = GyreInABox{FT}(τₒ, fₒ, β, Lˣ, Lʸ, H)

    if stommel
        turbulence = LinearDrag{FT}(λ)
    else
        turbulence = ConstantViscosity{FT}(ν)
    end

    if linear
        advection = nothing
    else
        advection = NonLinearAdvection()
    end

    model = SWModel(param_set, problem, turbulence, advection, c)
end

function shallow_init_state!(
    p::GyreInABox,
    T::TurbulenceClosure,
    state,
    aux,
    coords,
    t,
)
    if t == 0
        null_init_state!(p, T, state, aux, coords, 0)
    else
        gyre_init_state!(p, T, state, aux, coords, t)
    end
end

function shallow_init_aux!(p::GyreInABox, aux, geom)
    @inbounds y = geom.coord[2]

    Lʸ = p.Lʸ
    τₒ = p.τₒ
    fₒ = p.fₒ
    β = p.β

    aux.τ = @SVector [-τₒ * cos(π * y / Lʸ), 0, 0]
    aux.f = @SVector [0, 0, fₒ + β * (y - Lʸ / 2)]

    return nothing
end

#########################
# Timestepping function #
#########################

function run(mpicomm, topl, ArrayType, N, dt, FT, model, test)
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    Qe = init_ode_state(dg, FT(timeend))

    lsrk = LSRK144NiegemannDiehlBusch(dg, Q; dt = dt, t0 = 0)

    cb = ()

    if test > 2
        outprefix = @sprintf("ic_mpirank%04d_ic", MPI.Comm_rank(mpicomm))
        statenames = flattenednames(vars_state_conservative(model, eltype(Q)))
        auxnames = flattenednames(vars_state_auxiliary(model, eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)

        outprefix = @sprintf("exact_mpirank%04d", MPI.Comm_rank(mpicomm))
        statenames = flattenednames(vars_state_conservative(model, eltype(Qe)))
        auxnames = flattenednames(vars_state_auxiliary(model, eltype(Qe)))
        writevtk(outprefix, Qe, dg, statenames, dg.state_auxiliary, auxnames)

        step = [0]
        vtkpath = outname
        mkpath(vtkpath)
        cbvtk = GenericCallbacks.EveryXSimulationSteps(1000) do (init = false)
            outprefix = @sprintf(
                "%s/mpirank%04d_step%04d",
                vtkpath,
                MPI.Comm_rank(mpicomm),
                step[1]
            )
            @debug "doing VTK output" outprefix
            statenames =
                flattenednames(vars_state_conservative(model, eltype(Q)))
            auxnames = flattenednames(vars_state_auxiliary(model, eltype(Q)))
            writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)
            step[1] += 1
            nothing
        end
        cb = (cb..., cbvtk)
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = cb)

    error = euclidean_distance(Q, Qe) / norm(Qe)
    @info @sprintf(
        """Finished
        error = %.16e
        """,
        error
    )

    return error
end


################
# Start Driver #
################

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()
    mpicomm = MPI.COMM_WORLD

    model = setup_model(FT, stommel, linear, τₒ, fₒ, β, λ, ν, Lˣ, Lʸ, H)

    if test == 1
        cellsrange = 10:10
        orderrange = 4:4
        testval = 1.6068814534535144e-03
    elseif test == 2
        cellsrange = 5:5:10
        orderrange = 3:4
    elseif test > 2
        cellsrange = 5:5:25
        orderrange = 6:10
    end

    errors = zeros(FT, length(cellsrange), length(orderrange))
    for (i, Ne) in enumerate(cellsrange)
        brickrange = (
            range(FT(0); length = Ne + 1, stop = Lˣ),
            range(FT(0); length = Ne + 1, stop = Lʸ),
        )
        topl = BrickTopology(mpicomm, brickrange, periodicity = (false, false))

        for (j, N) in enumerate(orderrange)
            @info "running Ne $Ne and N $N with"
            dt = (Lˣ / c) / Ne / N^2
            @info @sprintf("\n dt = %f", dt)
            errors[i, j] = run(mpicomm, topl, ArrayType, N, dt, FT, model, test)
        end
    end

    @test errors[end, end] ≈ testval

    #=
    msg = ""
    for i in length(cellsrange)-1
      rate = log2(errors[i, end] - log2(errors[i+1, end]))
      msg *= @sprintf("\n rate for Ne %d = %e", cellsrange[i], rate)
    end
    @info msg

    msg = ""
    for j in length(orderrange)-1
      rate = log2(errors[end, j] - log2(errors[end, j+1]))
      msg *= @sprintf("\n rate for N  %d = %e", orderrange[j], rate)
    end
    @info msg
    =#

end
