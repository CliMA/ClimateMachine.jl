using ClimateMachine
ClimateMachine.init()
using ClimateMachine.Atmos
using ClimateMachine.Checkpoint
using ClimateMachine.ConfigTypes
using ClimateMachine.MoistThermodynamics
using ClimateMachine.VariableTemplates
using ClimateMachine.Grids
using ClimateMachine.ODESolvers

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI
using Printf
using StaticArrays
using Test

Base.@kwdef struct AcousticWaveSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function (setup::AcousticWaveSetup)(bl, state, aux, coords, t)
    # callable to set initial conditions
    FT = eltype(state)

    λ = longitude(bl, aux)
    φ = latitude(bl, aux)
    z = altitude(bl, aux)

    β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(FT(π) * β)) / 2
    g = sin(setup.nv * FT(π) * z / setup.domain_height)
    Δp = setup.γ * f * g
    p = aux.ref_state.p + Δp

    ts = PhaseDry_given_pT(bl.param_set, p, setup.T_ref)
    q_pt = PhasePartition(ts)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(ts)

    state.ρ = air_density(ts)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (e_int + e_pot)
    return nothing
end

function main()
    FT = Float64

    # DG polynomial order
    N = 4

    # Domain resolution
    nelem_horz = 4
    nelem_vert = 6
    resolution = (nelem_horz, nelem_vert)

    t0 = FT(0)
    timeend = FT(1800)
    # Timestep size (s)
    dt = FT(600)

    setup = AcousticWaveSetup{FT}()
    T_profile = IsothermalProfile(param_set, setup.T_ref)
    orientation = SphericalOrientation()
    ref_state = HydrostaticState(T_profile)
    turbulence = ConstantViscosityWithDivergence(FT(0))
    model = AtmosModel{FT}(
        AtmosGCMConfigType,
        param_set;
        orientation = orientation,
        ref_state = ref_state,
        turbulence = turbulence,
        moisture = DryModel(),
        source = Gravity(),
        init_state_conservative = setup,
    )

    driver_config = ClimateMachine.AtmosGCMConfiguration(
        "Checkpoint unit tests",
        N,
        resolution,
        setup.domain_height,
        param_set,
        setup;
        model = model,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t0,
        timeend,
        driver_config,
        ode_dt = dt,
    )

    isdir(ClimateMachine.Settings.checkpoint_dir) ||
    mkpath(ClimateMachine.Settings.checkpoint_dir)

    @testset "Checkpoint/restart unit tests" begin
        rm_checkpoint(
            ClimateMachine.Settings.checkpoint_dir,
            solver_config.name,
            solver_config.mpicomm,
            0,
        )
        write_checkpoint(
            solver_config,
            ClimateMachine.Settings.checkpoint_dir,
            solver_config.name,
            solver_config.mpicomm,
            0,
        )

        nm = replace(solver_config.name, " " => "_")
        cname = @sprintf(
            "%s_checkpoint_mpirank%04d_num%04d.jld2",
            nm,
            MPI.Comm_rank(solver_config.mpicomm),
            0,
        )
        cfull = joinpath(ClimateMachine.Settings.checkpoint_dir, cname)
        @test isfile(cfull)

        s_Q, s_aux, s_t = try
            read_checkpoint(
                ClimateMachine.Settings.checkpoint_dir,
                nm,
                driver_config.array_type,
                solver_config.mpicomm,
                0,
            )
        catch
            (nothing, nothing, nothing)
        end
        @test s_Q !== nothing
        @test s_aux !== nothing
        @test s_t !== nothing
        if Array ∉ typeof(s_Q).parameters
            s_Q = Array(s_Q)
            s_aux = Array(s_aux)
        end

        dg = solver_config.dg
        Q = solver_config.Q
        if Array ∈ typeof(Q).parameters
            h_Q = Q.realdata
            h_aux = dg.state_auxiliary.realdata
        else
            h_Q = Array(Q.realdata)
            h_aux = Array(dg.state_auxiliary.realdata)
        end
        t = ODESolvers.gettime(solver_config.solver)

        @test h_Q == s_Q
        @test h_aux == s_aux
        @test t == s_t

        rm_checkpoint(
            ClimateMachine.Settings.checkpoint_dir,
            solver_config.name,
            solver_config.mpicomm,
            0,
        )
        @test !isfile(cfull)
    end
end

main()
