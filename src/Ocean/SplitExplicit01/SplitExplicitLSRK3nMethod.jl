export SplitExplicitLSRK3nSolver

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using StaticArrays
using ...SystemSolvers
using ...MPIStateArrays: array_device, realview
using ...GenericCallbacks
#using Printf

using ...ODESolvers:
    AbstractODESolver, LowStorageRungeKutta3N, update!, updatedt!, getdt
import ...ODESolvers: dostep!

using ...BalanceLaws:
    tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

LSRK3N = LowStorageRungeKutta3N

@doc """
    SplitExplicitLSRK3nSolver(slow_solver, fast_solver; dt, t0 = 0, coupled = true)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q_fast} = f_fast(Q_fast, Q_slow, t)
  \\dot{Q_slow} = f_slow(Q_slow, Q_fast, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This method performs an operator splitting to timestep the Sea-Surface elevation
and vertically averaged horizontal velocity of the model at a faster rate than
the full model, using LowStorageRungeKutta3N time-stepping.

""" SplitExplicitLSRK3nSolver
mutable struct SplitExplicitLSRK3nSolver{SS, FS, RT, MSA, MSB} <:
               AbstractODESolver
    "slow solver"
    slow_solver::SS
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT
    "elapsed time steps"
    steps::Int
    "storage for transfer tendency"
    dQ2fast::MSA
    "saving original fast state"
    S_fast::MSB

    function SplitExplicitLSRK3nSolver(
        slow_solver::LSRK3N,
        fast_solver,
        Q = nothing;
        dt = getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))

        dQ2fast = similar(slow_solver.dQ)
        dQ2fast .= -0.0
        #- Warning: Number of fast-solution to save (here only 1, in S_fast) should be as
        #   large as number of non zero Butcher Coeff (including Weight "b") below 2 diagonal,
        #   i.e., with ns= Number of Stages and a(ns+1,:) = b(:), all non zero a(i,j)|_{i > j+1}.
        #  Saving only 1 fast-solution workd for LS3NRK33Heuns.
        S_fast = similar(fast_solver.dQ)
        S_fast .= -0.0
        MSA = typeof(dQ2fast)
        MSB = typeof(S_fast)
        return new{SS, FS, RT, MSA, MSB}(
            slow_solver,
            fast_solver,
            RT(dt),
            RT(t0),
            0,
            dQ2fast,
            S_fast,
        )
    end
end

function dostep!(
    Qvec,
    split::SplitExplicitLSRK3nSolver{SS},
    param,
    time::Real,
) where {SS <: LSRK3N}
    slow = split.slow_solver
    fast = split.fast_solver

    Qslow = Qvec.slow
    Qfast = Qvec.fast

    dQslow = slow.dQ
    dRslow = slow.dR
    dQ2fast = split.dQ2fast
    S_fast = split.S_fast

    slow_bl = slow.rhs!.balance_law
    fast_bl = fast.rhs!.balance_law

    rv_Q = realview(Qslow)
    rv_dQ = realview(dQslow)
    rv_dR = realview(dRslow)
    groupsize = 256

    slow_dt = getdt(slow)
    fast_dt_in = getdt(fast)
    rkA = slow.RKA
    rkB = slow.RKB
    rkC = slow.RKC
    rkW = slow.RKW
    nStages = length(rkC)

    rv_dR .= -0
    for s in 1:nStages
        # Current slow state time
        slow_stage_time = time + rkC[s] * slow_dt
        # @printf("-- main dostep! stage s=%3i , t= %10.2f\n",s,slow_stage_time)

        # Initialize fast model and set time-step and number of substeps we need
        fast_steps = [0 0 0]
        FT = typeof(slow_dt)
        fast_time_rec = [fast_dt_in FT(0) FT(0)]
        set_fast_for_stepping!(
            slow_bl,
            fast_bl,
            fast.rhs!,
            Qfast,
            S_fast,
            slow_dt,
            rkC,
            rkW,
            s,
            nStages,
            fast_time_rec,
            fast_steps,
        )
        # Initialize tentency adjustment before evaluation of slow mode
        initialize_adjustment!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
        )

        # Evaluate the slow mode
        # --> save tendency for the fast
        slow.rhs!(dQ2fast, Qslow, param, slow_stage_time, increment = false)

        # vertically integrate slow tendency to advance fast equation
        # and use vertical mean for slow model (negative source)
        # ---> work with dQ2fast as input
        tendency_from_slow_to_fast!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            dQ2fast,
        )

        # Compute (and RK update) slow tendency
        slow.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)

        # Update (RK-stage) slow state
        event = Event(array_device(Qslow))
        event = update!(array_device(Qslow), groupsize)(
            rv_dQ,
            rv_dR,
            rv_Q,
            rkA[s % nStages + 1, 1],
            rkA[s % nStages + 1, 2],
            rkB[s, 1],
            rkB[s, 2],
            slow_dt,
            nothing,
            nothing,
            nothing;
            ndrange = length(rv_Q),
            dependencies = (event,),
        )
        wait(array_device(Qslow), event)

        # Determine number of substeps we need
        fast_dt = fast_time_rec[1]
        nsubsteps = fast_steps[3]

        updatedt!(fast, fast_dt)

        for substep in 1:nsubsteps
            fast_time = time + fast_time_rec[3] + (substep - 1) * fast_dt
            # @printf("-- main dostep! substep=%3i , t= %10.2f\n",substep,fast_time)
            dostep!(Qfast, fast, param, fast_time)

            # cumulate fast solution
            cummulate_fast_solution!(
                fast_bl,
                fast.rhs!,
                Qfast,
                fast_time,
                fast_dt,
                substep,
                fast_steps,
                fast_time_rec,
            )
        end

        # reconcile slow equation using fast equation
        reconcile_from_fast_to_slow!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            fast_time_rec;
            lastStage = (s == nStages),
        )

    end

    # reset fast time-step to original value
    updatedt!(fast, fast_dt_in)

    # now do implicit mixing step
    nImplSteps = slow_bl.numImplSteps
    if nImplSteps > 0
        # 1. get implicit mising model, model state variable array and solver handles
        ivdc_dg = slow.rhs!.modeldata.ivdc_dg
        ivdc_bl = ivdc_dg.balance_law
        ivdc_Q = slow.rhs!.modeldata.ivdc_Q
        ivdc_solver = slow.rhs!.modeldata.ivdc_bgm_solver
        # ivdc_solver_dt = getdt(ivdc_solver) # would work if solver time-step was set
        # FT = typeof(slow_dt)
        # ivdc_solver_dt = slow_dt / FT(nImplSteps) # just recompute time-step
        ivdc_solver_dt = ivdc_bl.parent_om.ivdc_dt
        # println("ivdc_solver_dt = ",ivdc_solver_dt )
        # 2. setup start RHS, initial guess and values for computing mixing coeff
        ivdc_Q.θ .= Qslow.θ
        ivdc_RHS = slow.rhs!.modeldata.ivdc_RHS
        ivdc_RHS.θ .= Qslow.θ
        ivdc_RHS.θ .= ivdc_RHS.θ ./ ivdc_solver_dt
        ivdc_dg.state_auxiliary.θ_init .= ivdc_Q.θ
        # 3. Invoke iterative solver

        println("BEFORE maximum(ivdc_Q.θ[:]): ", maximum(ivdc_Q.realdata[:]))
        println("BEFORE minimum(ivdc_Q.θ[:]): ", minimum(ivdc_Q.realdata[:]))

        lm!(y, x) = ivdc_dg(y, x, nothing, 0; increment = false)
        solve_tot = 0
        iter_tot = 0
        for i in 1:nImplSteps
            solve_time = @elapsed iters =
                linearsolve!(lm!, ivdc_solver, ivdc_Q, ivdc_RHS)
            solve_tot = solve_tot + solve_time
            iter_tot = iter_tot + iters
            # Set new RHS and initial values
            ivdc_RHS.θ .= ivdc_Q.θ ./ ivdc_solver_dt
            ivdc_dg.state_auxiliary.θ_init .= ivdc_Q.θ
        end
        println("solver iters, time: ", iter_tot, ", ", solve_tot)

        println("AFTER  maximum(ivdc_Q.θ[:]): ", maximum(ivdc_Q.realdata[:]))
        println("AFTER  minimum(ivdc_Q.θ[:]): ", minimum(ivdc_Q.realdata[:]))

        # exit()
        # Now update
        Qslow.θ .= ivdc_Q.θ

    end

    return nothing
end
