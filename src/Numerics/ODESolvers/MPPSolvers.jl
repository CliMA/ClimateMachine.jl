export MPPSolver
using ..DGMethods: DGModel, mpp_step_initialize!, mpp_update!, mpp_initialize
import ..Mesh.Filters

mutable struct MPPSolver{RT, DGS, MPP} <: AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "elapsed time steps"
    steps::Int
    "DG ode solver"
    dgsolver::DGS
    "DG model"
    dg::DGModel
    "MPP data structure"
    mppdata::MPP
    function MPPSolver(
        mpptargets,
        dg_solver_constructor,
        dg::DGModel,
        state_prognostic;
        dt,
        t0 = 0,
    )
        RT = real(eltype(state_prognostic))
        mppdata = mpp_initialize(dg, state_prognostic, mpptargets)

        Q = (dg_state = state_prognostic, ∫dg_flux = mppdata.∫dg_flux)
        dgsolver = dg_solver_constructor(dg, Q; dt = dt, t0 = t0)

        new{typeof(dt), typeof(dgsolver), typeof(mppdata)}(
            dt,
            t0,
            0,
            dgsolver,
            dg,
            mppdata,
        )
    end
end

function dostep!(state_prognostic, mppsolver::MPPSolver, p, time)
    dg = mppsolver.dg
    mppdata = mppsolver.mppdata
    dgsolver = mppsolver.dgsolver
    dt = mppsolver.dt

    # computes the FVM fluxes from the current state
    mpp_step_initialize!(dg, state_prognostic, mppdata, time)

    #=
    # run the DG solver for 1 time step
    fill!(mppdata.∫dg_flux, 0)
    Q = (dg_state = state_prognostic, ∫dg_flux = mppdata.∫dg_flux)
    dg_dt = getdt(dgsolver)
    updatedt!(dgsolver, dt)
    solve!(
        Q,
        dgsolver,
        (mpptargets = mppdata.target, orig_p = p);
        numberofsteps = 1,
        adjustfinalstep = false,
    )
    updatedt!(dgsolver, dg_dt)

    # Correct the DG solution based on MPP
    mpp_update!(dg, state_prognostic, mppdata, dt)

    Filters.apply!(
        state_prognostic,
        mppdata.target,
        dg.grid,
        Filters.TMARFilter(),
    )
    =#

end
