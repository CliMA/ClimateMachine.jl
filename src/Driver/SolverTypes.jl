"""
    CLIMA solver types.
"""

using ..Atmos
using ..DGmethods
using ..Mesh.Grids
using ..ODESolvers

# export AbstractSolverType
# export ExplicitSolverType, IMEXSolverType, MultirateSolverType
# export setup_solver

"""
    AbstractSolverType

This is an abstract type representing a ODE solver configuration.
"""
abstract type AbstractSolverType end

"""
    setup_solver(solver_type::AbstractSolverType)

Returns an ODESolver implementation of the desired AbstractSolverType.
"""
function setup_solver(solver_type::AbstractSolverType, args...) end

"""
    ExplicitSolverType

Solver type for explicit ODE solvers.
"""
struct ExplicitSolverType <: AbstractSolverType
    linear_model::Type
    solver_method::Function
    function ExplicitSolverType(;linear_model=nothing,
                                 solver_method=LSRK54CarpenterKennedy)
        return new(solver_method, linear_model)
    end
end

"""
    setup_solver(solver_type::ExplicitSolverType)

Returns an ODESolver implementation of the desired ExplicitSolverType.
"""
@inline function setup_solver(solver_type::ExplicitSolverType,
                      dg::DGModel, Q::MPIStateArray, t0, dt;
                      linmodel=nothing)
    return solver_type.solver_method(dg, Q; dt=dt, t0=t0)
end

"""
    IMEXSolverType

Solver type for 1D-IMEX ODE solvers.
"""
struct IMEXSolverType <: AbstractSolverType
    linear_model::Type
    linear_solver::Type
    solver_method::Function
    function IMEXSolverType(;linear_model=AtmosAcousticGravityLinearModel,
                             linear_solver=ManyColumnLU,
                             solver_method=ARK2GiraldoKellyConstantinescu)
        return new(linear_model, linear_solver, solver_method)
    end
end

"""
    setup_solver(solver_type::IMEXSolverType)

Returns an ODESolver implementation of the desired IMEXSolverType.
"""
@inline function setup_solver(solver_type::ExplicitSolverType,
                      dg::DGModel, Q::MPIStateArray, t0, dt;
                      linmodel=nothing)
    if linmodel == nothing
        linmodel = solver_type.linear_model(dg.balancelaw)
    end

    vdg = DGModel(linmodel, dg.grid, dg.numfluxnondiff, dg.numfluxdiff, dg.gradnumflux,
                  auxstate=dg.auxstate, direction=VerticalDirection())
    solver = solver_type.solver_method(dg, vdg, solver_type.linear_solver(), Q;
                                       dt=dt, t0=t0)
    return solver
end

"""
    MultirateSolverType

Solver type for a two-rate multirate scheme.

The fast and slow solvers are of type: `AbstractSolverType`.
"""
struct MultirateSolverType <: AbstractSolverType
    solver_method::Type
    slow_method::Type
    fast_method::Type
    linear_model::Type
    numsubsteps::Int
    function MultirateSolverType(;solver_method=MultirateRungeKutta,
                 fast_method=ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
                 slow_method=ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
                 linear_model=AtmosAcousticGravityLinearModel,
                 numsubsteps=50)
      return new(solver_method, fast_method, slow_method, linear_model, numsubsteps)
    end
end

"""
    setup_solver(solver_type::MultirateSolverType)

Returns an ODESolver implementation of the desired MultirateSolverType.
"""
@inline function setup_solver(solver_type::MultirateSolverType,
                      dg::DGModel, Q::MPIStateArray, t0, dt;
                      linmodel=nothing)
    if linmodel !== nothing
        fast_model = linmodel
        fast_model = solver_type.linear_model(dg.balancelaw)
    end

    fast_model = solver_type.fast_model(dg.balancelaw)
    slow_model = RemainderModel(dg.balancelaw, (fast_model,))

    fast_dg = DGModel(fast_model, dg.grid, dg.numfluxnondiff, dg.numfluxdiff, dg.gradnumflux,
                      auxstate=dg.auxstate)
    slow_dg = DGModel(slow_model, dg.grid, dg.numfluxnondiff, dg.numfluxdiff, dg.gradnumflux,
                      auxstate=dg.auxstate)

    fast_solver_method = solver_type.fast_method
    slow_solver_method = solver_type.slow_method

    slow_solver = setup_solver(slow_solver_method, slow_dg, Q; t0=t0, dt=dt)
    fast_dt = slow_dt / solver_type.numsubsteps
    fast_solver = setup_solver(fast_solver_method, fast_dg, Q; t0=t0, dt=fast_dt)

    return solver_type.solver_method((slow_solver, fast_solver))
end
