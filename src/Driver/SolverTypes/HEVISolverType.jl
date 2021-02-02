
export HEVISolverType

"""
# Description
    HEVISolverType(FT;
        solver_method = ARK2ImplicitExplicitMidpoint,

        linear_max_subspace_size = Int(30)
        linear_atol = FT(-1.0)
        linear_rtol = FT(5e-5)

        nonlinear_max_iterations = Int(10)
        nonlinear_rtol = FT(1e-4)
        nonlinear_ϵ = FT(1.e-10)
        preconditioner_update_freq = Int(50)
    )

This solver type constructs a solver for ODEs with the
additively horizontal explicit vertical explicit~(HEVI) partitioned form. 
the equation is assumed to be decomposed as

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t) - l(Q, t)]
```

where `Q` is the state, `f` is the full tendency and `l` is the vertical implicit
operator. The splitting is done automatically.

# Arguments
- `solver_method` (Function): Function defining the particular additive
    Runge-Kutta method to be used for the HEVI method.
    Default: `ARK2ImplicitExplicitMidpoint`
- `linear_max_subspace_size` (Int): maximal dimension of each (batched)
    Krylov subspace. GEMRES, a iterative linear solver is applied 
    Default: `30`
- `linear_atol` (FT): absolute tolerance for linear solver convergence.
    Default: `-1.0`
- `linear_rtol` (FT): relative tolerance for linear solver convergence.
    Default: `5.0e-5`
- `nonlinear_max_iterations` (Int): max number of Newton iterations
    Default: `10`
- `nonlinear_rtol` (FT): relative tolerance for nonlinear solver convergence.
    Default: `1e-4`
- `nonlinear_ϵ` (FT): parameter denoting finite different step size for the 
   Jacobian approximation.
   Default: `1e-10`
- `preconditioner_update_freq` (Int): Int denoting how frequent you need 
    to update the preconditioner 
    -1: no preconditioner;
    positive number, update every freq times.
    Default: `50`
"""
struct HEVISolverType{FT} <: AbstractSolverType
    # Function for the HEVI method
    solver_method::Function

    linear_max_subspace_size::Int
    linear_atol::FT
    linear_rtol::FT

    nonlinear_max_iterations::Int
    nonlinear_rtol::FT
    nonlinear_ϵ::FT

    preconditioner_update_freq::Int

    function HEVISolverType(
        FT;
        solver_method = ARK2GiraldoKellyConstantinescu,
        linear_max_subspace_size = Int(30),
        linear_atol = FT(-1.0),
        linear_rtol = FT(5e-5),
        nonlinear_max_iterations = Int(10),
        nonlinear_rtol = FT(1e-4),
        nonlinear_ϵ = FT(1.e-10),
        preconditioner_update_freq = Int(50),
    )

        return new{FT}(
            solver_method,
            linear_max_subspace_size,
            linear_atol,
            linear_rtol,
            nonlinear_max_iterations,
            nonlinear_rtol,
            nonlinear_ϵ,
            preconditioner_update_freq,
        )
    end
end

"""
    getdtmodel(ode_solver::HEVISolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
function getdtmodel(ode_solver::HEVISolverType, bl)
    # Most restrictive dynamics are treated implicitly
    return bl
end

"""
# Description
    solversetup(
        ode_solver::HEVISolverType{FT},
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

Creates an ODE solver using a HEVI-type time-integration
scheme. All horizontal acoustic waves are treated explicitly,
while the 1-D vertical problem is treated implicitly. All
other dynamics (advection, diffusion) is treated explicitly
in the additive Runge-Kutta method.

# Comments:
Currently, the only HEVI-type splitting ClimateMachine can currently
do only involves splitting the acoustic processes; it is not currently
possible to perform more fine-grained separation of tendencies
(for example, including vertical advection or diffusion in the 1-D implicit problem)
"""
function solversetup(
    ode_solver::HEVISolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)


    # All we need to do is create a DGModel for the
    # vertical acoustic waves (determined from the `implicit_model`)
    vdg = DGModel(
        dg.balance_law,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient;
        state_auxiliary = dg.state_auxiliary,
        direction = VerticalDirection(),
        diffusion_direction = VerticalDirection(),
    )

    # linear solver relative tolerance rtol which should be slightly smaller than the nonlinear solver tol
    linearsolver = BatchedGeneralizedMinimalResidual(
        dg,
        Q;
        max_subspace_size = ode_solver.linear_max_subspace_size,
        atol = ode_solver.linear_atol,
        rtol = ode_solver.linear_rtol,
    )

    # N(q)(Q) = Qhat  => F(Q) = N(q)(Q) - Qhat
    # 
    # F(Q) == 0
    # ||F(Q^i) || / ||F(Q^0) || < tol
    # ϵ is a sensity parameter for this problem, it determines the finite difference Jacobian dF = (F(Q + ϵdQ) - F(Q))/ϵ
    # I have also try larger tol, but tol = 1e-3 does not work
    nonlinearsolver = JacobianFreeNewtonKrylovSolver(
        Q,
        linearsolver;
        tol = ode_solver.nonlinear_rtol,
        ϵ = ode_solver.nonlinear_ϵ,
        M = ode_solver.nonlinear_max_iterations,
    )

    # this is a second order time integrator, to change it to a first order time integrator
    # change it ARK1ForwardBackwardEuler, which can reduce the cost by half at the cost of accuracy 
    # and stability
    # preconditioner_update_freq = 50 means updating the preconditioner every 50 Newton solves, 
    # update it more freqent will accelerate the convergence of linear solves, but updating it 
    # is very expensive
    solver = ode_solver.solver_method(
        dg,
        vdg,
        NonLinearBackwardEulerSolver(
            nonlinearsolver;
            isadjustable = true,
            preconditioner_update_freq = ode_solver.preconditioner_update_freq,
        ),
        Q;
        dt = dt,
        t0 = t0,
        split_explicit_implicit = false,
        variant = NaiveVariant(),
    )


    return solver
end
