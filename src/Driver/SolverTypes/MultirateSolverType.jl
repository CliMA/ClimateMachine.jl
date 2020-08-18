
export MultirateSolverType

"""
# Description
    MultirateSolverType(;
        splitting_type = SlowFastSplitting(),
        fast_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        implicit_solver_adjustable = false,
        slow_method = LSRK54CarpenterKennedy,
        fast_method = LSRK54CarpenterKennedy,
        timestep_ratio = 100,
    )

This solver type constructs an ODE solver using a standard multirate
Runge-Kutta implementation. This solver computes solutions to ODEs with
the partitioned form:

```math
    \\dot{Q} = f_{fast}(Q, t) + f_{slow}(Q, t)
```

where the right-hand-side functions `f_fast` and `f_slow` denote
fast and slow dynamics respectively, depending on the state `Q`.

# Arguments
- `splitting_type` (DiscreteSplittingType): The type of discrete
    splitting to apply to the right-hand side.
    Default: `SlowFastSplitting()`
- `fast_model` (Type): The model describing fast dynamics.
    Default: `AtmosAcousticGravityLinearModel`
- `implicit_solver` (Type): An implicit solver for inverting the
    implicit system of equations (if using `HEVISplitting()`).
    Default: `ManyColumnLU`
- `implicit_solver_adjustable` (Bool): A flag identifying whether
    or not the `implicit_solver` can be updated as the time-step
    size changes. This is particularly important when using
    an implicit solver within a multirate scheme.
    Default: `false`
- `slow_method` (Function): Function defining the particular explicit
    Runge-Kutta method to be used for the slow processes.
    Default: `LSRK54CarpenterKennedy`
- `fast_method` (Function): Function defining the fast solver.
    Depending on the choice of `splitting_type`, this can be
    an explicit Runge Kutta method or a 1-D IMEX (additive Runge-Kutta)
    method.
    Default: `LSRK54CarpenterKennedy`
- `timestep_ratio` (Int): Integer denoting the ratio between the slow
    and fast time-step sizes.
    Default: `100`
- `discrete_splitting` (Boolean): Boolean denoting whether a PDE level or
    discretized level splitting should be used. If `true` then the PDE is
    discretized in such a way that `f_fast + f_slow` is equivalent to
    discretizing the original PDE directly.

### References
    @article{SchlegelKnothArnoldWolke2012,
        title={Implementation of multirate time integration methods for air
            pollution modelling},
        author={Schlegel, M and Knoth, O and Arnold, M and Wolke, R},
        journal={Geoscientific Model Development},
        volume={5},
        number={6},
        pages={1395--1405},
        year={2012},
        publisher={Copernicus GmbH}
    }
"""
struct MultirateSolverType{DS} <: AbstractSolverType
    # The type of discrete splitting to apply to the right-hand side
    splitting_type::DS
    # The model describing fast dynamics
    fast_model::Type
    # Choice of implicit solver
    implicit_solver::Type
    # Can the implicit solver be updated with changing dt?
    implicit_solver_adjustable::Bool
    # RK method for evaluating the slow processes
    slow_method::Function
    # RK method for evaluating the fast processes
    fast_method::Function
    # The ratio between slow and fast time-step sizes
    timestep_ratio::Int
    # Whether to use a PDE level or discrete splitting
    discrete_splitting::Bool

    function MultirateSolverType(;
        splitting_type = SlowFastSplitting(),
        fast_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        implicit_solver_adjustable = false,
        slow_method = LSRK54CarpenterKennedy,
        fast_method = LSRK54CarpenterKennedy,
        timestep_ratio = 100,
        discrete_splitting = false,
    )

        DS = typeof(splitting_type)

        return new{DS}(
            splitting_type,
            fast_model,
            implicit_solver,
            implicit_solver_adjustable,
            slow_method,
            fast_method,
            timestep_ratio,
            discrete_splitting,
        )
    end
end

"""
    getdtmodel(ode_solver::MultirateSolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
function getdtmodel(ode_solver::MultirateSolverType, bl)
    # Most restrictive dynamics are part of the fast model
    return ode_solver.fast_model(bl)
end

"""
# Description
    function solversetup(
        ode_solver::MultirateSolverType{SlowFastSplitting},
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

Creates an ODE solver for the partition slow-fast ODE
using a multirate method with explicit time-integration.
The splitting of the fast (acoustic and gravity waves)
dynamics is done in _all_ spatial directions. Examples
of a similar implementations include the following
references.

### References
    @article{SchlegelKnothArnoldWolke2012,
        title={Implementation of multirate time integration methods for air
            pollution modelling},
        author={Schlegel, M and Knoth, O and Arnold, M and Wolke, R},
        journal={Geoscientific Model Development},
        volume={5},
        number={6},
        pages={1395--1405},
        year={2012},
        publisher={Copernicus GmbH}
    }

    @article{Schlegel_2009,
        doi = {10.1016/j.cam.2008.08.009},
        year = {2009},
        month = {apr},
        publisher = {Elsevier {BV}},
        volume = {226},
        number = {2},
        pages = {345--357},
        author = {Martin Schlegel and Oswald Knoth and Martin Arnold and Ralf Wolke},
        title = {Multirate Runge-Kutta schemes for advection equations},
        journal = {Journal of Computational and Applied Mathematics}
    }
"""
function solversetup(
    ode_solver::MultirateSolverType{SlowFastSplitting},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    # Extract fast model and define a DG model
    # for the fast processes (acoustic/gravity waves
    # in all spatial directions)
    fast_model = ode_solver.fast_model(dg.balance_law)
    fast_dg = DGModel(
        fast_model,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = EveryDirection(),
    )

    # Using the RemainderModel, we subtract away the
    # fast processes and define a DG model for the
    # slower processes (advection and diffusion)
    if ode_solver.discrete_splitting
        numerical_flux_first_order =
            (dg.numerical_flux_first_order, (dg.numerical_flux_first_order,))
    else
        numerical_flux_first_order = dg.numerical_flux_first_order
    end
    slow_dg = remainder_DGModel(
        dg,
        (fast_dg,);
        numerical_flux_first_order = numerical_flux_first_order,
    )

    slow_solver = ode_solver.slow_method(slow_dg, Q; dt = dt)
    fast_dt = dt / ode_solver.timestep_ratio
    fast_solver = ode_solver.fast_method(fast_dg, Q; dt = fast_dt)

    solver = MultirateRungeKutta((slow_solver, fast_solver), t0 = t0)

    return solver
end

"""
# Description
    solversetup(
        ode_solver::MultirateSolverType{HEVISplitting},
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

Creates an ODE solver for the partition slow-fast ODE
using a multirate method with HEVI time-integration.
The splitting of the fast (acoustic and gravity waves)
dynamics is performed by splitting the fast model
into horizontal and vertical directions. All horizontal
acoustic waves are treated explicitly, while the 1-D
vertical problem is treated implicitly. The HEVI-splitting
of the acoustic waves is handled using an IMEX additive
Runge-Kutta method in the fast (inner-most) solver.

Examples of similar multirate-HEVI approaches include
the following references.

### References
    @article{doms2011description,
        title={A Description of the Nonhydrostatic Regional COSMO model.
            Part I: Dynamics and Numerics},
        author={Doms, G{\"u}nther and Baldauf, M},
        journal={Deutscher Wetterdienst, Offenbach},
        year={2011}
    }

    @article{Tomita_2008,
        doi = {10.1137/070692273},
        year = {2008},
        month = {jan},
        publisher = {Society for Industrial and Applied Mathematics ({SIAM})},
        volume = {30},
        number = {6},
        pages = {2755--2776},
        author = {Hirofumi Tomita and Koji Goto and Masaki Satoh},
        title = {A New Approach to Atmospheric General Circulation Model:
            Global Cloud Resolving Model {NICAM} and its Computational Performance},
        journal = {{SIAM} Journal on Scientific Computing}
    }

# Comments:
Currently, the only HEVI-type splitting ClimateMachine can currently
do only involves splitting the acoustic processes; it is not currently
possible to perform more fine-grained separation of tendencies
(for example, including vertical advection or diffusion in the 1-D implicit problem)
"""
function solversetup(
    ode_solver::MultirateSolverType{HEVISplitting},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    # Extract fast model and define a DG model
    # for the fast processes
    fast_model = ode_solver.fast_model(dg.balance_law)

    # Full DG model for the acoustic waves in all directions
    acoustic_dg_full = DGModel(
        fast_model,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = EveryDirection(),
    )

    # DG model for the vertical acoustic waves only
    acoustic_dg_vert = DGModel(
        fast_model,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = VerticalDirection(),
    )

    # Compute fast time-step size from target ratio
    fast_dt = dt / ode_solver.timestep_ratio

    # Fast solver for the acoustic/gravity waves using
    # a HEVI-type splitting and a 1-D IMEX method
    fast_solver = ode_solver.fast_method(
        acoustic_dg_full,
        acoustic_dg_vert,
        LinearBackwardEulerSolver(
            ode_solver.implicit_solver();
            isadjustable = ode_solver.implicit_solver_adjustable,
        ),
        Q;
        dt = fast_dt,
        t0 = t0,
        # NOTE: This needs to be `false` since the ARK method will
        # evaluate the explicit part using the RemainderModel
        # (Difference between acoustic_dg_full and acoustic_dg_vert)
        split_explicit_implicit = false,
        # NOTE: Do we want to support more variants?
        variant = LowStorageVariant(),
    )

    # Finally, we subtract away the
    # fast processes and define a DG model for the
    # slower processes (advection and diffusion)
    if ode_solver.discrete_splitting
        numerical_flux_first_order =
            (dg.numerical_flux_first_order, (dg.numerical_flux_first_order,))
    else
        numerical_flux_first_order = dg.numerical_flux_first_order
    end
    slow_dg = remainder_DGModel(
        dg,
        (acoustic_dg_full,);
        numerical_flux_first_order = numerical_flux_first_order,
    )

    slow_solver = ode_solver.slow_method(slow_dg, Q; dt = dt, t0 = t0)

    solver = MultirateRungeKutta((slow_solver, fast_solver), t0 = t0)

    return solver
end
