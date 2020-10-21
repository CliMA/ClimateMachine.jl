
export MISSolverType

"""
# Description
    MISSolverType(;
        splitting_type = SlowFastSplitting(),
        fast_model = AtmosAcousticGravityLinearModel,
        mis_method = MIS2,
        fast_method = LSRK54CarpenterKennedy,
        nsubsteps = 50,
    )

This solver type constructs an ODE solver using a generalization
of the split-explicit Runge-Kutta method. Known as the Multirate
Infinitesimal Step (MIS) method, this solver solves ODEs with
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
- `mis_method` (Function): Function defining the particular MIS
    method to be used.
    Default: `MIS2`
- `fast_method` (Function): Function defining the fast solver.
    Default: `LSRK54CarpenterKennedy`
- `nsubsteps` (Tuple): Tuple denoting the total number of times
    to substep the fast process.
    Default: `(50,)`
- `discrete_splitting` (Boolean): Boolean denoting whether a PDE level or
    discretized level splitting should be used. If `true` then the PDE is
    discretized in such a way that `f_fast + f_slow` is equivalent to
    discretizing the original PDE directly.
    Default: `false`
- `hivi_splitting` (Boolean): Boolean denoting whether the fast processes need to
    be split into their horizontal and vertical components. For instance this is
    necessary, when a MIS method is chosen as the fast solver.
    Default: `false`
- `numerical_flux_first_order_fast` (NumericalFluxFirstOrder): The type of
    numerical flux of first order to be used for the fast processes.
- `numerical_flux_first_order_slow` (NumericalFluxFirstOrder): The type of
    numerical flux of first order to be used for the slow processes.
    Default: `RusanovNumericalFlux()`

### References
    @article{KnothWensch2014,
        title={Generalized split-explicit Runge--Kutta methods for
            the compressible Euler equations},
        author={Knoth, Oswald and Wensch, Joerg},
        journal={Monthly Weather Review},
        volume={142},
        number={5},
        pages={2067--2081},
        year={2014}
    }
"""
struct MISSolverType{DS} <: AbstractSolverType
    # The type of discrete splitting to apply to the right-hand side
    splitting_type::DS
    # The model describing fast dynamics
    fast_model::Type
    # Main MIS function
    mis_method::Function
    # Fast RK solver
    fast_method::Function
    # Substepping parameter for the fast processes
    nsubsteps::Tuple
    # Whether to use a PDE level or discrete splitting
    discrete_splitting::Bool
    # Whether to split the fast processes into horizontal and vertical components
    hivi_splitting::Bool
    # Numerical flux of first order for the fast processes
    numerical_flux_first_order_fast::NumericalFluxFirstOrder
    # Numerical flux of first order for the slow processes
    numerical_flux_first_order_slow::NumericalFluxFirstOrder

    function MISSolverType(;
        splitting_type = SlowFastSplitting(),
        fast_model = AtmosAcousticGravityLinearModel,
        mis_method = MIS2,
        fast_method = LSRK54CarpenterKennedy,
        nsubsteps = (50,),
        discrete_splitting = false,
        hivi_splitting = false,
        numerical_flux_first_order_fast = RusanovNumericalFlux(),
        numerical_flux_first_order_slow = RusanovNumericalFlux(),
    )

        DS = typeof(splitting_type)

        return new{DS}(
            splitting_type,
            fast_model,
            mis_method,
            fast_method,
            nsubsteps,
            discrete_splitting,
            hivi_splitting,
            numerical_flux_first_order_fast,
            numerical_flux_first_order_slow,
        )
    end
end

"""
    getdtmodel(ode_solver::MISSolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
function getdtmodel(ode_solver::MISSolverType, bl)
    # Most restrictive dynamics are part of the fast model
    return ode_solver.fast_model(bl)
end

"""
# Description
    function solversetup(
        ode_solver::MISSolverType{SlowFastSplitting},
        dg,
        Q,
        dt,
        t0,
        diffusion_direction,
    )

Creates an ODE solver for the partition slow-fast ODE
using an MIS method with explicit time-integration.
The splitting of the fast (acoustic and gravity waves)
dynamics is done in _all_ spatial directions.
"""
function solversetup(
    ode_solver::MISSolverType{SlowFastSplitting},
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
    fast_method = ode_solver.fast_method
    fast_dg = DGModel(
        fast_model,
        dg.grid,
        ode_solver.numerical_flux_first_order_fast,
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
            (ode_solver.numerical_flux_first_order_slow, (ode_solver.numerical_flux_first_order_slow,))
    else
        numerical_flux_first_order = ode_solver.numerical_flux_first_order_slow
    end
    slow_dg = remainder_DGModel(
        dg,
        (fast_dg,);
        numerical_flux_first_order = numerical_flux_first_order,
    )

    if ode_solver.hivi_splitting
        fast_dg_h = DGModel(
            fast_model,
            dg.grid,
            ode_solver.numerical_flux_first_order_fast,
            dg.numerical_flux_second_order,
            dg.numerical_flux_gradient,
            state_auxiliary = dg.state_auxiliary,
            state_gradient_flux = dg.state_gradient_flux,
            states_higher_order = dg.states_higher_order,
            direction = HorizontalDirection(),
        )
        fast_dg_v = DGModel(
            fast_model,
            dg.grid,
            ode_solver.numerical_flux_first_order_fast,
            dg.numerical_flux_second_order,
            dg.numerical_flux_gradient,
            state_auxiliary = dg.state_auxiliary,
            state_gradient_flux = dg.state_gradient_flux,
            states_higher_order = dg.states_higher_order,
            direction = VerticalDirection(),
        )
        fast_dg = (fast_dg_h, fast_dg_v)
        if length(ode_solver.nsubsteps) == 1
            nsubsteps = getnsubsteps(
                ode_solver.mis_method,
                ode_solver.nsubsteps[1],
                real(eltype(Q)),
            )
            fast_method = (dg, Q) -> ode_solver.fast_method(
                    dg,
                    Q,
                    dt / ode_solver.nsubsteps[1],
                    nsubsteps,
            )
        elseif length(ode_solver.nsubsteps) == 2
            fast_method = (dg, Q) -> ode_solver.fast_method(
                dg,
                Q,
                ode_solver.nsubsteps[2]
            )
        end
    end

    solver = ode_solver.mis_method(
        slow_dg,
        fast_dg,
        fast_method,
        ode_solver.nsubsteps[1],
        Q;
        dt = dt,
        t0 = t0,
    )

    return solver
end
