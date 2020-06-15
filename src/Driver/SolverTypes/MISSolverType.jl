
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
    \\dot{Q} = f_fast(Q, t) + f_slow(Q, t)
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
- `nsubsteps` (Int): Integer denoting the total number of times
    to substep the fast process.
    Default: `50`

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
    nsubsteps::Int

    function MISSolverType(;
        splitting_type = SlowFastSplitting(),
        fast_model = AtmosAcousticGravityLinearModel,
        mis_method = MIS2,
        fast_method = LSRK54CarpenterKennedy,
        nsubsteps = 50,
    )

        DS = typeof(splitting_type)

        return new{DS}(
            splitting_type,
            fast_model,
            mis_method,
            fast_method,
            nsubsteps,
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
    slow_model = RemainderModel(dg.balance_law, (fast_model,))
    slow_dg = DGModel(
        slow_model,
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        # Ensure diffusion direction is passed to the correct
        # DG model
        diffusion_direction = diffusion_direction,
    )

    solver = ode_solver.mis_method(
        slow_dg,
        fast_dg,
        ode_solver.fast_method,
        ode_solver.nsubsteps,
        Q;
        dt = dt,
        t0 = t0,
    )

    return solver
end
