
export MISSolverType

"""
    MISSolverType(;
        splitting_type = SlowFastSplitting(),
        linear_model = AtmosAcousticGravityLinearModel,
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
- `linear_model` (Type): The linear model describing fast dynamics.
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
        title={Generalized split-explicit Runge--Kutta methods for the compressible Euler equations},
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
    # Linear model describing fast dynamics
    linear_model::Type
    # Main MIS function
    mis_method::Function
    # Fast RK solver
    fast_method::Function
    # Substepping parameter for the fast processes
    nsubsteps::Int

    function MISSolverType(;
        splitting_type = SlowFastSplitting(),
        linear_model = AtmosAcousticGravityLinearModel,
        mis_method = MIS2,
        fast_method = LSRK54CarpenterKennedy,
        nsubsteps = 50,
    )

        DS = typeof(splitting_type)

        return new{DS}(
            splitting_type,
            linear_model,
            mis_method,
            fast_method,
            nsubsteps,
        )
    end
end

function solversetup(
    ode_solver::MISSolverType{DS},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
) where {DS <: SlowFastSplitting}

    # Extract linear model and define a DG model
    # for the fast processes (acoustic/gravity waves
    # in all spatial directions)
    linmodel = ode_solver.linear_model(dg.balance_law)
    fast_dg = DGModel(
        linmodel,
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
    slow_model = RemainderModel(dg.balance_law, (linmodel,))
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
