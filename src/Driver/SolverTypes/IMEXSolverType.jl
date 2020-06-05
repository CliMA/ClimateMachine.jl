
export IMEXSolverType

"""
# Description
    IMEXSolverType(;
        splitting_type = HEVISplitting(),
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        implicit_solver_adjustable = false,
        solver_method = ARK2GiraldoKellyConstantinescu,
        solver_storage_variant = LowStorageVariant(),
    )

This solver type constructs a solver for ODEs with the
additively-partitioned form:

```math
  \\dot{Q} = [l(Q, t)] + [f(Q, t) - l(Q, t)]
```

where `Q` is the state, `f` is the full tendency and `l` is the chosen
implicit operator.

# Arguments
- `splitting_type` (DiscreteSplittingType): The type of discrete
    splitting to apply to the right-hand side.
    Default: `HEVISplitting()`
- `implicit_model` (Type): The model describing dynamics to be
    treated implicitly.
    Default: `AtmosAcousticGravityLinearModel`
- `implicit_solver` (Type): A solver for inverting the
    implicit system of equations.
    Default: `ManyColumnLU`
- `implicit_solver_adjustable` (Bool): A flag identifying whether
    or not the `implicit_solver` can be updated as the time-step
    size changes.
    Default: `false`
- `solver_method` (Function): Function defining the particular additive
    Runge-Kutta method to be used for the IMEX method.
    Default: `ARK2GiraldoKellyConstantinescu`
- `solver_storage_variant` (Type): Storage type for the additive
    Runge-Kutta method.
    Default: `LowStorageVariant()`

### References
    @article{giraldo2013implicit,
      title={Implicit-explicit formulations of a three-dimensional
             nonhydrostatic unified model of the atmosphere ({NUMA})},
      author={Giraldo, Francis X and Kelly, James F and Constantinescu, Emil M},
      journal={SIAM Journal on Scientific Computing},
      volume={35},
      number={5},
      pages={B1162--B1194},
      year={2013},
      publisher={SIAM}
    }
"""
struct IMEXSolverType{DS, ST} <: AbstractSolverType
    # The type of discrete splitting to apply to the right-hand side
    splitting_type::DS
    # The implicit model
    implicit_model::Type
    # Choice of implicit solver
    implicit_solver::Type
    # Can the implicit solver be updated with changing dt?
    implicit_solver_adjustable::Bool
    # Function for the IMEX method
    solver_method::Function
    # Storage type for the ARK scheme
    solver_storage_variant::ST

    function IMEXSolverType(;
        splitting_type = HEVISplitting(),
        implicit_model = AtmosAcousticGravityLinearModel,
        implicit_solver = ManyColumnLU,
        implicit_solver_adjustable = false,
        solver_method = ARK2GiraldoKellyConstantinescu,
        solver_storage_variant = LowStorageVariant(),
    )

        DS = typeof(splitting_type)
        ST = typeof(solver_storage_variant)

        return new{DS, ST}(
            splitting_type,
            implicit_model,
            implicit_solver,
            implicit_solver_adjustable,
            solver_method,
            solver_storage_variant,
        )
    end
end

"""
    getdtmodel(ode_solver::IMEXSolverType, bl)

A function which returns a model representing the dynamics
with the most restrictive time-stepping requirements.
"""
function getdtmodel(ode_solver::IMEXSolverType, bl)
    # Most restrictive dynamics are treated implicitly
    return ode_solver.implicit_model(bl)
end

"""
# Description
    solversetup(
        ode_solver::IMEXSolverType{HEVISplitting},
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
    ode_solver::IMEXSolverType{HEVISplitting},
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    # All we need to do is create a DGModel for the
    # vertical acoustic waves (determined from the `implicit_model`)
    vdg = DGModel(
        ode_solver.implicit_model(dg.balance_law),
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = VerticalDirection(),
    )

    solver = ode_solver.solver_method(
        dg,
        vdg,
        LinearBackwardEulerSolver(
            ode_solver.implicit_solver();
            isadjustable = ode_solver.implicit_solver_adjustable,
        ),
        Q;
        dt = dt,
        t0 = t0,
        # NOTE: This needs to be `false` since the ARK method will
        # evaluate the explicit part using the RemainderModel
        # (Difference between full DG model (dg) and the
        # DG model associated with the 1-D implicit problem (vdg))
        split_explicit_implicit = false,
        variant = ode_solver.solver_storage_variant,
    )

    return solver
end
