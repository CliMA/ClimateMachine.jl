
export ThreeDimIMEXSolverType

struct ThreeDimIMEXSolverType <: AbstractSolverType
    # The implicit model
    implicit_model::Type
    # Function for the IMEX method
    solver_method::Function
    # Storage type for the ARK scheme
    solver_storage_variant
    # Split tendency or not
    split_explicit_implicit::Bool
    # Whether to use a PDE level or discrete splitting
    discrete_splitting::Bool
    max_iter
    rtol

    function ThreeDimIMEXSolverType(;
        implicit_model = AtmosAcousticGravityLinearModel,
        solver_method = ARK2GiraldoKellyConstantinescu,
        solver_storage_variant = LowStorageVariant(),
        split_explicit_implicit = true,
        discrete_splitting = false,
        max_gmres_iter = 50,
        solver_rtol = 1e-10,
    )
        @assert discrete_splitting || split_explicit_implicit

        return new(
            implicit_model,
            solver_method,
            solver_storage_variant,
            split_explicit_implicit,
            discrete_splitting,
            max_gmres_iter,
            solver_rtol,
        )
    end
end

function getdtmodel(ode_solver::ThreeDimIMEXSolverType, bl)
    return ode_solver.implicit_model(bl)
end

function solversetup(
    ode_solver::ThreeDimIMEXSolverType,
    dg,
    Q,
    dt,
    t0,
    diffusion_direction,
)

    adg = DGModel(
        ode_solver.implicit_model(dg.balance_law),
        dg.grid,
        dg.numerical_flux_first_order,
        dg.numerical_flux_second_order,
        dg.numerical_flux_gradient,
        state_auxiliary = dg.state_auxiliary,
        state_gradient_flux = dg.state_gradient_flux,
        states_higher_order = dg.states_higher_order,
        direction = EveryDirection(),
    )

    implicit_solver = GeneralizedMinimalResidual(Q; M = ode_solver.max_iter, rtol = ode_solver.rtol)

    if ode_solver.split_explicit_implicit
        if ode_solver.discrete_splitting
            numerical_flux_first_order = (
                dg.numerical_flux_first_order,
                (dg.numerical_flux_first_order,),
            )
        else
            numerical_flux_first_order = dg.numerical_flux_first_order
        end
        rem_dg = remainder_DGModel(
            dg,
            (adg,);
            numerical_flux_first_order = numerical_flux_first_order,
        )
        solver = ode_solver.solver_method(
            rem_dg,
            adg,
            LinearBackwardEulerSolver(
                implicit_solver;
                isadjustable = true,
            ),
            Q;
            split_explicit_implicit = true,
            dt = dt,
            t0 = t0,
        )
    else
        solver = ode_solver.solver_method(
            dg,
            adg,
            LinearBackwardEulerSolver(
                implicit_solver;
                isadjustable = true,
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
    end

    return solver
end
