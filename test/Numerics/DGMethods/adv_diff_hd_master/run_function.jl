struct ConstantHyperDiffusion{dim, dir, FT} <: HyperDiffusionProblem
    D::FT
    l::FT
    m::FT
end

function run(
    mpicomm,
    ArrayType,
    dim,
    topl,
    grid,
    model,
    N,
    FT,
    direction,
    τ,
    l,
    m,
)

    dx = min_node_distance(grid, HorizontalDirection())
    dz = min_node_distance(grid, VerticalDirection())
    @info "Δ(horz) Δ(vert)" (dx, dz)

    D = (dx/2)^4/2/τ

    dg = DGModel(
            model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction(),
        )

    # init state array (as a MPIStateArray) at t0, using init_state_prognostic! of the DGModel
    t0 = FT(0)
    Q0 = init_ode_state(dg, t0) 

    # init state array tendency (as a MPIStateArray) at t0
    ∂Q∂t_DG = similar(Q0)
    
    # update ∂Q∂t_DG at t0 using the DGModel
    dg(∂Q∂t_DG, Q0, nothing, 0)

    ∂Q∂t_anal = compute_analytical(::SphericalHarm, dg, Q0, D ) 

    # time integration
    # timestepper for 1 step
    Q_DGlsrk = Q0
    
    dt = dx^4 / 25 / sum(D)
    @info dt
    timeend = dt * 5

    Q_anal = init_ode_state(dg, timeend)

    lsrk = LSRK54CarpenterKennedy(dg, Q_DGlsrk; dt = dt, t0 = 0)
    solve!(Q_DGlsrk, lsrk; timeend = timeend)
    @info "Q(t) ANA vs DG: " norm(Q_anal.ρ .- Q_DGlsrk.ρ)/norm(Q_anal.ρ) 

    @show "rhs ANA vs DG" norm(∂Q∂t_anal .- ∂Q∂t_DG) / norm(∂Q∂t_anal)

    # do_output(mpicomm, "output", dg, Q0, rhs_anal, model)
    # do_output(mpicomm, "output", dg, Q_DGlsrk, Q_anal, model)

    rel_error_Q = norm(Q_anal.ρ .- Q_DGlsrk.ρ)/norm(Q_anal.ρ)
    rel_error_rhs = norm(rhs_anal .- rhs_DGsource) / norm(rhs_anal)
    
    return dg, Q_DGlsrk, Q_anal, rel_error_Q, rel_error_rhs
    # return norm(Q_anal-Q_DGlsrk)/norm(Q_anal)
end