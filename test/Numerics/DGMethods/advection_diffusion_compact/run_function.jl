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
    N,
    FT,
    direction,
    τ,
    l,
    m,
)

    dx = min_node_distance(grid, HorizontalDirection())
    dz = min_node_distance(grid, VerticalDirection())

    D = (dx/2)^4/2/τ

    model = HyperDiffusion{dim}(ConstantHyperDiffusion{dim, direction(), FT}(D, l, m))
    dg = DGModel(
            model,
            grid,
            CentralNumericalFluxFirstOrder(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
            direction = direction(),
            diffusion_direction = EveryDirection()
        )

    Q0 = init_ode_state(dg, FT(0))
    @info "Δ(horz) Δ(vert)" (dx, dz)

    ϵ = 1e-3

    rhs_DGsource = similar(Q0)
    dg(rhs_DGsource, Q0, nothing, 0)

    # analycal vs analycal
    # analytical solution for Y_{l,m}
    rhs_anal = .- dg.state_auxiliary.c * D .* Q0

    # timestepper for 1 step
    Q_DGlsrk = Q0
    dg1 = dg
    
    dt = dx^4 / 25 / sum(D)
    @info dt

    Q_anal = init_ode_state(dg1, dt)

    lsrk = LSRK54CarpenterKennedy(dg1, Q_DGlsrk; dt = dt, t0 = 0)
    solve!(Q_DGlsrk, lsrk; timeend = dt)
    @info "DG stepper vs rhs: " norm(Q_anal-Q_DGlsrk)/norm(Q_anal) 

    # ana ρ(t) + finite diff in time
    # rhs_FD = (init_ode_state(dg, FT(ϵ)) .- Q0) ./ ϵ

    # @show "ANA" norm(rhs_anal)
    # @show "FD" norm(rhs_FD)
    # @show "DG" norm(rhs_DGsource)
    # @show "ANA vs FD" norm(rhs_anal .- rhs_FD)/norm(rhs_anal)
    # @show "ANA vs DG" norm(rhs_anal .- rhs_DGsource) / norm(rhs_anal)
    # @show "FD vs DG" norm(rhs_FD .- rhs_DGsource) / norm(rhs_FD)


    # do_output(mpicomm, "output", dg, Q0, rhs_anal, model)
    # do_output(mpicomm, "output", dg, Q_DGlsrk, Q_anal, model)

    rel_error = norm(rhs_anal .- rhs_DGsource) / norm(rhs_anal)
    return dg, model, Q_DGlsrk, Q_anal, rel_error
    # return norm(Q_anal-Q_DGlsrk)/norm(Q_anal)
end