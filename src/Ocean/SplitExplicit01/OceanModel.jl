struct OceanModel{PS, P, T} <: AbstractOceanModel
    param_set::PS
    problem::P
    ρₒ::T
    cʰ::T
    cᶻ::T
    add_fast_substeps::T
    numImplSteps::T
    ivdc_dt::T
    αᵀ::T
    νʰ::T
    νᶻ::T
    κʰ::T
    κᶻ::T
    κᶜ::T
    fₒ::T
    β::T
    function OceanModel{FT}(
        param_set,
        problem;
        ρₒ = FT(1000),  # kg/m^3
        cʰ = FT(0),     # m/s
        cᶻ = FT(0),     # m/s
        add_fast_substeps = 0,
        numImplSteps = 0,
        ivdc_dt = FT(1),
        αᵀ = FT(2e-4),  # 1/K
        νʰ = FT(5e3),   # m^2/s
        νᶻ = FT(5e-3),  # m^2/s
        κʰ = FT(1e3),   # m^2/s
        κᶻ = FT(1e-4),  # vertical diffusivity, m^2/s
        κᶜ = FT(1e-4),  # convective adjustment vertical diffusivity, m^2/s
        fₒ = FT(1e-4),  # Hz
        β = FT(1e-11),  # Hz/m
    ) where {FT <: AbstractFloat}
        return new{typeof(param_set), typeof(problem), FT}(
            param_set,
            problem,
            ρₒ,
            cʰ,
            cᶻ,
            add_fast_substeps,
            numImplSteps,
            ivdc_dt,
            αᵀ,
            νʰ,
            νᶻ,
            κʰ,
            κᶻ,
            κᶜ,
            fₒ,
            β,
        )
    end
end

function calculate_dt(grid, ::OceanModel, _...)
    #=
      minΔx = min_node_distance(grid, HorizontalDirection())
      minΔz = min_node_distance(grid, VerticalDirection())

      CFL_gravity = minΔx / model.cʰ
      CFL_diffusive = minΔz^2 / (1000 * model.κᶻ)
      CFL_viscous = minΔz^2 / model.νᶻ

      dt = 1 // 2 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])
    =#
    # FT = eltype(grid)
    # dt = FT(1)

    return nothing
end

"""
    OceanDGModel()

helper function to add required filtering
not used in the Driver+Config setup
"""
function OceanDGModel(
    bl::OceanModel,
    grid,
    numfluxnondiff,
    numfluxdiff,
    gradnumflux;
    kwargs...,
)
    vert_filter = CutoffFilter(grid, polynomialorder(grid) - 1)
    exp_filter = ExponentialFilter(grid, 1, 8)

    flowintegral_dg = DGModel(
        FlowIntegralModel(bl),
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
    )

    tendency_dg = DGModel(
        TendencyIntegralModel(bl),
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
    )

    conti3d_dg = DGModel(
        Continuity3dModel(bl),
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux,
    )
    FT = eltype(grid)
    conti3d_Q = init_ode_state(conti3d_dg, FT(0); init_on_cpu = true)

    ivdc_dg = DGModel(
        IVDCModel(bl),
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux;
        direction = VerticalDirection(),
    )
    ivdc_Q = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true) # Not sure this is needed since we set values later,
    # but we'll do it just in case!

    ivdc_RHS = init_ode_state(ivdc_dg, FT(0); init_on_cpu = true) # Not sure this is needed since we set values later,
    # but we'll do it just in case!

    ivdc_bgm_solver = BatchedGeneralizedMinimalResidual(
        ivdc_dg,
        ivdc_Q;
        max_subspace_size = 10,
    )

    modeldata = (
        dg_2D = kwargs[1][1],
        Q_2D = kwargs[1][2],
        vert_filter = vert_filter,
        exp_filter = exp_filter,
        flowintegral_dg = flowintegral_dg,
        tendency_dg = tendency_dg,
        conti3d_dg = conti3d_dg,
        conti3d_Q = conti3d_Q,
        ivdc_dg = ivdc_dg,
        ivdc_Q = ivdc_Q,
        ivdc_RHS = ivdc_RHS,
        ivdc_bgm_solver = ivdc_bgm_solver,
    )

    return DGModel(
        bl,
        grid,
        numfluxnondiff,
        numfluxdiff,
        gradnumflux;
        kwargs...,
        modeldata = modeldata,
    )
end

function vars_state(m::OceanModel, ::Prognostic, T)
    @vars begin
        u::SVector{2, T}
        η::T
        θ::T
    end
end

function init_state_prognostic!(m::OceanModel, Q::Vars, A::Vars, localgeo, t)
    return ocean_init_state!(m, m.problem, Q, A, localgeo, t)
end

function vars_state(m::OceanModel, ::Auxiliary, T)
    @vars begin
        w::T
        pkin::T         # kinematic pressure: ∫(-g αᵀ θ)
        wz0::T          # w at z=0
        u_d::SVector{2, T}  # velocity deviation from vertical mean
        ΔGu::SVector{2, T}
        y::T     # y-coordinate of the box
    end
end

function init_state_auxiliary!(
    m::OceanModel,
    state_aux::MPIStateArray,
    grid,
    direction,
)
    init_state_auxiliary!(
        m,
        (m, A, tmp, geom) -> ocean_init_aux!(m, m.problem, A, geom),
        state_aux,
        grid,
        direction,
    )
end

function vars_state(m::OceanModel, ::Gradient, T)
    @vars begin
        u::SVector{2, T}
        ud::SVector{2, T}
        θ::T
    end
end

@inline function compute_gradient_argument!(
    m::OceanModel,
    G::Vars,
    Q::Vars,
    A,
    t,
)
    G.u = Q.u
    G.ud = A.u_d
    G.θ = Q.θ

    return nothing
end

function vars_state(m::OceanModel, ::GradientFlux, T)
    @vars begin
        ν∇u::SMatrix{3, 2, T, 6}
        κ∇θ::SVector{3, T}
    end
end

@inline function compute_gradient_flux!(
    m::OceanModel,
    D::Vars,
    G::Grad,
    Q::Vars,
    A::Vars,
    t,
)
    ν = viscosity_tensor(m)
    #   D.ν∇u = ν * G.u
    D.ν∇u = @SMatrix [
        m.νʰ * G.ud[1, 1] m.νʰ * G.ud[1, 2]
        m.νʰ * G.ud[2, 1] m.νʰ * G.ud[2, 2]
        m.νᶻ * G.u[3, 1] m.νᶻ * G.u[3, 2]
    ]

    κ = diffusivity_tensor(m, G.θ[3])
    D.κ∇θ = κ * G.θ

    return nothing
end

@inline viscosity_tensor(m::OceanModel) = Diagonal(@SVector [m.νʰ, m.νʰ, m.νᶻ])

@inline function diffusivity_tensor(m::OceanModel, ∂θ∂z)

    if m.numImplSteps > 0
        κ = (@SVector [m.κʰ, m.κʰ, m.κᶻ * 0.5])
    else
        ∂θ∂z < 0 ? κ = (@SVector [m.κʰ, m.κʰ, m.κᶜ]) : κ = (@SVector [
            m.κʰ,
            m.κʰ,
            m.κᶻ,
        ])
    end

    return Diagonal(κ)
end

"""
    vars_integral(::OceanModel)

location to store integrands for bottom up integrals
∇hu = the horizontal divegence of u, e.g. dw/dz
"""
function vars_state(m::OceanModel, ::UpwardIntegrals, T)
    @vars begin
        ∇hu::T
        buoy::T
        #       ∫u::SVector{2, T}
    end
end

"""
    integral_load_auxiliary_state!(::OceanModel)

copy w to var_integral
this computation is done pointwise at each nodal point

arguments:
m -> model in this case OceanModel
I -> array of integrand variables
Q -> array of state variables
A -> array of aux variables
"""
@inline function integral_load_auxiliary_state!(
    m::OceanModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.∇hu = A.w # borrow the w value from A...
    I.buoy = grav(m.param_set) * m.αᵀ * Q.θ # buoyancy to integrate vertically from top (=reverse)
    #   I.∫u = Q.u

    return nothing
end

"""
    integral_set_auxiliary_state!(::OceanModel)

copy integral results back out to aux
this computation is done pointwise at each nodal point

arguments:
m -> model in this case OceanModel
A -> array of aux variables
I -> array of integrand variables
"""
@inline function integral_set_auxiliary_state!(m::OceanModel, A::Vars, I::Vars)
    A.w = I.∇hu
    A.pkin = -I.buoy
    #   A.∫u = I.∫u

    return nothing
end

"""
    vars_reverse_integral(::OceanModel)

location to store integrands for top down integrals
αᵀθ = density perturbation
"""
function vars_state(m::OceanModel, ::DownwardIntegrals, T)
    @vars begin
        buoy::T
    end
end

"""
    reverse_integral_load_auxiliary_state!(::OceanModel)

copy αᵀθ to var_reverse_integral
this computation is done pointwise at each nodal point

arguments:
m -> model in this case OceanModel
I -> array of integrand variables
A -> array of aux variables
"""
@inline function reverse_integral_load_auxiliary_state!(
    m::OceanModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.buoy = A.pkin

    return nothing
end

"""
    reverse_integral_set_auxiliary_state!(::OceanModel)

copy reverse integral results back out to aux
this computation is done pointwise at each nodal point

arguments:
m -> model in this case OceanModel
A -> array of aux variables
I -> array of integrand variables
"""
@inline function reverse_integral_set_auxiliary_state!(
    m::OceanModel,
    A::Vars,
    I::Vars,
)
    A.pkin = I.buoy

    return nothing
end

@inline function flux_first_order!(
    m::OceanModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
    direction,
)
    @inbounds begin
        u = Q.u # Horizontal components of velocity
        θ = Q.θ
        w = A.w   # vertical velocity
        pkin = A.pkin
        v = @SVector [u[1], u[2], w]
        Iʰ = @SMatrix [
            1 -0
            -0 1
            -0 -0
        ]

        # ∇h • (g η)
        #- jmc: put back this term to check
        #       η = Q.η
        #       F.u += grav(m.param_set) * η * Iʰ

        # ∇ • (u θ)
        F.θ += v * θ

        # ∇h • pkin
        F.u += pkin * Iʰ

        # ∇h • (v ⊗ u)
        # F.u += v * u'
    end

    return nothing
end

@inline function flux_second_order!(
    m::OceanModel,
    F::Grad,
    Q::Vars,
    D::Vars,
    HD::Vars,
    A::Vars,
    t::Real,
)
    # horizontal viscosity done in horizontal model
    #   F.u -= @SVector([0, 0, 1]) * D.ν∇u[3, :]'
    #- jmc: put back this term to check
    F.u -= D.ν∇u

    F.θ -= D.κ∇θ

    return nothing
end

@inline function source!(
    m::OceanModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t::Real,
    direction,
)
    @inbounds begin
        u = Q.u    # Horizontal components of velocity
        ud = A.u_d # Horizontal velocity deviation from vertical mean

        # f × u
        f = coriolis_force(m, A.y)
        # S.u -= @SVector [-f * u[2], f * u[1]]
        S.u -= @SVector [-f * ud[2], f * ud[1]]

        #- borotropic tendency adjustment
        S.u += A.ΔGu

        # switch this to S.η if you comment out the fast mode in MultistateMultirateRungeKutta
        S.η += A.wz0
    end

    return nothing
end

@inline coriolis_force(m::OceanModel, y) = m.fₒ + m.β * y

function update_auxiliary_state!(
    dg::DGModel,
    m::OceanModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    FT = eltype(Q)
    A = dg.state_auxiliary
    MD = dg.modeldata

    # `update_auxiliary_state!` gets called twice, once for the real elements
    # and once for the ghost elements. Only apply the filters to the real elems.
    if elems == dg.grid.topology.realelems
        # required to ensure that after integration velocity field is divergence free
        vert_filter = MD.vert_filter
        # Q[1] = u[1] = u, Q[2] = u[2] = v
        apply!(Q, (1, 2), dg.grid, vert_filter, direction = VerticalDirection())

        exp_filter = MD.exp_filter
        # Q[4] = θ
        apply!(Q, (4,), dg.grid, exp_filter, direction = VerticalDirection())
    end

    #----------
    # Compute Divergence of Horizontal Flow field using "conti3d_dg" DGmodel

    conti3d_dg = dg.modeldata.conti3d_dg
    # ct3d_Q = dg.modeldata.conti3d_Q
    # ct3d_dQ = similar(ct3d_Q)
    # fill!(ct3d_dQ, 0)
    #- Instead, use directly conti3d_Q to store dQ (since we will not update the state)
    ct3d_dQ = dg.modeldata.conti3d_Q

    ct3d_bl = conti3d_dg.balance_law

    # call "conti3d_dg" DGmodel
    # note: with "increment = false", just return tendency (no state update)
    p = nothing
    conti3d_dg(ct3d_dQ, Q, p, t; increment = false)

    # Copy from ct3d_dQ.θ which is realy ∇h•u into A.w (which will be integrated)
    function f!(::OceanModel, dQ, A, t)
        @inbounds begin
            A.w = dQ.θ
        end
    end
    update_auxiliary_state!(f!, dg, m, ct3d_dQ, t, elems)
    #----------

    Nq, Nqk, _, _, nelemv, nelemh, nrealelemh, _ = basic_grid_info(dg)

    # compute integrals for w and pkin
    indefinite_stack_integral!(dg, m, Q, A, t, elems) # bottom -> top
    reverse_indefinite_stack_integral!(dg, m, Q, A, t, elems) # top -> bottom

    # copy down wz0
    # We are unable to use vars (ie A.w) for this because this operation will
    # return a SubArray, and adapt (used for broadcasting along reshaped arrays)
    # has a limited recursion depth for the types allowed.
    nb_aux_m = number_states(m, Auxiliary())
    data_m = reshape(A.data, Nq^2, Nqk, nb_aux_m, nelemv, nelemh)

    # project w(z=0) down the stack
    index_w = varsindex(vars_state(m, Auxiliary(), FT), :w)
    index_wz0 = varsindex(vars_state(m, Auxiliary(), FT), :wz0)
    flat_wz0 = @view data_m[:, end:end, index_w, end:end, 1:nrealelemh]
    boxy_wz0 = @view data_m[:, :, index_wz0, :, 1:nrealelemh]
    boxy_wz0 .= flat_wz0

    # Compute Horizontal Flow deviation from vertical mean

    flowintegral_dg = dg.modeldata.flowintegral_dg
    flowint = flowintegral_dg.balance_law
    update_auxiliary_state!(flowintegral_dg, flowint, Q, 0, elems)

    ## get top value (=integral over full depth)
    nb_aux_flw = number_states(flowint, Auxiliary())
    data_flw = reshape(
        flowintegral_dg.state_auxiliary.data,
        Nq^2,
        Nqk,
        nb_aux_flw,
        nelemv,
        nelemh,
    )
    index_∫u = varsindex(vars_state(flowint, Auxiliary(), FT), :∫u)

    flat_∫u = @view data_flw[:, end:end, index_∫u, end:end, 1:nrealelemh]

    ## make a copy of horizontal velocity
    A.u_d .= Q.u

    ## and remove vertical mean velocity
    index_ud = varsindex(vars_state(m, Auxiliary(), FT), :u_d)
    boxy_ud = @view data_m[:, :, index_ud, :, 1:nrealelemh]
    boxy_ud .-= flat_∫u / m.problem.H

    return true
end

@inline wavespeed(m::OceanModel, n⁻, _...) =
    abs(SVector(m.cʰ, m.cʰ, m.cᶻ)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
#@inline function update_penalty!(
#   ::Union{RusanovNumericalFlux, CentralNumericalFluxFirstOrder},
function update_penalty!(
    ::RusanovNumericalFlux,
    ::OceanModel,
    n⁻,
    λ,
    ΔQ::Vars,
    Q⁻,
    A⁻,
    Q⁺,
    A⁺,
    t,
)
    ΔQ.η = -0

    return nothing
end

@inline function boundary_state!(nf, ocean::OceanModel, args...)
    boundary_conditions = ocean.problem.boundary_conditions
    return ocean_boundary_state!(nf, boundary_conditions, ocean, args...)
end
