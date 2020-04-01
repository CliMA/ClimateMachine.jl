using Logging, Printf

"""
    advective_courant(::HBModel)

calculates the CFL condition due to advection

"""
@inline function advective_courant(
    m::HBModel,
    Q::Vars,
    A::Vars,
    D::Vars,
    Δx,
    Δt,
    t,
    direction = VerticalDirection(),
)
    if direction isa VerticalDirection
        ū = norm(A.w)
    elseif direction isa HorizontalDirection
        ū = norm(Q.u)
    else
        v = @SVector [Q.u[1], Q.u[2], A.w]
        ū = norm(v)
    end

    return Δt * ū / Δx
end

"""
    nondiffusive_courant(::HBModel)

calculates the CFL condition due to gravity waves

"""
@inline function nondiffusive_courant(
    m::HBModel,
    Q::Vars,
    A::Vars,
    D::Vars,
    Δx,
    Δt,
    t,
    direction = HorizontalDirection(),
)
    return Δt * m.cʰ / Δx
end
"""
    viscous_courant(::HBModel)

calculates the CFL condition due to viscosity

"""
@inline function viscous_courant(
    m::HBModel,
    Q::Vars,
    A::Vars,
    D::Vars,
    Δx,
    Δt,
    t,
    direction = VerticalDirection(),
)
    ν̄ = norm_viscosity(m, direction)

    return Δt * ν̄ / Δx^2
end

@inline norm_viscosity(m::HBModel, ::VerticalDirection) = m.νᶻ
@inline norm_viscosity(m::HBModel, ::HorizontalDirection) = sqrt(2) * m.νʰ
@inline norm_viscosity(m::HBModel, ::EveryDirection) = sqrt(2 * m.νʰ^2 + m.νᶻ^2)


"""
    diffusive_courant(::HBModel)

calculates the CFL condition due to temperature diffusivity
factor of 1000 is for convective adjustment

"""
@inline function diffusive_courant(
    m::HBModel,
    Q::Vars,
    A::Vars,
    D::Vars,
    Δx,
    Δt,
    t,
    direction = VerticalDirection(),
)
    κ̄ = norm_diffusivity(m, direction)

    return Δt * κ̄ / Δx^2
end

@inline norm_diffusivity(m::HBModel, ::VerticalDirection) = 1000 * m.κᶻ
@inline norm_diffusivity(m::HBModel, ::HorizontalDirection) = sqrt(2) * m.κʰ
@inline norm_diffusivity(m::HBModel, ::EveryDirection) =
    sqrt(2 * m.κʰ^2 + (1000 * m.κᶻ)^2)

"""
    calculate_dt(dg, model::HBModel, Q, Courant_number, direction::EveryDirection, t)

calculates the time step based on grid spacing and model parameters
takes minimum of advective, gravity wave, diffusive, and viscous CFL

"""
@inline function calculate_dt(
    dg,
    model::HBModel,
    Q,
    Courant_number,
    t,
    ::EveryDirection,
)
    Δt = one(eltype(Q))

    CFL_advective =
        courant(advective_courant, dg, model, Q, Δt, t, VerticalDirection())
    CFL_gravity = courant(
        nondiffusive_courant,
        dg,
        model,
        Q,
        Δt,
        t,
        HorizontalDirection(),
    )
    CFL_viscous =
        courant(viscous_courant, dg, model, Q, Δt, t, VerticalDirection())
    CFL_diffusive =
        courant(diffusive_courant, dg, model, Q, Δt, t, VerticalDirection())

    CFLs = [CFL_advective, CFL_gravity, CFL_viscous, CFL_diffusive]
    dts = [Courant_number / CFL for CFL in CFLs]
    dt = min(dts...)

    @info @sprintf(
        """Calculating timestep
         Advective Constraint    = %.1f seconds
         Nondiffusive Constraint = %.1f seconds
         Viscous Constraint      = %.1f seconds
         Diffusive Constrait     = %.1f seconds
         Timestep                = %.1f seconds""",
        dts...,
        dt
    )

    return dt
end


"""
    calculate_dt(dg, bl::LinearHBModel, Q, Courant_number,
                 direction::EveryDirection)

calculates the time step based on grid spacing and model parameters
takes minimum of gravity wave, diffusive, and viscous CFL

"""
@inline function calculate_dt(
    dg,
    model::LinearHBModel,
    Q,
    Courant_number,
    t,
    ::EveryDirection,
)
    Δt = one(eltype(Q))
    ocean = model.ocean

    CFL_advective =
        courant(advective_courant, dg, ocean, Q, Δt, t, VerticalDirection())
    CFL_gravity = courant(
        nondiffusive_courant,
        dg,
        ocean,
        Q,
        Δt,
        t,
        HorizontalDirection(),
    )
    CFL_viscous =
        courant(viscous_courant, dg, ocean, Q, Δt, t, HorizontalDirection())
    CFL_diffusive =
        courant(diffusive_courant, dg, ocean, Q, Δt, t, HorizontalDirection())

    CFLs = [CFL_advective, CFL_gravity, CFL_viscous, CFL_diffusive]
    dts = [Courant_number / CFL for CFL in CFLs]
    dt = min(dts...)

    @info @sprintf(
        """Calculating timestep
         Advective Constraint    = %.1f seconds
         Nondiffusive Constraint = %.1f seconds
         Viscous Constraint      = %.1f seconds
         Diffusive Constrait     = %.1f seconds
         Timestep                = %.1f seconds""",
        dts...,
        dt
    )


    return dt
end
