export ExplicitFSALRungeKutta
export ERK23BogackiShampine, ERK45DormandPrince

mutable struct ExplicitFSALRungeKutta{T, RT, AT, Nstages, Nstages_sq} <:
               AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "rhs function"
    rhs!
    "Storage for the RHS evaluations"
    Rstages::NTuple{Nstages, AT}
    "Storage for the stages"
    Qstages::NTuple{Nstages, AT}
    order::Int
    embedded_order::Int
    "RK coefficient vector A (rhs scaling)"
    RKA::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    "RK coefficient vector B (rhs add in scaling)"
    RKB::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK coefficient vector B for the embedded scheme"
    RKB_embedded::SArray{Tuple{Nstages}, RT, 1, Nstages}
    "RK coefficient vector C (time scaling)"
    RKC::SArray{Tuple{Nstages}, RT, 1, Nstages}

    function ExplicitFSALRungeKutta(
        rhs!,
        order,
        embedded_order,
        RKA,
        RKB,
        RKB_embedded,
        RKC,
        Q::AT;
        dt = 0,
        t0 = 0,
    ) where {AT <: AbstractArray}
        T = eltype(Q)
        RT = real(T)
        Nstages = length(RKB)

        Rstages = ntuple(i -> similar(Q), Nstages)
        Qstages = ntuple(i -> similar(Q), Nstages)

        new{T, RT, AT, Nstages, Nstages^2}(
            RT(dt),
            RT(t0),
            rhs!,
            Rstages,
            Qstages,
            order,
            embedded_order,
            RKA,
            RKB,
            RKB_embedded,
            RKC,
        )
    end
end

embedded_order(erk::ExplicitFSALRungeKutta) = erk.embedded_order
updatedt!(erk::ExplicitFSALRungeKutta, dt) = erk.dt = dt
updatetime!(erk::ExplicitFSALRungeKutta, time) = (erk.t = time)

function ExplicitFSALRungeKutta(
    spacedisc::AbstractSpaceMethod,
    RKA,
    RKB,
    RKC,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    rhs! =
        (x...; increment) ->
            SpaceMethods.odefun!(spacedisc, x..., increment = increment)
    ExplicitFSALRungeKutta(
        rhs!,
        RKA,
        RKB,
        RKB_embedded,
        RKC,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function dostep!(
    (Qnp1, Qn, error_estimate),
    erk::ExplicitFSALRungeKutta,
    p,
    time,
    dt;
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
)
    RKA, RKB, RKC = erk.RKA, erk.RKB, erk.RKC
    RKB_embedded = erk.RKB_embedded
    Nstages = length(RKB)
    rhs! = erk.rhs!
    Rstages, Qstages = erk.Rstages, erk.Qstages

    groupsize = 256

    for s in 1:Nstages
        event = Event(device(Qn))
        event = stage_update!(device(Qn), groupsize)(
            realview(Qn),
            realview.(Qstages),
            realview.(Rstages),
            RKA,
            dt,
            Val(s),
            slow_δ,
            slow_rv_dQ,
            ndrange = length(realview(Qn)),
            dependencies = (event,),
        )
        wait(device(Qn), event)
        rhs!(Rstages[s], Qstages[s], p, time + RKC[s] * dt, increment = false)
    end

    event = Event(device(Qn))
    event = solution_update!(device(Qn), groupsize)(
        realview(Qnp1),
        realview(Qn),
        realview(error_estimate),
        realview.(Qstages),
        realview.(Rstages),
        RKB,
        RKB_embedded,
        dt,
        Val(Nstages),
        slow_δ,
        slow_rv_dQ,
        slow_scaling,
        ndrange = length(realview(Qn)),
        dependencies = (event,),
    )
    wait(device(Qn), event)
end

dostep!(
    Q::AbstractArray,
    erk::ExplicitFSALRungeKutta,
    p,
    time,
    dt,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
) = dostep!(
    (Q, Q, nothing),
    erk,
    p,
    time,
    dt,
    slow_δ = slow_δ,
    slow_rv_dQ = slow_rv_dQ,
    slow_scaling = slow_scaling,
)

@kernel function stage_update!(
    Q,
    Qstages,
    Rstages,
    RKA,
    dt,
    ::Val{is},
    slow_δ,
    slow_dQ,
) where {is}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[is - 1][i] += slow_δ * slow_dQ[i]
        end
        Qstages_is_i = Q[i]
        @unroll for js in 1:(is - 1)
            Qstages_is_i += dt * RKA[is, js] * Rstages[js][i]
        end
        Qstages[is][i] = Qstages_is_i
    end
end

@kernel function solution_update!(
    Qnp1,
    Qn,
    error_estimate,
    Qstages,
    Rstages,
    RKB,
    RKB_embedded,
    dt,
    ::Val{Nstages},
    slow_δ,
    slow_dQ,
    slow_scaling,
) where {Nstages}
    i = @index(Global, Linear)
    @inbounds begin
        if slow_δ !== nothing
            Rstages[Nstages][i] += slow_δ * slow_dQ[i]
        end
        if slow_scaling !== nothing
            slow_dQ[i] *= slow_scaling
        end

        Qnp1[i] = Qstages[end][i]
        if error_estimate !== nothing
            error_estimate[i] = Qn[i] - Qnp1[i]
            @unroll for is in 1:Nstages
                error_estimate[i] += RKB_embedded[is] * dt * Rstages[is][i]
            end
        end
    end
end

function ERK23BogackiShampine(
    F,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    #! format: off
    RKA = [0       0       0       0; 
           1 // 2  0       0       0;
           0       3 // 4  0       0;
           2 // 9  1 // 3  4 // 9  0]
    #! format: on

    RKB = [2 // 9, 1 // 3, 4 // 9, 0]
    RKB_embedded = [7 // 24, 1 // 4, 1 // 3, 1 // 8]
    RKC = [0, 1 // 2, 3 // 4, 1]

    ExplicitFSALRungeKutta(
        F,
        3,
        2,
        RKA,
        RKB,
        RKB_embedded,
        RKC,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function ERK45DormandPrince(
    F,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    #! format: off
    RKA = [0              0              0               0            0              0        0; 
           1 // 5         0              0               0            0              0        0;
           3 // 40        9 // 40        0               0            0              0        0;
           44 // 45      -56 // 15       32 // 9         0            0              0        0;
           19372 // 6561 -25360 // 2187  64448 // 6561  -212 // 729   0              0        0;
           9017 // 3168  -355 // 33      46732 // 5247   49 // 176   -5103 // 18656  0        0;
           35//384        0              500 // 1113     125 // 192  -2187 // 6784   11 // 84 0]
    #! format: on

    RKB = [35 // 384, 0, 500 // 1113, 125 // 192, -2187 // 6784, 11 // 84, 0]
    RKB_embedded = [
        5179 // 57600,
        0,
        7571 // 16695,
        393 // 640,
        -92097 // 339200,
        187 // 2100,
        1 // 40,
    ]

    RKC = [0, 1 // 5, 3 // 10, 4 // 5, 8 // 9, 1, 1]

    ExplicitFSALRungeKutta(
        F,
        5,
        4,
        RKA,
        RKB,
        RKB_embedded,
        RKC,
        Q;
        dt = dt,
        t0 = t0,
    )
end
