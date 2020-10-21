
export MultirateInfinitesimalStep,
    TimeScaledRHS,
    MISRK1,
    MIS2,
    MISRK2a,
    MISRK2b,
    MIS3C,
    MISRK3,
    MIS4,
    MIS4a,
    MISKWRK43,
    TVDMISA,
    TVDMISB,
    getnsubsteps

"""
    TimeScaledRHS(a, b, rhs!)

When evaluate at time `t`, evaluates `rhs!` at time `a + bt`.
"""
mutable struct TimeScaledRHS{N,RT}
    a::RT
    b::RT
    rhs!
    function TimeScaledRHS(a,b,rhs!)
    RT = typeof(a)
    if isa(rhs!, Tuple)
      N=length(rhs!)
    else
      N=1
    end
    new{N,RT}(a, b, rhs!)
    end
end

function (o::TimeScaledRHS{1,RT} where {RT})(dQ, Q, params, tau; increment)
  o.rhs!(dQ, Q, params, o.a + o.b * tau; increment = increment)
end
function (o::TimeScaledRHS{2,RT} where {RT})(dQ, Q, params, tau, i; increment)
  o.rhs![i](dQ, Q, params, o.a + o.b * tau; increment = increment)
end

mutable struct OffsetRHS{AT}
    offset::AT
    rhs!
    function OffsetRHS(offset, rhs!)
        AT = typeof(offset)
        new{AT}(offset, rhs!)
    end
end

function (o::OffsetRHS{AT} where {AT})(dQ, Q, params, tau; increment)
    o.rhs!(dQ, Q, params, tau; increment = increment)
    dQ .+= o.offset
end

"""
MultirateInfinitesimalStep(slowrhs!, fastrhs!, fastmethod,
                           α, β, γ,
                           Q::AT; dt=0, t0=0) where {AT<:AbstractArray}

This is a time stepping object for explicitly time stepping the partitioned differential
equation given by right-hand-side functions `f_fast` and `f_slow` with the state `Q`, i.e.,

```math
  \\dot{Q} = f_fast(Q, t) + f_slow(Q, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multirate infinitesimal step Runge-Kutta scheme
based on the provided `α`, `β` and `γ` tableaux and `fastmethod` for solving
the fast modes.

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
mutable struct MultirateInfinitesimalStep{
    T,
    RT,
    AT,
    FS,
    Nstages,
    Nstagesm1,
    Nstagesm2,
    Nstages_sq,
} <: AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "elapsed time steps"
    steps::Int
    "storage for y_n"
    yn::AT
    "Storage for ``Y_nj - y_n``"
    ΔYnj::NTuple{Nstagesm2, AT}
    "Storage for ``f(Y_nj)``"
    fYnj::NTuple{Nstagesm1, AT}
    "Storage for offset"
    offset::AT
    "slow rhs function"
    slowrhs!
    "RHS for fast solver"
    tsfastrhs!::TimeScaledRHS{N,RT} where N
    "fast rhs method"
    fastsolver::FS
    "number of substeps per stage"
    nsubsteps::Int
    α::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    β::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    γ::SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}
    d::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
    c::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
    c̃::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
    function MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q::AT;
        dt = 0,
        t0 = 0,
    ) where {AT <: AbstractArray}

        T = eltype(Q)
        RT = real(T)

        Nstages = size(α, 1)

        yn = similar(Q)
        ΔYnj = ntuple(_ -> similar(Q), Nstages - 2)
        fYnj = ntuple(_ -> similar(Q), Nstages - 1)
        offset = similar(Q)
        tsfastrhs! = TimeScaledRHS(RT(0), RT(0), fastrhs!)
        fastsolver = fastmethod(tsfastrhs!, Q)

        d = sum(β, dims = 2)

        c = similar(d)
        for i in eachindex(c)
            c[i] = d[i]
            if i > 1
                c[i] += sum(j -> (α[i, j] + γ[i, j]) * c[j], 1:(i - 1))
            end
        end
        c̃ = α * c

        new{
            T,
            RT,
            AT,
            typeof(fastsolver),
            Nstages,
            Nstages - 1,
            Nstages - 2,
            Nstages^2,
        }(
            RT(dt),
            RT(t0),
            0,
            yn,
            ΔYnj,
            fYnj,
            offset,
            slowrhs!,
            tsfastrhs!,
            fastsolver,
            nsubsteps,
            α,
            β,
            γ,
            d,
            c,
            c̃,
        )
    end
end

function MultirateInfinitesimalStep(
    mis,
    op::TimeScaledRHS{2,RT} where {RT},
    fastmethod,
    Q = nothing;
    dt = 0,
    t0 = 0,
    nsubsteps = 1,
) where {AT<:AbstractArray}

    return mis(
        op.rhs![1],
        op.rhs![2],
        fastmethod,
        nsubsteps,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function dostep!(
    Q,
    mis::MultirateInfinitesimalStep,
    p,
    time::Real,
    nsubsteps::Int,
    iStage::Int,
    slow_δ = nothing,
    slow_rv_dQ = nothing,
    slow_scaling = nothing,
)
    if isa(mis.slowrhs!, OffsetRHS{AT} where {AT})
        mis.slowrhs!.offset = slow_rv_dQ
    else
        mis.slowrhs! = OffsetRHS(slow_rv_dQ, mis.slowrhs!)
    end
    for i = 1:nsubsteps
        dostep!(Q, mis, p, time)
    end
end

function dostep!(Q, mis::MultirateInfinitesimalStep, p, time)
    dt = mis.dt
    FT = eltype(dt)
    α = mis.α
    β = mis.β
    γ = mis.γ
    yn = mis.yn
    ΔYnj = mis.ΔYnj
    fYnj = mis.fYnj
    offset = mis.offset
    d = mis.d
    c = mis.c
    c̃ = mis.c̃
    slowrhs! = mis.slowrhs!
    fastsolver = mis.fastsolver
    fastrhs! = mis.tsfastrhs!
    nsubsteps = mis.nsubsteps

    nstages = size(α, 1)

    copyto!(yn, Q) # first stage
    for i in 2:nstages
        slowrhs!(fYnj[i - 1], Q, p, time + c[i - 1] * dt, increment = false)

        groupsize = 256
        event = Event(array_device(Q))
        if abs(d[i]) < 1.e-10
            event = update!(array_device(Q), groupsize)(
                realview(Q),
                realview(offset),
                Val(i),
                realview(yn),
                map(realview, ΔYnj[1:(i - 2)]),
                map(realview, fYnj[1:(i - 1)]),
                α[i, :],
                β[i, :],
                γ[i, :],
                dt;
                ndrange = length(realview(Q)),
                dependencies = (event,),
            )
            wait(array_device(Q), event)
            Q .+= dt.*offset
        else
            event = update!(array_device(Q), groupsize)(
                realview(Q),
                realview(offset),
                Val(i),
                realview(yn),
                map(realview, ΔYnj[1:(i - 2)]),
                map(realview, fYnj[1:(i - 1)]),
                α[i, :],
                β[i, :],
                γ[i, :],
                d[i],
                dt;
                ndrange = length(realview(Q)),
                dependencies = (event,),
            )
            wait(array_device(Q), event)

            fastrhs!.a = time + c̃[i] * dt
            fastrhs!.b = (c[i] - c̃[i]) / d[i]

            τ = zero(FT)
            nsubstepsLoc=ceil(Int,nsubsteps*d[i]);
            dτ = d[i] * dt / nsubstepsLoc
            updatetime!(fastsolver, τ)
            updatedt!(fastsolver, dτ)
            # TODO: we want to be able to write
            #   solve!(Q, fastsolver, p; numberofsteps = mis.nsubsteps)  #(1c)
            # especially if we want to use StormerVerlet, but need some way to pass in `offset`
            dostep!(Q, fastsolver, p, τ, nsubstepsLoc, i, FT(1), realview(offset), nothing)  #(1c)
        end
    end
end

@kernel function update!(
    Q,
    offset,
    ::Val{i},
    yn,
    ΔYnj,
    fYnj,
    αi,
    βi,
    γi,
    dt,
) where {i}
    e = @index(Global, Linear)
    @inbounds begin
        if i > 2
            ΔYnj[i - 2][e] = Q[e] - yn[e] # is 0 for i == 2
        end
        Q[e] = yn[e] # (1a)
        offset[e] = (βi[1]) .* fYnj[1][e] # (1b)
        @unroll for j in 2:(i - 1)
            Q[e] += αi[j] .* ΔYnj[j - 1][e] # (1a cont.)
            offset[e] +=
                (γi[j] / dt) * ΔYnj[j - 1][e] +
                βi[j] * fYnj[j][e] # (1b cont.)
        end
    end
end

@kernel function update!(
    Q,
    offset,
    ::Val{i},
    yn,
    ΔYnj,
    fYnj,
    αi,
    βi,
    γi,
    d_i,
    dt,
) where {i}
    e = @index(Global, Linear)
    @inbounds begin
        if i > 2
            ΔYnj[i - 2][e] = Q[e] - yn[e] # is 0 for i == 2
        end
        Q[e] = yn[e] # (1a)
        offset[e] = (βi[1] / d_i) .* fYnj[1][e] # (1b)
        @unroll for j in 2:(i - 1)
            Q[e] += αi[j] .* ΔYnj[j - 1][e] # (1a cont.)
            offset[e] +=
                (γi[j] / (d_i * dt)) * ΔYnj[j - 1][e] +
                (βi[j] / d_i) * fYnj[j][e] # (1b cont.)
        end
    end
end

function MISRK1(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT<:AbstractArray}
    FT = eltype(Q)
    RT = real(FT)
    α = zeros(2, 2)
    β = beta(MISRK1, RT)
    γ = zeros(2, 2)
    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MISRK1), RT::DataType)
    β = [
        0 0
        1 0
    ]
end

function MIS2(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0 0
        0 0 0 0
        0 RT(0.536946566710) 0 0
        0 RT(0.480892968551) RT(0.500561163566) 0
    ]

    β = beta(MIS2, RT)

    γ = [
        0 0 0 0
        0 0 0 0
        0 RT(0.652465126004) 0 0
        0 RT(-0.0732769849457) RT(0.144902430420) 0
    ]

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MIS2), RT::DataType)
    β = [
        0 0 0 0
        RT(0.126848494553) 0 0 0
        RT(-0.784838278826) RT(1.37442675268) 0 0
        RT(-0.0456727081749) RT(-0.00875082271190) RT(0.524775788629) 0
    ]
end

function MISRK2a(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT<:AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0
        0 0 0
        0 1 0
    ]

    β = beta(MISRK2a, RT)

    γ = zeros(3, 3)

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MISRK2a), RT::DataType)
    β = [
        0 0 0
        RT(0.5) 0 0
        RT(-0.5) 1 0
    ]
end

function MISRK2b(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT<:AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0
        0 0 0
        0 1 0
    ]

    β = beta(MISRK2b, RT)

    γ = zeros(3, 3)

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MISRK2b), RT::DataType)
    β = [
        0 0 0
        1 0 0
        RT(-0.5) RT(0.5) 0
    ]
end

function MIS3C(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0 0
        0 0 0 0
        0 RT(0.589557277145) 0 0
        0 RT(0.544036601551) RT(0.565511042564) 0
    ]

    β = beta(MIS3C, RT)

    γ = [
        0 0 0 0
        0 0 0 0
        0 RT(0.142798786398) 0 0
        0 RT(-0.0428918957402) RT(0.0202720980282) 0
    ]

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MIS3C), RT::DataType)
    β = [
        0 0 0 0
        RT(0.397525189225) 0 0 0
        RT(-0.227036463644) RT(0.624528794618) 0 0
        RT(-0.00295238076840) RT(-0.270971764284) RT(0.671323159437) 0
    ]
end

function MISRK3(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT<:AbstractArray}
    FT = eltype(Q)
    RT = real(FT)
    α = zeros(4, 4)
    β = beta(MISRK3, RT)
    γ = zeros(4, 4)
    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MISRK3), RT::DataType)
    β = [
        0 0 0 0
        RT(0.3333333333333333) 0 0 0
        0 RT(0.5) 0 0
        0 0 1 0
    ]
end

function MIS4(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0 0 0
        0 0 0 0 0
        0 RT(0.914092810304) 0 0 0
        0 RT(1.14274417397) RT(-0.295211246188) 0 0
        0 RT(0.112965282231) RT(0.337369411296) RT(0.503747183119) 0
    ]

    β = beta(MIS4, RT)

    γ = [
        0 0 0 0 0
        0 0 0 0 0
        0 RT(0.678951983291) 0 0 0
        0 RT(-1.38974164070) RT(0.503864576302) 0 0
        0 RT(-0.375328608282) RT(0.320925021109) RT(-0.158259688945) 0
    ]

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MIS4), RT::DataType)
    β = [
        0 0 0 0 0
        RT(0.136296478423) 0 0 0 0
        RT(0.280462398979) RT(-0.0160351333596) 0 0 0
        RT(0.904713355208) RT(-1.04011183154) RT(0.652337563489) 0 0
        RT(0.0671969845546) RT(-0.365621862610) RT(-0.154861470835) RT(0.970362444469) 0
    ]
end

function MIS4a(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0 0 0
        0 0 0 0 0
        0 RT(0.52349249922385610) 0 0 0
        0 RT(1.1683374366893629) RT(-0.75762080241712637) 0 0
        0 RT(-0.036477233846797109) RT(0.56936148730740477) RT(0.47746263002599681) 0
    ]

    # β[5,1] in the paper is incorrect
    # the correct value is used below (from authors)
    β = beta(MIS4a, RT)

    γ = [
        0 0 0 0 0
        0 0 0 0 0
        0 RT(0.13145089796226542) 0 0 0
        0 RT(-0.36855857648747881) RT(0.33159232636600550) 0 0
        0 RT(-0.065767130537473045) RT(0.040591093109036858) RT(0.064902111640806712) 0
    ]

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MIS4a), RT::DataType)
    β = [
        0 0 0 0 0
        RT(0.38758444641450318) 0 0 0 0
        RT(-0.025318448354142823) RT(0.38668943087310403) 0 0 0
        RT(0.20899983523553325) RT(-0.45856648476371231) RT(0.43423187573425748) 0 0
        RT(-0.10048822195663100) RT(-0.46186171956333327) RT(0.83045062122462809) RT(0.27014914900250392) 0
    ]
end

function MISKWRK43(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT<:AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0 0 0
        0 0 0 0 0
        0 1 0 0 0
        0 0 1 0 0
        0 0 0 1 0
    ]

    β = beta(MISKWRK43, RT)

    γ = zeros(5, 5)

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(MISKWRK43), RT::DataType)
    β = [
        0 0 0 0 0
        0.5 0 0 0 0
        -RT(2 // 3) RT(2 // 3) 0 0 0
        RT(0.5) -1 1 0 0
        -RT(1 // 6) RT(2 // 3) -RT(2 // 3) RT(1 // 6) 0
    ]
end

function TVDMISA(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0 0
        0 0 0 0
        0 RT(0.1946360605647457) 0 0
        0 RT(0.3971200136786614) RT(0.2609434606211801) 0
    ]

    β = beta(TVDMISA, RT)

    γ = [
        0 0 0 0
        0 0 0 0
        0 RT(0.5624048933209129) 0 0
        0 RT(0.4408467475713277) RT(-0.2459300561692391) 0
    ]

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(TVDMISA), RT::DataType)
    β = [
        0 0 0 0
        RT(2 // 3) 0 0 0
        RT(-0.28247174703488398) RT(4 // 9) 0 0
        RT(-0.31198081960042401) RT(0.18082737579913699) RT(9 // 16) 0
    ]
end

function TVDMISB(
    slowrhs!,
    fastrhs!,
    fastmethod,
    nsubsteps,
    Q::AT;
    dt = 0,
    t0 = 0,
) where {AT <: AbstractArray}
    FT = eltype(Q)
    RT = real(FT)

    α = [
        0 0 0 0
        0 0 0 0
        0 RT(0.42668232863311001) 0 0
        0 RT(0.26570779016173801) RT(0.41489966891866698) 0
    ]

    β = beta(TVDMISB, RT)

    γ = [
        0 0 0 0
        0 0 0 0
        0 RT(0.28904389120139701) 0 0
        0 RT(0.45113560071334202) RT(-0.25006656847591002) 0
    ]

    MultirateInfinitesimalStep(
        slowrhs!,
        fastrhs!,
        fastmethod,
        nsubsteps,
        α,
        β,
        γ,
        Q;
        dt = dt,
        t0 = t0,
    )
end

function beta(::typeof(TVDMISB), RT::DataType)
    β = [
        0 0 0 0
        RT(2 // 3) 0 0 0
        RT(-0.25492859100078202) RT(4 // 9) 0 0
        RT(-0.26452517179288798) RT(0.11424084424766399) RT(9 // 16) 0
    ]
end

function getnsubsteps(mis, ns::Int, RT::DataType)
  d = sum(beta(mis, RT), dims = 2)
  return d ./ ns
end
