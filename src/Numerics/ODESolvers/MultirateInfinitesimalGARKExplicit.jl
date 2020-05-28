export MRIGARKExplicit
export MRIGARKERK33aSandu, MRIGARKERK45aSandu

"""
    MRIParam(p, γs, Rs, ts, Δts)

Construct a type for passing the data around for the `MRIGARKExplicit` explicit
time stepper to follow on methods. `p` is the original user defined ODE
parameters, `γs` and `Rs` are the MRI parameters and stage values, respectively.
`ts` and `Δts` are the stage time and stage time step.
"""
struct MRIParam{P, T, AT, N, M}
    p::P
    γs::NTuple{M, SArray{NTuple{1, N}, T, 1, N}}
    Rs::NTuple{N, AT}
    ts::T
    Δts::T
    function MRIParam(
        p::P,
        γs::NTuple{M},
        Rs::NTuple{N, AT},
        ts,
        Δts,
    ) where {P, M, N, AT}
        T = eltype(γs[1])
        new{P, T, AT, N, M}(p, γs, Rs, ts, Δts)
    end
end

# We overload get property to access the original param
function Base.getproperty(mriparam::MRIParam, s::Symbol)
    if s === :p
        p = getfield(mriparam, :p)
        return p isa MRIParam ? p.p : p
    else
        getfield(mriparam, s)
    end
end

"""
    MRIGARKExplicit(f!, fastsolver, Γs, γ̂s, Q, Δt, t0)

Construct an explicit MultiRate Infinitesimal General-structure Additive
Runge--Kutta (MRI-GARK) scheme to solve

```math
    \\dot{y} = f(y, t) + g(y, t)
```

where `f` is the slow tendency function and `g` is the fast tendency function;
see Sandu (2019).

The fast tendency is integrated using the `fastsolver` and the slow tendency
using the MRI-GARK scheme. Namely, at each stage the scheme solves

```math
               v(T_i) &= Y_i \\\\
             \\dot{v} &= f(v, t) + \\sum_{j=1}^{i} \\bar{γ}_{ij}(t) R_j \\\\
    \\bar{γ}_{ijk}(t) &= \\sum_{k=0}^{NΓ-1} γ_{ijk} τ(t)^k / Δc_s \\\\
                 τ(t) &= (t - t_s) / Δt \\\\
              Y_{i+1} &= v(T_i + c_s * Δt)
```

where ``Y_1 = y_n`` and ``y_{n+1} = Y_{Nstages+1}``.

Here ``R_j = g(Y_j, t_0 + c_j * Δt)`` is the tendency for stage ``j``,
``γ_{ijk}`` are the GARK coupling coefficients,
``NΓ`` is the number of sets of GARK coupling coefficients there are
``Δc_s = \\sum_{j=1}^{Nstages} γ_{sj1} = c_{s+1} - c_s`` is the scaling
increment between stage times. The ODE for ``v(t)`` is solved using the
`fastsolver`.  Note that this form of the scheme is based on Definition 2.2 of
Sandu (2019), but ODE for ``v(t)`` is written to go from ``t_s`` to
``T_i + c_s * Δt`` as opposed to ``0`` to ``1``.

Currently only ['LowStorageRungeKutta2N`](@ref) schemes are supported for
`fastsolver`

The coefficients defined by `γ̂s` can be used for an embedded scheme (only the
last stage is different).

The available concrete implementations are:

  - [`MRIGARKERK33aSandu`](@ref)
  - [`MRIGARKERK45aSandu`](@ref)

### References

    @article{Sandu2019,
        title={A class of multirate infinitesimal gark methods},
        author={Sandu, Adrian},
        journal={SIAM Journal on Numerical Analysis},
        volume={57},
        number={5},
        pages={2300--2327},
        year={2019},
        publisher={SIAM},
        doi={10.1137/18M1205492}
    }
"""
mutable struct MRIGARKExplicit{T, RT, AT, Nstages, NΓ, FS, Nstages_sq} <:
               AbstractODESolver
    "time step"
    dt::RT
    "time"
    t::RT
    "rhs function"
    slowrhs!
    "Storage for RHS during the `MRIGARKExplicit` update"
    Rstages::NTuple{Nstages, AT}
    "RK coefficient matrices for coupling coefficients"
    Γs::NTuple{NΓ, SArray{NTuple{2, Nstages}, RT, 2, Nstages_sq}}
    "RK coefficient matrices for embedded scheme"
    γ̂s::NTuple{NΓ, SArray{NTuple{1, Nstages}, RT, 1, Nstages}}
    "RK coefficient vector C (time scaling)"
    Δc::SArray{NTuple{1, Nstages}, RT, 1, Nstages}
    "fast solver"
    fastsolver::FS

    function MRIGARKExplicit(
        slowrhs!,
        fastsolver,
        Γs,
        γ̂s,
        Q::AT,
        dt,
        t0,
    ) where {AT <: AbstractArray}
        NΓ = length(Γs)
        Nstages = size(Γs[1], 1)
        T = eltype(Q)
        RT = real(T)

        # Compute the Δc coefficients
        Δc = sum(Γs[1], dims = 2)[:]

        # Scale in the Δc to the Γ and γ̂, and convert to real type
        Γs = ntuple(k -> RT.(Γs[k] ./ Δc), NΓ)
        γ̂s = ntuple(k -> RT.(γ̂s[k] / Δc[Nstages]), NΓ)

        # Convert to real type
        Δc = RT.(Δc)

        # create storage for the stage values
        Rstages = ntuple(i -> similar(Q), Nstages)

        FS = typeof(fastsolver)
        new{T, RT, AT, Nstages, NΓ, FS, Nstages^2}(
            RT(dt),
            RT(t0),
            slowrhs!,
            Rstages,
            Γs,
            γ̂s,
            Δc,
            fastsolver,
        )
    end
end

function dostep!(Q, mrigark::MRIGARKExplicit, param, time::Real)
    dt = mrigark.dt
    fast = mrigark.fastsolver

    Rs = mrigark.Rstages
    Δc = mrigark.Δc
    Nstages = length(Δc)
    slowrhs! = mrigark.slowrhs!
    Γs = mrigark.Γs
    NΓ = length(Γs)

    ts = time
    groupsize = 256
    for s in 1:Nstages
        # Stage dt
        dts = Δc[s] * dt

        p = param isa MRIParam ? param.p : param
        slowrhs!(Rs[s], Q, p, ts, increment = false)
        if param isa MRIParam
            # fraction of the step slower stage increment we are on
            τ = (ts - param.ts) / param.Δts
            event = Event(array_device(Q))
            event = mri_update_rate!(array_device(Q), groupsize)(
                realview(Rs[s]),
                τ,
                param.γs,
                param.Rs;
                ndrange = length(realview(Rs[s])),
                dependencies = (event,),
            )
            wait(array_device(Q), event)
        end

        γs = ntuple(k -> ntuple(j -> Γs[k][s, j], s), NΓ)
        mriparam = MRIParam(param, γs, realview.(Rs[1:s]), ts, dts)
        updatetime!(mrigark.fastsolver, ts)
        solve!(Q, mrigark.fastsolver, mriparam; timeend = ts + dts)

        # update time
        ts += dts
    end
end

@kernel function mri_update_rate!(dQ, τ, γs, Rs)
    i = @index(Global, Linear)
    @inbounds begin
        NΓ = length(γs)
        Ns = length(γs[1])
        dqi = dQ[i]

        for s in 1:Ns
            ri = Rs[s][i]
            sc = γs[NΓ][s]
            for k in (NΓ - 1):-1:1
                sc = sc * τ + γs[k][s]
            end
            dqi += sc * ri
        end

        dQ[i] = dqi
    end
end

"""
    MRIGARKERK33aSandu(f!, fastsolver, Q; dt, t0 = 0, δ = -1 // 2)

The 3rd order, 3 stage scheme from Sandu (2019). The parameter `δ` defaults to
the value suggested by Sandu, but can be varied.
"""
function MRIGARKERK33aSandu(slowrhs!, fastsolver, Q; dt, t0 = 0, δ = -1 // 2)
    T = eltype(Q)
    RT = real(T)
    #! format: off
    Γ0 = [
                1 // 3           0 // 1          0 // 1
        (-6δ - 7) // 12  (6δ + 11) // 12         0 // 1
                0 // 1    (6δ - 5) // 12  (3 - 2δ) // 4
    ]
    γ̂0 = [     1 // 12          -1 // 3          7 // 12]

    Γ1 = [
               0 // 1          0 // 1   0 // 1
        (2δ + 1) // 2  -(2δ + 1) // 2   0 // 1
               1 // 2  -(2δ + 1) // 2   δ // 1
    ]
    γ̂1 = [     0 // 1          0 // 1   0 // 1]
    #! format: on
    MRIGARKExplicit(slowrhs!, fastsolver, (Γ0, Γ1), (γ̂0, γ̂1), Q, dt, t0)
end

"""
    MRIGARKERK45aSandu(f!, fastsolver, Q; dt, t0 = 0)

The 4th order, 5 stage scheme from Sandu (2019).
"""
function MRIGARKERK45aSandu(slowrhs!, fastsolver, Q; dt, t0 = 0)
    T = eltype(Q)
    RT = real(T)
    #! format: off
    Γ0 = [
                  1 // 5                 0 // 1                 0 // 1               0 // 1             0 // 1
                -53 // 16              281 // 80                0 // 1               0 // 1             0 // 1
         -36562993 // 71394880    34903117 // 17848720  -88770499 // 71394880        0 // 1             0 // 1
          -7631593 // 71394880  -166232021 // 35697440    6068517 // 1519040   8644289 // 8924360       0 // 1
            277061 // 303808       -209323 // 1139280    -1360217 // 1139280   -148789 // 56964    147889 // 45120
    ]
    γ̂0 = [-1482837 // 759520        175781 // 71205       -790577 // 1139280     -6379 // 56964        47 // 96]
    Γ1 = [
               0 // 1                0 // 1               0 // 1               0 // 1             0 // 1
             503 // 80            -503 // 80              0 // 1               0 // 1             0 // 1
        -1365537 // 35697440   4963773 // 7139488  -1465833 // 2231090         0 // 1             0 // 1
        66974357 // 35697440  21445367 // 7139488        -3 // 1        -8388609 // 4462180       0 // 1
          -18227 // 7520             2 // 1               1 // 1               5 // 1        -41933 // 7520
    ]
    γ̂1 = [  6213 // 1880         -6213 // 1880            0 // 1               0 // 1             0 // 1]
    #! format: on
    MRIGARKExplicit(slowrhs!, fastsolver, (Γ0, Γ1), (γ̂0, γ̂1), Q, dt, t0)
end
