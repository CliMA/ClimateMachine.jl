using NLsolve

struct MoistBubbleProfile
    Δz::Float64
    z::Array{Float64,1}
    Val::Array{Float64,2}
end

function MoistBubbleProfile(param_set::AbstractParameterSet, FT)

    _R_d::FT = R_d(param_set)
    _R_v::FT = R_v(param_set)
    _cp_d::FT = cp_d(param_set)
    _cp_v::FT = cp_v(param_set)
    _cp_l::FT = cp_l(param_set)
    p0::FT = MSLP(param_set)
    _grav::FT = grav(param_set)
    L00::FT = LH_v0(param_set) + (_cp_l - _cp_v) * T_freeze(param_set)
    _LH_v0::FT = LH_v0(param_set)

    n = 1000
    z = zeros(n + 1)
    Val = zeros(n + 1, 4)
    r_t0 = 2.e-2
    r_t0 = 0.0
    θ_e0 = 320
    θ_e0 = 300
    Δz = 10

    function resMoisture!(F, z, y, yPrime)
        p = y[1]
        ρ = y[2]
        T = y[3]
        r_t = y[4]
        r_v = y[5]
        ρ_qv = y[6]
        θ_e = y[7]
        pPrime = yPrime[1]
        F = zeros(7, 1)

        ρ_d = ρ / (1 + r_t)
        p_d = _R_d * ρ_d * T
        T_C = T - 273.15
        p_vs = saturation_vapor_pressure(param_set, T, _LH_v0, _cp_v - _cp_l)
        L = L00 - (_cp_l - _cp_v) * T
        F[1] = pPrime + _grav * ρ
        F[2] = p - (_R_d * ρ_d + _R_v * ρ_qv) * T
        F[3] =
            θ_e -
            T *
            (p_d / p0)^(-_R_d / (_cp_d + _cp_l * r_t)) *
            exp(L * r_v / ((_cp_d + _cp_l * r_t) * T))
        F[4] = r_t - r_t0
        F[5] = ρ_qv - ρ_d * r_v
        F[6] = θ_e - θ_e0
        a = p_vs / (_R_v * T) - ρ_qv
        b = ρ - ρ_qv - ρ_d
        F[7] = a + b - sqrt(a * a + b * b)
        return F
    end
    function setImplEuler(z, Δz, y0)
        function implEuler(y)
            return resMoisture!(F, z, y, (y - y0) / Δz)
        end
    end

    y = zeros(7)
    p = 1.e5
    ρ = 1.4
    r_t = r_t0
    r_v = r_t0
    ρ_qv = ρ * r_v
    θ_e = θ_e0
    T = θ_e

    yPrime = zeros(7)
    y0 = zeros(7)
    y0[1] = p
    y0[2] = ρ
    y0[3] = T
    y0[4] = r_t
    y0[5] = r_v
    y0[6] = ρ_qv
    y0[7] = θ_e


    z0 = 0.0
    Δz = 0.01
    y = deepcopy(y0)
    F = setImplEuler(z0, Δz, y0)
    res = nlsolve(F, y0)
    p = res.zero[1]
    ρ = res.zero[2]
    T = res.zero[3]
    r_t = res.zero[4]
    r_v = res.zero[5]
    ρ_qv = res.zero[6]
    θ_e = res.zero[7]
    ρ_d = ρ / (1 + r_t)
    ρ_qc = ρ - ρ_d - ρ_qv
    κ_M = (_R_d * ρ_d + _R_v * ρ_qv) / (_cp_d * ρ_d + _cp_v * ρ_qv + _cp_l * ρ_qc)
    ρ_θ = ρ * (p0 / p)^κ_M * T * (1 + (_R_v / _R_d) * r_v) / (1 + r_t)
    Val[1, 1] = ρ
    Val[1, 2] = ρ_θ
    Val[1, 3] = ρ_qv
    Val[1, 4] = ρ_qc
    Δz = 10.0
    z[1] = 0
    for i = 1:n
        #y0 = deepcopy(res.zero)
        copyto!(y0, res.zero)
        F = setImplEuler(z, Δz, y0)
        res = nlsolve(F, y0)
        z[i+1] = z[i] + Δz
        p = res.zero[1]
        ρ = res.zero[2]
        T = res.zero[3]
        r_t = res.zero[4]
        r_v = res.zero[5]
        ρ_qv = res.zero[6]
        θ_e = res.zero[7]
        ρ_d = ρ / (1 + r_t)
        ρ_qc = ρ - ρ_d - ρ_qv
        κ_M = (_R_d * ρ_d + _R_v * ρ_qv) / (_cp_d * ρ_d + _cp_v * ρ_qv + _cp_l * ρ_qc)
        ρ_θ = ρ * (p0 / p)^κ_M * T * (1 + (_R_v / _R_d) * r_v) / (1 + r_t)
        Val[i+1, 1] = ρ
        Val[i+1, 2] = ρ_θ
        Val[i+1, 3] = ρ_qv
        Val[i+1, 4] = ρ_qc
    end
    MoistBubbleProfile(Δz, z, Val)
end
