const γ_exact = 1.4
const μ_exact = 1 // 100
@noinline ρ_g(t, x, y, z, ::Val{2}) =
    sin(pi * x) * cos(pi * t) * cos(pi * y) + 3
@noinline U_g(t, x, y, z, ::Val{2}) =
    (sin(pi * x) * cos(pi * t) * cos(pi * y) + 3) *
    sin(pi * x) *
    cos(pi * t) *
    cos(pi * y)
@noinline V_g(t, x, y, z, ::Val{2}) =
    (sin(pi * x) * cos(pi * t) * cos(pi * y) + 3) *
    sin(pi * x) *
    cos(pi * t) *
    cos(pi * y)
@noinline W_g(t, x, y, z, ::Val{2}) = 0
@noinline E_g(t, x, y, z, ::Val{2}) =
    sin(pi * x) * cos(pi * t) * cos(pi * y) + 100
@noinline Sρ_g(t, x, y, z, ::Val{2}) =
    pi * (
        2 * sin(2 * pi * x) - 2 * sin(2 * pi * y) - sin(pi * (2 * t - 2 * x)) +
        sin(pi * (2 * t + 2 * x)) +
        sin(pi * (2 * t - 2 * y)) - sin(pi * (2 * t + 2 * y)) +
        2 * sin(pi * (2 * x + 2 * y)) +
        sin(pi * (-2 * t + 2 * x + 2 * y)) +
        sin(pi * (2 * t + 2 * x + 2 * y)) +
        10 * cos(pi * (-t + x + y)) - 2 * cos(pi * (t - x + y)) +
        2 * cos(pi * (t + x - y)) +
        14 * cos(pi * (t + x + y))
    ) / 8
@noinline SU_g(t, x, y, z, ::Val{2}) =
    pi * (
        -2.0 * sin(pi * t) * sin(pi * x)^2 * cos(pi * t) * cos(pi * y)^2 -
        3.0 * sin(pi * t) * sin(pi * x) * cos(pi * y) -
        3.0 * sin(pi * x)^3 * sin(pi * y) * cos(pi * t)^3 * cos(pi * y)^2 -
        6.0 * sin(pi * x)^2 * sin(pi * y) * cos(pi * t)^2 * cos(pi * y) +
        1.8 * sin(pi * x)^2 * cos(pi * t)^3 * cos(pi * x) * cos(pi * y)^3 +
        3.6 * sin(pi * x) * cos(pi * t)^2 * cos(pi * x) * cos(pi * y)^2 +
        0.0233333333333333 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        0.00333333333333333 * pi * sin(pi * y) * cos(pi * t) * cos(pi * x) +
        0.4 * cos(pi * t) * cos(pi * x) * cos(pi * y)
    )
@noinline SV_g(t, x, y, z, ::Val{2}) =
    pi * (
        -2.0 * sin(pi * t) * sin(pi * x)^2 * cos(pi * t) * cos(pi * y)^2 -
        3.0 * sin(pi * t) * sin(pi * x) * cos(pi * y) -
        1.8 * sin(pi * x)^3 * sin(pi * y) * cos(pi * t)^3 * cos(pi * y)^2 -
        3.6 * sin(pi * x)^2 * sin(pi * y) * cos(pi * t)^2 * cos(pi * y) +
        3.0 * sin(pi * x)^2 * cos(pi * t)^3 * cos(pi * x) * cos(pi * y)^3 -
        0.4 * sin(pi * x) * sin(pi * y) * cos(pi * t) +
        6.0 * sin(pi * x) * cos(pi * t)^2 * cos(pi * x) * cos(pi * y)^2 +
        0.0233333333333333 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        0.00333333333333333 * pi * sin(pi * y) * cos(pi * t) * cos(pi * x)
    )
@noinline SW_g(t, x, y, z, ::Val{2}) = 0
@noinline SE_g(t, x, y, z, ::Val{2}) =
    pi * (
        -1.0 * sin(pi * t) * sin(pi * x) * cos(pi * y) +
        1.6 * sin(pi * x)^4 * sin(pi * y) * cos(pi * t)^4 * cos(pi * y)^3 +
        3.6 * sin(pi * x)^3 * sin(pi * y) * cos(pi * t)^3 * cos(pi * y)^2 -
        1.6 * sin(pi * x)^3 * cos(pi * t)^4 * cos(pi * x) * cos(pi * y)^4 -
        0.0233333333333333 *
        pi *
        sin(pi * x)^2 *
        sin(pi * y)^2 *
        cos(pi * t)^2 -
        2.8 * sin(pi * x)^2 * sin(pi * y) * cos(pi * t)^2 * cos(pi * y) -
        3.6 * sin(pi * x)^2 * cos(pi * t)^3 * cos(pi * x) * cos(pi * y)^3 +
        0.0466666666666667 *
        pi *
        sin(pi * x)^2 *
        cos(pi * t)^2 *
        cos(pi * y)^2 +
        0.0133333333333333 *
        pi *
        sin(pi * x) *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y) - 140.0 * sin(pi * x) * sin(pi * y) * cos(pi * t) +
        2.8 * sin(pi * x) * cos(pi * t)^2 * cos(pi * x) * cos(pi * y)^2 -
        0.0233333333333333 *
        pi *
        cos(pi * t)^2 *
        cos(pi * x)^2 *
        cos(pi * y)^2 + 140.0 * cos(pi * t) * cos(pi * x) * cos(pi * y)
    )
@noinline ρ_g(t, x, y, z, ::Val{3}) =
    sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) + 3
@noinline U_g(t, x, y, z, ::Val{3}) =
    (sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) + 3) *
    sin(pi * x) *
    cos(pi * t) *
    cos(pi * y) *
    cos(pi * z)
@noinline V_g(t, x, y, z, ::Val{3}) =
    (sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) + 3) *
    sin(pi * x) *
    cos(pi * t) *
    cos(pi * y) *
    cos(pi * z)
@noinline W_g(t, x, y, z, ::Val{3}) =
    (sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) + 3) *
    sin(pi * x) *
    sin(pi * z) *
    cos(pi * t) *
    cos(pi * y)
@noinline E_g(t, x, y, z, ::Val{3}) =
    sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) + 100
@noinline Sρ_g(t, x, y, z, ::Val{3}) =
    pi * (
        -sin(pi * t) * sin(pi * x) * cos(pi * y) * cos(pi * z) -
        2 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z)^2 -
        3 * sin(pi * x) * sin(pi * y) * cos(pi * t) * cos(pi * z) +
        2 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        3 * sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) +
        3 * cos(pi * t) * cos(pi * x) * cos(pi * y) * cos(pi * z)
    )
@noinline SU_g(t, x, y, z, ::Val{3}) =
    pi * (
        -2.0 *
        sin(pi * t) *
        sin(pi * x)^2 *
        cos(pi * t) *
        cos(pi * y)^2 *
        cos(pi * z)^2 -
        3.0 * sin(pi * t) * sin(pi * x) * cos(pi * y) * cos(pi * z) -
        3.0 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^3 -
        2.0 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^3 *
        cos(pi * z) +
        1.0 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^3 -
        6.0 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        0.6 *
        sin(pi * x)^2 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z) -
        3.0 * sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        1.8 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^3 +
        3.0 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z)^2 -
        1.2 *
        sin(pi * x) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 +
        3.6 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        0.0333333333333333 *
        pi *
        sin(pi * x) *
        cos(pi * t) *
        cos(pi * y) *
        cos(pi * z) +
        0.00333333333333333 *
        pi *
        sin(pi * y) *
        cos(pi * t) *
        cos(pi * x) *
        cos(pi * z) -
        0.00333333333333333 *
        pi *
        cos(pi * t) *
        cos(pi * x) *
        cos(pi * y) *
        cos(pi * z) +
        0.4 * cos(pi * t) * cos(pi * x) * cos(pi * y) * cos(pi * z)
    )
@noinline SV_g(t, x, y, z, ::Val{3}) =
    pi * (
        -2.0 *
        sin(pi * t) *
        sin(pi * x)^2 *
        cos(pi * t) *
        cos(pi * y)^2 *
        cos(pi * z)^2 -
        3.0 * sin(pi * t) * sin(pi * x) * cos(pi * y) * cos(pi * z) +
        0.6 *
        sin(pi * x)^3 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z) -
        1.8 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^3 -
        2.0 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^3 *
        cos(pi * z) +
        1.0 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^3 +
        1.2 *
        sin(pi * x)^2 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * y) -
        3.6 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        3.0 * sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        3.0 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^3 +
        3.0 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z)^2 -
        0.4 * sin(pi * x) * sin(pi * y) * cos(pi * t) * cos(pi * z) +
        0.00333333333333333 *
        pi *
        sin(pi * x) *
        sin(pi * y) *
        cos(pi * t) *
        cos(pi * z) +
        6.0 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        0.0333333333333333 *
        pi *
        sin(pi * x) *
        cos(pi * t) *
        cos(pi * y) *
        cos(pi * z) +
        0.00333333333333333 *
        pi *
        sin(pi * y) *
        cos(pi * t) *
        cos(pi * x) *
        cos(pi * z)
    )
@noinline SW_g(t, x, y, z, ::Val{3}) =
    pi *
    (
        -2.0 *
        sin(pi * t) *
        sin(pi * x)^2 *
        cos(pi * t) *
        cos(pi * y)^2 *
        cos(pi * z) - 3.0 * sin(pi * t) * sin(pi * x) * cos(pi * y) -
        3.0 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^2 -
        0.8 * sin(pi * x)^3 * sin(pi * z)^2 * cos(pi * t)^3 * cos(pi * y)^3 +
        2.8 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^2 -
        6.0 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z) +
        3.0 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^2 +
        7.2 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z) -
        0.00333333333333333 * pi * sin(pi * x) * sin(pi * y) * cos(pi * t) +
        6.0 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z) - 0.4 * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        0.0333333333333333 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        0.00333333333333333 * pi * cos(pi * t) * cos(pi * x) * cos(pi * y)
    ) *
    sin(pi * z)
@noinline SE_g(t, x, y, z, ::Val{3}) =
    pi * (
        0.001171875 *
        (1 - cos(2 * pi * x))^2 *
        (1 - cos(4 * pi * z)) *
        (cos(2 * pi * t) + 1)^2 *
        (cos(2 * pi * y) + 1)^2 -
        1.0 * sin(pi * t) * sin(pi * x) * cos(pi * y) * cos(pi * z) +
        0.8 *
        sin(pi * x)^4 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^4 *
        cos(pi * y)^3 *
        cos(pi * z)^2 +
        1.6 *
        sin(pi * x)^4 *
        sin(pi * y) *
        cos(pi * t)^4 *
        cos(pi * y)^3 *
        cos(pi * z)^4 +
        0.2 * sin(pi * x)^4 * sin(pi * z)^4 * cos(pi * t)^4 * cos(pi * y)^4 -
        0.4 * sin(pi * x)^4 * cos(pi * t)^4 * cos(pi * y)^4 * cos(pi * z)^4 +
        1.8 *
        sin(pi * x)^3 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z) +
        3.6 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^3 -
        0.8 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^4 *
        cos(pi * x) *
        cos(pi * y)^4 *
        cos(pi * z)^2 +
        0.6 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^3 *
        cos(pi * z) -
        1.6 *
        sin(pi * x)^3 *
        cos(pi * t)^4 *
        cos(pi * x) *
        cos(pi * y)^4 *
        cos(pi * z)^4 -
        1.2 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^3 -
        0.01 *
        pi *
        sin(pi * x)^2 *
        sin(pi * y)^2 *
        sin(pi * z)^2 *
        cos(pi * t)^2 -
        0.0233333333333333 *
        pi *
        sin(pi * x)^2 *
        sin(pi * y)^2 *
        cos(pi * t)^2 *
        cos(pi * z)^2 -
        0.0233333333333333 *
        pi *
        sin(pi * x)^2 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * y) -
        2.8 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        0.01 *
        pi *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        1.8 *
        sin(pi * x)^2 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z) -
        1.4 * sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        0.0133333333333333 *
        pi *
        sin(pi * x)^2 *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * y)^2 -
        3.6 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^3 +
        0.0533333333333333 *
        pi *
        sin(pi * x)^2 *
        cos(pi * t)^2 *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        1.4 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z)^2 +
        0.0133333333333333 *
        pi *
        sin(pi * x) *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y) *
        cos(pi * z)^2 -
        140.0 * sin(pi * x) * sin(pi * y) * cos(pi * t) * cos(pi * z) +
        0.0233333333333333 *
        pi *
        sin(pi * x) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 +
        0.01 *
        pi *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        2.8 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        140.0 * sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) -
        0.01 *
        pi *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * x)^2 *
        cos(pi * y)^2 -
        0.0233333333333333 *
        pi *
        cos(pi * t)^2 *
        cos(pi * x)^2 *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        140.0 * cos(pi * t) * cos(pi * x) * cos(pi * y) * cos(pi * z)
    )
