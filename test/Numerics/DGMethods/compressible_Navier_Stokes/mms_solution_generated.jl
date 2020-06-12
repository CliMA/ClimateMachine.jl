const γ_exact = 7 // 5
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
        -600 * sin(pi * t) * sin(pi * x)^2 * cos(pi * t) * cos(pi * y)^2 -
        900 * sin(pi * t) * sin(pi * x) * cos(pi * y) -
        900 * sin(pi * x)^3 * sin(pi * y) * cos(pi * t)^3 * cos(pi * y)^2 -
        1800 * sin(pi * x)^2 * sin(pi * y) * cos(pi * t)^2 * cos(pi * y) +
        540 * sin(pi * x)^2 * cos(pi * t)^3 * cos(pi * x) * cos(pi * y)^3 +
        1080 * sin(pi * x) * cos(pi * t)^2 * cos(pi * x) * cos(pi * y)^2 +
        7 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        pi * sin(pi * y) * cos(pi * t) * cos(pi * x) +
        120 * cos(pi * t) * cos(pi * x) * cos(pi * y)
    ) / 300
@noinline SV_g(t, x, y, z, ::Val{2}) =
    pi * (
        -600 * sin(pi * t) * sin(pi * x)^2 * cos(pi * t) * cos(pi * y)^2 -
        900 * sin(pi * t) * sin(pi * x) * cos(pi * y) -
        540 * sin(pi * x)^3 * sin(pi * y) * cos(pi * t)^3 * cos(pi * y)^2 -
        1080 * sin(pi * x)^2 * sin(pi * y) * cos(pi * t)^2 * cos(pi * y) +
        900 * sin(pi * x)^2 * cos(pi * t)^3 * cos(pi * x) * cos(pi * y)^3 -
        120 * sin(pi * x) * sin(pi * y) * cos(pi * t) +
        1800 * sin(pi * x) * cos(pi * t)^2 * cos(pi * x) * cos(pi * y)^2 +
        7 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        pi * sin(pi * y) * cos(pi * t) * cos(pi * x)
    ) / 300
@noinline SW_g(t, x, y, z, ::Val{2}) = 0
@noinline SE_g(t, x, y, z, ::Val{2}) =
    pi * (
        -300 * sin(pi * t) * sin(pi * x) * cos(pi * y) +
        480 * sin(pi * x)^4 * sin(pi * y) * cos(pi * t)^4 * cos(pi * y)^3 +
        1080 * sin(pi * x)^3 * sin(pi * y) * cos(pi * t)^3 * cos(pi * y)^2 -
        480 * sin(pi * x)^3 * cos(pi * t)^4 * cos(pi * x) * cos(pi * y)^4 -
        7 * pi * sin(pi * x)^2 * sin(pi * y)^2 * cos(pi * t)^2 -
        840 * sin(pi * x)^2 * sin(pi * y) * cos(pi * t)^2 * cos(pi * y) -
        1080 * sin(pi * x)^2 * cos(pi * t)^3 * cos(pi * x) * cos(pi * y)^3 +
        14 * pi * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        4 *
        pi *
        sin(pi * x) *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y) - 42000 * sin(pi * x) * sin(pi * y) * cos(pi * t) +
        840 * sin(pi * x) * cos(pi * t)^2 * cos(pi * x) * cos(pi * y)^2 -
        7 * pi * cos(pi * t)^2 * cos(pi * x)^2 * cos(pi * y)^2 +
        42000 * cos(pi * t) * cos(pi * x) * cos(pi * y)
    ) / 300
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
        -600 *
        sin(pi * t) *
        sin(pi * x)^2 *
        cos(pi * t) *
        cos(pi * y)^2 *
        cos(pi * z)^2 -
        900 * sin(pi * t) * sin(pi * x) * cos(pi * y) * cos(pi * z) -
        900 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^3 -
        600 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^3 *
        cos(pi * z) +
        300 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^3 -
        1800 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        180 *
        sin(pi * x)^2 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z) -
        900 * sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        540 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^3 +
        900 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z)^2 -
        360 *
        sin(pi * x) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 +
        1080 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        10 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) +
        pi * sin(pi * y) * cos(pi * t) * cos(pi * x) * cos(pi * z) -
        pi * cos(pi * t) * cos(pi * x) * cos(pi * y) * cos(pi * z) +
        120 * cos(pi * t) * cos(pi * x) * cos(pi * y) * cos(pi * z)
    ) / 300
@noinline SV_g(t, x, y, z, ::Val{3}) =
    pi * (
        -600 *
        sin(pi * t) *
        sin(pi * x)^2 *
        cos(pi * t) *
        cos(pi * y)^2 *
        cos(pi * z)^2 -
        900 * sin(pi * t) * sin(pi * x) * cos(pi * y) * cos(pi * z) +
        180 *
        sin(pi * x)^3 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z) -
        540 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^3 -
        600 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^3 *
        cos(pi * z) +
        300 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^3 +
        360 *
        sin(pi * x)^2 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * y) -
        1080 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        900 * sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        900 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^3 +
        900 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z)^2 -
        120 * sin(pi * x) * sin(pi * y) * cos(pi * t) * cos(pi * z) +
        pi * sin(pi * x) * sin(pi * y) * cos(pi * t) * cos(pi * z) +
        1800 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        10 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) +
        pi * sin(pi * y) * cos(pi * t) * cos(pi * x) * cos(pi * z)
    ) / 300
@noinline SW_g(t, x, y, z, ::Val{3}) =
    pi *
    (
        -600 *
        sin(pi * t) *
        sin(pi * x)^2 *
        cos(pi * t) *
        cos(pi * y)^2 *
        cos(pi * z) - 900 * sin(pi * t) * sin(pi * x) * cos(pi * y) -
        900 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^2 -
        240 * sin(pi * x)^3 * sin(pi * z)^2 * cos(pi * t)^3 * cos(pi * y)^3 +
        840 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^2 -
        1800 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z) +
        900 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^2 +
        2160 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z) -
        pi * sin(pi * x) * sin(pi * y) * cos(pi * t) +
        1800 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z) - 120 * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        10 * pi * sin(pi * x) * cos(pi * t) * cos(pi * y) +
        pi * cos(pi * t) * cos(pi * x) * cos(pi * y)
    ) *
    sin(pi * z) / 300
@noinline SE_g(t, x, y, z, ::Val{3}) =
    pi * (
        45 *
        (1 - cos(2 * pi * x))^2 *
        (1 - cos(4 * pi * z)) *
        (cos(2 * pi * t) + 1)^2 *
        (cos(2 * pi * y) + 1)^2 / 128 -
        300 * sin(pi * t) * sin(pi * x) * cos(pi * y) * cos(pi * z) +
        240 *
        sin(pi * x)^4 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^4 *
        cos(pi * y)^3 *
        cos(pi * z)^2 +
        480 *
        sin(pi * x)^4 *
        sin(pi * y) *
        cos(pi * t)^4 *
        cos(pi * y)^3 *
        cos(pi * z)^4 +
        60 * sin(pi * x)^4 * sin(pi * z)^4 * cos(pi * t)^4 * cos(pi * y)^4 -
        120 * sin(pi * x)^4 * cos(pi * t)^4 * cos(pi * y)^4 * cos(pi * z)^4 +
        540 *
        sin(pi * x)^3 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z) +
        1080 *
        sin(pi * x)^3 *
        sin(pi * y) *
        cos(pi * t)^3 *
        cos(pi * y)^2 *
        cos(pi * z)^3 -
        240 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^4 *
        cos(pi * x) *
        cos(pi * y)^4 *
        cos(pi * z)^2 +
        180 *
        sin(pi * x)^3 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * y)^3 *
        cos(pi * z) -
        480 *
        sin(pi * x)^3 *
        cos(pi * t)^4 *
        cos(pi * x) *
        cos(pi * y)^4 *
        cos(pi * z)^4 -
        360 * sin(pi * x)^3 * cos(pi * t)^3 * cos(pi * y)^3 * cos(pi * z)^3 -
        3 * pi * sin(pi * x)^2 * sin(pi * y)^2 * sin(pi * z)^2 * cos(pi * t)^2 -
        7 * pi * sin(pi * x)^2 * sin(pi * y)^2 * cos(pi * t)^2 * cos(pi * z)^2 -
        7 *
        pi *
        sin(pi * x)^2 *
        sin(pi * y) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * y) -
        840 *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        3 *
        pi *
        sin(pi * x)^2 *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * y) *
        cos(pi * z)^2 -
        540 *
        sin(pi * x)^2 *
        sin(pi * z)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z) -
        420 * sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 +
        4 * pi * sin(pi * x)^2 * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * y)^2 -
        1080 *
        sin(pi * x)^2 *
        cos(pi * t)^3 *
        cos(pi * x) *
        cos(pi * y)^3 *
        cos(pi * z)^3 +
        16 *
        pi *
        sin(pi * x)^2 *
        cos(pi * t)^2 *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        420 * sin(pi * x)^2 * cos(pi * t)^2 * cos(pi * y)^2 * cos(pi * z)^2 +
        4 *
        pi *
        sin(pi * x) *
        sin(pi * y) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y) *
        cos(pi * z)^2 -
        42000 * sin(pi * x) * sin(pi * y) * cos(pi * t) * cos(pi * z) +
        7 *
        pi *
        sin(pi * x) *
        sin(pi * z)^2 *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 +
        3 *
        pi *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        840 *
        sin(pi * x) *
        cos(pi * t)^2 *
        cos(pi * x) *
        cos(pi * y)^2 *
        cos(pi * z)^2 +
        42000 * sin(pi * x) * cos(pi * t) * cos(pi * y) * cos(pi * z) -
        3 * pi * sin(pi * z)^2 * cos(pi * t)^2 * cos(pi * x)^2 * cos(pi * y)^2 -
        7 * pi * cos(pi * t)^2 * cos(pi * x)^2 * cos(pi * y)^2 * cos(pi * z)^2 +
        42000 * cos(pi * t) * cos(pi * x) * cos(pi * y) * cos(pi * z)
    ) / 300
