"""
    power_spectrum_3d(::AtmosLESConfigType, u, v, w, L, dim, nor)

Calculates the Powerspectrum from the 3D velocity fields `u`, `v`, `w`. Inputs
need to be equi-spaced and the domain is assumed to be the same size and
have the same number of points in all directions.

# Arguments
- L: size domain
- dim: number of points
- nor: normalization factor
"""
function power_spectrum_3d(::AtmosLESConfigType, u, v, w, L, dim, nor)
    u_fft = fft(u)
    v_fft = fft(v)
    w_fft = fft(w)

    mu = Array((abs.(u_fft) / size(u, 1)^3))
    mv = Array((abs.(v_fft) / size(v, 1)^3))
    mw = Array((abs.(w_fft) / size(w, 1)^3))
    if mod(dim, 2) == 0
        rx = range(0, stop = dim - 1, step = 1) .- dim / 2 .+ 1
        ry = range(0, stop = dim - 1, step = 1) .- dim / 2 .+ 1
        rz = range(0, stop = dim - 1, step = 1) .- dim / 2 .+ 1
        R_x = circshift(rx', (1, dim / 2 + 1))
        R_y = circshift(ry', (1, dim / 2 + 1))
        R_z = circshift(rz', (1, dim / 2 + 1))
        k_nyq = dim / 2
    else
        rx = range(0, stop = dim - 1, step = 1) .- (dim - 1) / 2
        ry = range(0, stop = dim - 1, step = 1) .- (dim - 1) / 2
        rz = range(0, stop = dim - 1, step = 1) .- (dim - 1) / 2
        R_x = circshift(rx', (1, (dim + 1) / 2))
        R_y = circshift(ry', (1, (dim + 1) / 2))
        R_z = circshift(rz', (1, (dim + 1) / 2))
        k_nyq = (dim - 1) / 2
    end
    r = zeros(size(rx, 1), size(ry, 1), size(rz, 1))
    for i in 1:size(rx, 1), j in 1:size(ry, 1), l in 1:size(rz, 1)
        r[i, j, l] = sqrt(R_x[i]^2 + R_y[j]^2 + R_z[l]^2)
    end
    dx = 2 * pi / L
    k = range(1, stop = k_nyq, step = 1) .* dx
    endk = size(k, 1)
    contribution = zeros(size(k, 1))
    spectrum = zeros(size(k, 1))
    for N in 2:Int64(k_nyq - 1)
        for i in 1:size(rx, 1), j in 1:size(ry, 1), l in 1:size(rz, 1)
            if (r[i, j, l] * dx <= (k'[N + 1] + k'[N]) / 2) &&
               (r[i, j, l] * dx > (k'[N] + k'[N - 1]) / 2)
                spectrum[N] =
                    spectrum[N] + mu[i, j, l]^2 + mv[i, j, l]^2 + mw[i, j, l]^2
                contribution[N] = contribution[N] + 1
            end
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1), l in 1:size(rz, 1)
        if (r[i, j, l] * dx <= (k'[2] + k'[1]) / 2)
            spectrum[1] =
                spectrum[1] + mu[i, j, l]^2 + mv[i, j, l]^2 + mw[i, j, l]^2
            contribution[1] = contribution[1] + 1
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1), l in 1:size(rz, 1)
        if (r[i, j, l] * dx <= k'[endk]) &&
           (r[i, j, l] * dx > (k'[endk] + k'[endk - 1]) / 2)
            spectrum[endk] =
                spectrum[endk] + mu[i, j, l]^2 + mv[i, j, l]^2 + mw[i, j, l]^2
            contribution[endk] = contribution[endk] + 1
        end
    end
    spectrum = spectrum .* 2 .* pi .* k .^ 2 ./ (contribution .* dx .^ 3) ./ nor

    return spectrum, k
end
