# helper functions for spherical harmonics spectra

"""
    SpectralSphericalMesh

Struct with mesh information.
"""
mutable struct SpectralSphericalMesh
    # grid info
    num_fourier::Int64
    num_spherical::Int64
    nλ::Int64
    nθ::Int64
    nd::Int64
    Δλ::Float64
    qwg::Array{Float64, 3}
    qnm::Array{Float64, 3}   # n,m coordinates
    wave_numbers::Array{Int64, 2}

    # variables
    var_grid::Array{Float64, 3}
    var_fourier::Array{ComplexF64, 3}
    var_spherical::Array{ComplexF64, 4}
    var_spectrum::Array{Float64, 3}

end
function SpectralSphericalMesh(nθ::Int64, nd::Int64)
    nλ = 2nθ
    Δλ = 2π / nλ

    num_fourier = Int64((2 * nθ - 1) / 3) # number of truncated zonal wavenumbers (m): minimum truncation given nθ - e.g.: nlat = 32 -> T21 (can change manually for more a severe truncation)
    num_spherical = Int64(num_fourier + 1) # number of total wavenumbers (n)

    radius = Float64(6371000)
    wave_numbers = compute_wave_numbers(num_fourier, num_spherical)

    qwg = zeros(Float64, num_fourier + 1, num_spherical + 1, nθ)
    qnm = zeros(Float64, num_fourier + 1, num_spherical + 2, nθ)

    var_fourier = zeros(Complex{Float64}, nλ, nθ, nd)
    var_grid = zeros(Float64, nλ, nθ, nd)
    nθ_half = div(nθ, 2)
    var_spherical =
        zeros(Complex{Float64}, num_fourier + 1, num_spherical + 1, nd, nθ_half)
    var_spectrum = zeros(Float64, num_fourier + 1, num_spherical + 1, nd)

    SpectralSphericalMesh(
        num_fourier,
        num_spherical,
        nλ,
        nθ,
        nd,
        Δλ,
        qwg,
        qnm,
        wave_numbers,
        var_grid,
        var_fourier,
        var_spherical,
        var_spectrum,
    )
end

"""
    compute_legendre!(num_fourier, num_spherical, sinθ, nθ)

Normalized associated Legendre polynomials, P_{m,l} = qnm

# Arguments:
- num_fourier
- num_spherical
- sinθ
- nθ

# References:
- Ehrendorfer, M. (2011) Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics
- Winch, D. (2007) Spherical harmonics, in Encyclopedia of Geomagnetism and Paleomagnetism, Eds Gubbins D. and Herrero-Bervera, E., Springer

# Details (using notation and Eq. references from Ehrendorfer, 2011):

    l=0,1...∞    and m = -l, -l+1, ... l-1, l

    P_{0,0} = 1, such that 1/4π ∫∫YYdS = δ (where Y = spherical harmonics, S = domain surface area)
    P_{m,m} = sqrt((2m+1)/2m) cosθ P_{m-1m-1}
    P_{m+1,m} = sqrt(2m+3) sinθ P_{m m}
    sqrt((l^2-m^2)/(4l^2-1))P_{l,m} = P_{l-1, m} -  sqrt(((l-1)^2-m^2)/(4(l-1)^2 - 1))P_{l-2,m}

    THe normalization assures that 1/2 ∫_{-1}^1 P_{l,m}(sinθ) P_{n,m}(sinθ) dsinθ = δ_{n,l}

    Julia index starts with 1, so qnm[m+1,l+1] = P_l^m
"""
function compute_legendre!(num_fourier, num_spherical, sinθ, nθ)
    qnm = zeros(Float64, num_fourier + 1, num_spherical + 2, nθ)

    cosθ = sqrt.(1 .- sinθ .^ 2)
    ε = zeros(Float64, num_fourier + 1, num_spherical + 2)

    qnm[1, 1, :] .= 1
    for m in 1:num_fourier
        qnm[m + 1, m + 1, :] = -sqrt((2m + 1) / (2m)) .* cosθ .* qnm[m, m, :] # Eq. B.20
        qnm[m, m + 1, :] = sqrt(2m + 1) * sinθ .* qnm[m, m, :] # Eq. B.22
    end
    qnm[num_fourier + 1, num_fourier + 2, :] =
        sqrt(2 * (num_fourier + 2)) * sinθ .*
        qnm[num_fourier + 1, num_fourier + 1, :]

    for m in 0:num_fourier
        for l in (m + 2):(num_spherical + 1)
            ε1 = sqrt(((l - 1)^2 - m^2) ./ (4 * (l - 1)^2 - 1))
            ε2 = sqrt((l^2 - m^2) ./ (4 * l^2 - 1))
            qnm[m + 1, l + 1, :] =
                (sinθ .* qnm[m + 1, l, :] - ε1 * qnm[m + 1, l - 1, :]) / ε2 # Eq. B.18
        end
    end

    return qnm[:, 1:(num_spherical + 1), :]
end

"""
    compute_gaussian!(n)

Compute sin(latitude) and the weight factors for Gaussian integration

# Arguments
- n: number of latitudes

# References
- Ehrendorfer, M., Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics, 2011

# Details (following notation from Ehrendorfer, 2011):
    Pn(x) is an odd function
    solve half of the n roots and weightes of Pn(x) # n = 2n_half
    P_{-1}(x) = 0
    P_0(x) = 1
    P_1(x) = x
    nP_n(x) = (2n-1)xP_{n-1}(x) - (n-1)P_{n-2}(x)
    P'_n(x) = n/(x^2-1)(xP_{n}(x) - P_{n-1}(x))
    x -= P_n(x)/P'_{n}()
    Initial guess xi^{0} = cos(π(i-0.25)/(n+0.5))
    wi = 2/(1-xi^2)/P_n'(xi)^2
"""
function compute_gaussian!(n)
    itermax = 10000
    tol = 1.0e-15

    sinθ = zeros(Float64, n)
    wts = zeros(Float64, n)

    n_half = Int64(n / 2)
    for i in 1:n_half
        dp = 0.0
        z = cos(pi * (i - 0.25) / (n + 0.5))
        for iter in 1:itermax
            p2 = 0.0
            p1 = 1.0

            for j in 1:n
                p3 = p2 # Pj-2
                p2 = p1 # Pj-1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j  #Pj
            end
            # P'_n
            dp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / dp
            if (abs(z - z1) <= tol)
                break
            end
            if iter == itermax
                @error("Compute_Gaussian! does not converge!")
            end
        end

        sinθ[i], sinθ[n - i + 1], = -z, z
        wts[i] = wts[n - i + 1] = 2.0 / ((1.0 - z * z) * dp * dp)
    end

    return sinθ, wts
end

"""
    trans_grid_to_spherical!(mesh::SpectralSphericalMesh, pfield::Array{Float64,2})

Transforms a variable on a Gaussian grid (pfield[nλ, nθ]) into the spherical harmonics domain (var_spherical2d[num_fourier+1, num_spherical+1])

Here λ = longitude, θ = latitude, η = sinθ, m = zonal wavenumber, n = total wavenumber:
var_spherical2d = F_{m,n}    # Output variable in spectral space (Complex{Float64}[num_fourier+1, num_spherical+1])
qwg = P_{m,n}(η)w(η)         # Weighted Legendre polynomials (Float64[num_fourier+1, num_spherical+1, nθ])
var_fourier2d = g_{m, θ}     # Untruncated Fourier transformation (Complex{Float64} [nλ, nθ])
pfield = F(λ, η)             # Input variable on Gaussian grid Float64[nλ, nθ]

# Arguments
- mesh: struct with mesh information
- pfield: variable on Gaussian grid to be transformed

# References
- Ehrendorfer, M., Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics, 2011
- A. Wiin-Nielsen (1967) On the annual variation and spectral distribution of atmospheric energy, Tellus, 19:4, 540-559, DOI: 10.3402/tellusa.v19i4.9822
"""
function trans_grid_to_spherical!(
    mesh::SpectralSphericalMesh,
    pfield::Array{Float64, 2},
)

    num_fourier, num_spherical = mesh.num_fourier, mesh.num_spherical
    var_fourier2d, var_spherical2d =
        mesh.var_fourier[:, :, 1] * 0.0, mesh.var_spherical[:, :, 1, :] * 0.0
    nλ, nθ, nd = mesh.nλ, mesh.nθ, mesh.nd

    # Retrieve weighted Legendre polynomials
    qwg = mesh.qwg # qwg[m,n,nθ]

    # Fourier transformation
    for j in 1:nθ
        var_fourier2d[:, j] = fft(pfield[:, j], 1) / nλ
    end

    # Complete spherical harmonic transformation
    @assert(nθ % 2 == 0)
    nθ_half = div(nθ, 2)
    for m in 1:(num_fourier + 1)
        for n in m:num_spherical
            var_fourier2d_t = transpose(var_fourier2d[m, :])  # truncates var_fourier(nlon, nhlat) to (nfourier,nlat)
            if (n - m) % 2 == 0
                var_spherical2d[m, n, :] .=
                    (
                        var_fourier2d_t[1:nθ_half] .+
                        var_fourier2d_t[nθ:-1:(nθ_half + 1)]
                    ) .* qwg[m, n, 1:nθ_half] ./ 2.0
            else
                var_spherical2d[m, n, :] .=
                    (
                        var_fourier2d_t[1:nθ_half] .-
                        var_fourier2d_t[nθ:-1:(nθ_half + 1)]
                    ) .* qwg[m, n, 1:nθ_half] ./ 2.0
            end
        end
    end

    return var_spherical2d
end

function compute_wave_numbers(num_fourier::Int64, num_spherical::Int64)
    """
    See wave_numers[i,j] saves the wave number of this basis
    """
    wave_numbers = zeros(Int64, num_fourier + 1, num_spherical + 1)

    for m in 0:num_fourier
        for n in m:num_spherical
            wave_numbers[m + 1, n + 1] = n
        end
    end

    return wave_numbers

end
