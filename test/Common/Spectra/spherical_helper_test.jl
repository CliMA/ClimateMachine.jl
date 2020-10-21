# additional helper functions for spherical harmonics spectra

"""
    TransSphericalToGrid!(mesh, snm )

Transforms a variable expressed in spherical harmonics (var_spherical[num_fourier+1, num_spherical+1]) onto a Gaussian grid (pfield[nλ, nθ])

[THIS IS USED FOR TESTING ONLY]

    With F_{m,n} = (-1)^m F_{-m,n}*
    P_{m,n} = (-1)^m P_{-m,n}

    F(λ, η) = ∑_{m= -N}^{N} ∑_{n=|m|}^{N} F_{m,n} P_{m,n}(η) e^{imλ}
    = ∑_{m= 0}^{N} ∑_{n=m}^{N} F_{m,n} P_{m,n} e^{imλ} + ∑_{m= 1}^{N} ∑_{n=m}^{N} F_{-m,n} P_{-m,n} e^{-imλ}

    Here η = sinθ, N = num_fourier, and denote
    ! extra coeffients in snm n > N are not used.

    ∑_{n=m}^{N} F_{m,n} P_{m,n}     = g_{m}(η) m = 1, ... N
    ∑_{n=m}^{N} F_{m,n} P_{m,n}/2.0 = g_{m}(η) m = 0

    We have

    F(λ, η) = ∑_{m= 0}^{N} g_{m}(η) e^{imλ} + ∑_{m= 0}^{N} g_{m}(η)* e^{-imλ}
    = 2real{ ∑_{m= 0}^{N} g_{m}(η) e^{imλ} }

    snm = F_{m,n}         # Complex{Float64} [num_fourier+1, num_spherical+1]
    qnm = P_{m,n,η}         # Float64[num_fourier+1, num_spherical+1, nθ]
    fourier_g = g_{m, η} # Complex{Float64} nλ×nθ with padded 0s fourier_g[num_fourier+2, :] == 0.0
    pfiled = F(λ, η)      # Float64[nλ, nθ]

    ! use all spherical harmonic modes


# Arguments
- mesh: struct with mesh information
- snm: spherical variable

# References
- Ehrendorfer, M., Spectral Numerical Weather Prediction Models, Appendix B, Society for Industrial and Applied Mathematics, 2011
"""
function trans_spherical_to_grid!(mesh, snm)
    num_fourier, num_spherical = mesh.num_fourier, mesh.num_spherical
    nλ, nθ, nd = mesh.nλ, mesh.nθ, mesh.nd

    qnm = mesh.qnm

    fourier_g = mesh.var_fourier .* 0.0
    fourier_s = mesh.var_fourier .* 0.0

    @assert(nθ % 2 == 0)
    nθ_half = div(nθ, 2)
    for m in 1:(num_fourier + 1)
        for n in m:num_spherical
            snm_t = transpose(snm[m, n, :, 1:nθ_half]) #snm[m,n, :] is complex number
            if (n - m) % 2 == 0
                fourier_s[m, 1:nθ_half, :] .+=
                    qnm[m, n, 1:nθ_half] .* sum(snm_t, dims = 1)   #even function part
            else
                fourier_s[m, (nθ_half + 1):nθ, :] .+=
                    qnm[m, n, 1:nθ_half] .* sum(snm_t, dims = 1)   #odd function part
            end
        end
    end
    fourier_g[:, 1:nθ_half, :] .=
        fourier_s[:, 1:nθ_half, :] .+ fourier_s[:, (nθ_half + 1):nθ, :]
    fourier_g[:, nθ:-1:(nθ_half + 1), :] .=
        fourier_s[:, 1:nθ_half, :] .- fourier_s[:, (nθ_half + 1):nθ, :] # this got ignored...

    fourier_g[1, :, :] ./= 2.0
    pfield = zeros(Float64, nλ, nθ, nd)
    for j in 1:nθ
        pfield[:, j, :] .= 2.0 * nλ * real.(ifft(fourier_g[:, j, :], 1)) #fourier for the first dimension
    end
    return pfield
end
