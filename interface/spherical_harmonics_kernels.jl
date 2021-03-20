export Compute_Legendre!, Compute_Gaussian!


function Compute_Legendre!(num_fourier, num_spherical, sinθ, nθ)
    qnm = zeros(Float64, num_fourier+1, num_spherical+2, nθ)
    dqnm = zeros(Float64, num_fourier+1, num_spherical+1, nθ)

    cosθ = sqrt.(1 .- sinθ.^2)
    ε = zeros(Float64, num_fourier+1, num_spherical+2)

    qnm[1, 1, :] .= 1.0
    for m = 1:num_fourier
        qnm[m+1, m+1, :] = sqrt((2m+1)/(2m)) .* cosθ .* qnm[m, m,:]
    end
    
    for m = 1:num_fourier+1
        qnm[m, m+1, :] = sqrt(2*m+1) * sinθ .* qnm[m,m, :] 
    end
    
    for m = 0:num_fourier
        for l = m:num_spherical+1
            #ε[m,l] = sqrt(((l-1)^2 - (m-1)^2) ./ (2*(l-1) + 1))
            ε[m+1,l+1] = sqrt((l^2 - m^2) ./ (4*l^2 - 1))
        end
    end

    for m = 0:num_fourier
        for l = m+2:num_spherical+1
            #m=0, l=2
            #qnm[m+1,l+1,:] = sqrt(2*l-1) /ε[m+1,l+1] * (sinθ .* qnm[m+1,l,:] -  ε[m+1,l]/sqrt(2*l-3)*qnm[m+1,l-1,:])
            qnm[m+1,l+1,:] = (sinθ .* qnm[m+1,l,:] -  ε[m+1,l]*qnm[m+1,l-1,:])/ε[m+1,l+1]
        end
    end

    for m = 0:num_fourier
        for l = m:num_spherical
            if l == m
                dqnm[m+1,l+1,:] = (-l*ε[m+1, l+2]*qnm[m+1,l+2,:])./(cosθ.^2)
            else
                dqnm[m+1,l+1,:] = (-l*ε[m+1, l+2]*qnm[m+1,l+2,:] + (l+1)*ε[m+1,l+1]*qnm[m+1,l,:])./(cosθ.^2)
            end

        end
    end

    return qnm[:,1:num_spherical+1,:], dqnm
    # d P_{m,l}/dμ = -nε[m,l+1]P_{m,l+1} + (l+1)ε_{m,l}P_{m,l-1}     
end

get_c(l,r) = -l*(l+1)/r^2

calc_Plm(φ, l, m) = Compute_Legendre!(m, l, sin(φ), length(φ))

function calc_Ylm(φ, λ, l, m)
    qnm = calc_Plm(φ, l, m)[1][m+1,l+1,:][1]
    return real(qnm .* exp(im*m*λ))
end

