function Compute_Legendre!(num_fourier, num_spherical, sinθ, nθ)
    # Spectral Numerical Weather Prediction Models Appendix B
    # f(θ, λ) = ∑_{l=0} ∑_{m=-l}^{l} f_{lm} P_{lm}(sinθ)e^{imλ} (Y_{l,m} = P_{l,m} e^{i m λ} )
    # l=0,1...∞    and m = -l, -l+1, ... l-1, l
    # P_{0,0} = 1, such that 1/4π ∫∫YYdS = δ
    # P_{m,m} = sqrt((2m+1)/2m) cosθ P_{m-1m-1} 
    # P_{m+1,m} = sqrt(2m+3) sinθ P_{m m} 
    # sqrt((l^2-m^2)/(4l^2-1))P_{l,m} = x  P_{l-1, m} -  sqrt(((l-1)^2-m^2)/(4(l-1)^2 - 1))P_{l-2,m}
    # ε[m,l] = sqrt((l^2- m^2)/(4l^2 - 1))
    # (1-μ^2)d P_{m,l}/dμ = -nε[m,l+1]P_{m,l+1} + (l+1)ε_{m,l}P_{m,l-1}
    # Julia index starts with 1 qnm[m+1,l+1] = P[l,m]

    # dqnm = dP/dμ
    
    # 


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

