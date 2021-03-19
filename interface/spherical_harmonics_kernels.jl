export Compute_Legendre!, Compute_Gaussian!


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
    

# function Compute_Gaussian!(n)
#     # Pn(x) is an odd function
#     # solve half of the n roots and weightes of Pn(x) # n = 2n_half
#     # P_{-1}(x) = 0
#     # P_0(x) = 1
#     # P_1(x) = x
#     # nP_n(x) = (2n-1)xP_{n-1}(x) - (n-1)P_{n-2}(x)
#     # P'_n(x) = n/(x^2-1)(xP_{n}(x) - P_{n-1}(x))
#     # x -= P_n(x)/P'_{n}()
#     # Initial guess xi^{0} = cos(π(i-0.25)/(n+0.5)) 
#     # wi = 2/(1-xi^2)/P_n'(xi)^2 
    
#     itermax = 10000
#     tol = 1.0e-15

 
#     sinθ = zeros(Float64, n)
#     wts = zeros(Float64, n)

#     n_half = Int64(n/2)
#     for i=1:n_half
#         dp = 0.0
#         z = cos(pi*(i - 0.25)/(n + 0.5))
#         for iter=1:itermax
#             p2 = 0.0
#             p1 = 1.0
            
#             for j=1:n
#                 p3 = p2 # Pj-2
#                 p2 = p1 # Pj-1
#                 p1 = ((2.0*j - 1.0)*z*p2 - (j - 1.0)*p3)/j  #Pj
#             end
#             # P'_n
#             dp = n*(z*p1 - p2)/(z*z - 1.0)
#             z1 = z
#             z  = z1 - p1/dp
#             if(abs(z - z1) <= tol)
#                 break;
#             end
#             if iter == itermax
#                 @error("Compute_Gaussian! does not converge!")
#             end
#         end
        
#         sinθ[i], sinθ[n-i+1],  = -z, z
#         wts[i] = wts[n-i+1]  = 2.0/((1.0 - z*z)*dp*dp)
#     end

#     return sinθ, wts
# end

# function test()
    
#     num_fourier, nθ = 21, 32
#     num_spherical = num_fourier+1
#     nλ = nθ

#     sinθ, wts = Compute_Gaussian!(nθ)
#     #compare with https://pomax.github.io/bezierinfo/legendre-gauss.html


#     qnm, dqnm = Compute_Legendre!(num_fourier, num_spherical, sinθ, nθ)

#     q44 = sqrt(35)/4*(1 .- sinθ.^2).^(3/2) 
#     q34 = sqrt(105/8)*(sinθ .- sinθ.^3)
#     q13 = sqrt(5)/2*(3sinθ.^2 .- 1)
#     q14 = sqrt(7)/2*(5sinθ.^3 .- 3sinθ)
#     q24 = sqrt(21)/4*(5sinθ.^2 .- 1).*sqrt.(1 .- sinθ.^2) 


#     dq44 = sqrt(35)/4*(1 .- sinθ.^2).^(1/2) *(3/2).*(-2sinθ)  
#     dq34 = sqrt(105/8)*(1 .- 3*sinθ.^2)
#     dq13 = sqrt(5)/2*(6sinθ)
#     dq14 = sqrt(7)/2*(15sinθ.^2 .- 3)
#     dq24 = sqrt(21)/4*(10*sinθ).*sqrt.(1 .- sinθ.^2) + sqrt(21)/4*(5sinθ.^2 .- 1)./sqrt.(1 .- sinθ.^2)*0.5 .* (-2sinθ)

#     #compare with exact form
#     @show norm(qnm[4,4,:] - q44)
#     @show norm(qnm[3,4,:] - q34)
#     @show norm(qnm[1,3,:] - q13) 
#     @show norm(qnm[1,4,:] - q14)
#     @show norm(qnm[2,4,:] - q24) 

#     @show norm(dqnm[4,4,:] - dq44)
#     @show norm(dqnm[3,4,:] - dq34)
#     @show norm(dqnm[1,3,:] - dq13) 
#     @show norm(dqnm[1,4,:] - dq14)
#     @show norm(dqnm[2,4,:] - dq24) 

#     #check 1/2∫_{-1}^{1}P_{n,m} P_{l,m} = δ_{n,l}

#     D = zeros()
    
#     for m=1:num_fourier+1
#         for l=m:num_spherical+1
#             for n=m:num_spherical+1
#                 @assert( (0.5*sum(qnm[m,n,:].*qnm[m,l,:].*wts) - Float64(n==l)) < 1.0e-10)
#             end
#         end
#     end
# end


# get_c(l,r) = l^2*(l+1)^2/r^4
get_c(l,r) = l*(l+1)/r^2

calc_Plm(φ, l, m) = Compute_Legendre!(m, l, sin(φ), length(φ))

function calc_Ylm(φ, λ, l, m)
    qnm = calc_Plm(φ, l, m)[1][m+1,l+1,:][1]
    return real(qnm .* exp(im*m*λ))
end