N = 2
A = randn(N,N)
x = randn(N)
b2 = zeros(N)
b = zeros(N)
for k in 1:N
    local_flux_3 = x[k]
    for n in 1:N
        b[n] += A[k, n] * local_flux_3
    end
    b2[k] += b[k]
end
A' * x - b  # correct
A' * x - b2 # incorrect
##

## Figuring out what the different things mean:
if dim == 1
    Np = (N + 1)
    Nfp = 1
    nface = 2
elseif dim == 2
    Np = (N + 1) * (N + 1)
    Nfp = (N + 1)
    nface = 4
elseif dim == 3
    Np = (N + 1) * (N + 1) * (N + 1)
    Nfp = (N + 1) * (N + 1)
    nface = 6
end
# no 19
for f in [1,2,3,4]
    for e in [1]
        for n in collect(1:(N+1)^(dims-1))
        normal_vector = SVector(
            grid.sgeo[grid.n1id, n, f, e],
            grid.sgeo[grid.n2id, n, f, e],
            grid.sgeo[grid.n3id, n, f, e],
        )
        # Get surface mass, volume mass inverse
        sM, vMI = grid.sgeo[grid.sMid, n, f, e], grid.sgeo[_vMI, n, f, e]
        id⁻, id⁺ = grid.vmap⁻[n, f, e], grid.vmap⁺[n, f, e]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1
        e⁻ = ((id⁻ - 1) ÷ Np) + 1
        # not sure I understand vid⁻ and vid⁺
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        local_flux  = normal_vector[1] * (flux[vid⁻, e⁻, 1] + flux[vid⁺, e⁺, 1])/2
        local_flux += normal_vector[2] * (flux[vid⁻, e⁻, 2] + flux[vid⁺, e⁺, 2])/2
        local_flux += normal_vector[3] * (flux[vid⁻, e⁻, 3] + flux[vid⁺, e⁺, 3])/2
        println("----------------------")
        println("For face = ", f, ", element e = ", e, ", gl point = ", n)
        println("The (x,y,z) coordinates are ")
        println((x[id⁻], y[id⁻], z[id⁻]))
        println("The normal vector is ")
        println(normal_vector)
        println("vmap⁻=", id⁻)
        println("vid⁻=", vid⁻)
        println("local flux=", local_flux)
        println("----------------------")
        x[vid⁻, e]
        normal_vector
        end
    end
end

##
v_flux_divergence[grid.vmap⁻]
s_flux_divergence[grid.vmap⁻]
scatter(x[interior], y[interior])
##
Φ = reshape(flux, (N+1, N+1, size(flux)[2], size(flux)[3]))
divΦ = reshape(v_flux_divergence, (N+1, N+1, size(x)[2]))
∮divΦ = reshape(s_flux_divergence, (N+1, N+1, size(x)[2]))
## grid.sgeo[grid.sMid, :, :, :]
Φ = reshape(flux, (N+1, N+1, N+1, size(flux)[2], size(flux)[3]))
divΦ = reshape(v_flux_divergence, (N+1, N+1, N+1, size(x)[2]))
∮divΦ = reshape(s_flux_divergence, (N+1, N+1, N+1, size(x)[2]))

##
Mr = reshape(M, (N+1,N+1,N+1))
Mir = reshape(Mi, (N+1,N+1,N+1))
i = 1
j = 1 
# in the single element case
for i in 1:N+1
    for j in 1:N+1
        println("------------------")
        println("for i = ", i , " and j = ", j)
        println("The local flux 3 should be ")
        println(Mr[i,j,:] .* Φ[i,j,:,1,3])
        println("The value should be")
        println(Mir[i,j,:] .* (grid.D[1]' * ( Mr[i,j,:] .* Φ[i,j,:,1,3] )))
        println("The volume term is ")
        println(divΦ[i,j,:])
        println("The surface term is ")
        println(∮divΦ[i,j,:])
        println("------------------")
    end
end
## more details
for i in 1:N+1
    for j in 1:N+1
        for k in 1:N+1
            println("------------------")
            println("for i = ", i , " and j = ", j, " and k = ", k)
            println("The local flux 3 should be ")
            println(Mr[i,j,k] .* Φ[i,j,k,1,3])
            println("The value should be")
            tmp = Mir[i,j,k] .* (grid.D[1]' * ( Mr[i,j,:] .* Φ[i,j,:,1,3] ))
            println(tmp[k])
            println("The volume term is ")
            println(divΦ[i,j,k])
            println("The surface term is ")
            println(∮divΦ[i,j,k])
            println("------------------")
        end
    end
end

##
i = 1 
j = 1
tmp = zeros(N+1)
s_D = copy(grid.D[1])
flux = zeros(N+1)
for k in 1:N+1
    for n in 1:N+1
        tmp[n] += Mir[i,j,k] * (s_D[k,n] * ( Mr[i,j,n] * Φ[i,j,k,1,3] ))
    end
    flux[k] += tmp[k]
end
println("correct computation")
println(tmp)
println("current computation")
println(flux)
println("should be")
println(∮divΦ[i,j,:])
#=
println("computation from kernel")
println(divΦ[i,j,:])
=#

##
## Figuring out what the different things mean:
#=
if dim == 1
    Np = (N + 1)
    Nfp = 1
    nface = 2
elseif dim == 2
    Np = (N + 1) * (N + 1)
    Nfp = (N + 1)
    nface = 4
elseif dim == 3
    Np = (N + 1) * (N + 1) * (N + 1)
    Nfp = (N + 1) * (N + 1)
    nface = 6
end
# no 19
for f in [1,2,3,4]
    for e in [1]
        for n in collect(1:(N+1)^(dims-1))
        normal_vector = SVector(
            grid.sgeo[grid.n1id, n, f, e],
            grid.sgeo[grid.n2id, n, f, e],
            grid.sgeo[grid.n3id, n, f, e],
        )
        # Get surface mass, volume mass inverse
        sM, vMI = grid.sgeo[grid.sMid, n, f, e], grid.sgeo[_vMI, n, f, e]
        id⁻, id⁺ = grid.vmap⁻[n, f, e], grid.vmap⁺[n, f, e]
        e⁺ = ((id⁺ - 1) ÷ Np) + 1
        e⁻ = ((id⁻ - 1) ÷ Np) + 1
        # not sure I understand vid⁻ and vid⁺
        vid⁻, vid⁺ = ((id⁻ - 1) % Np) + 1, ((id⁺ - 1) % Np) + 1
        local_flux  = normal_vector[1] * (flux[vid⁻, e⁻, 1] + flux[vid⁺, e⁺, 1])/2
        local_flux += normal_vector[2] * (flux[vid⁻, e⁻, 2] + flux[vid⁺, e⁺, 2])/2
        local_flux += normal_vector[3] * (flux[vid⁻, e⁻, 3] + flux[vid⁺, e⁺, 3])/2
        println("----------------------")
        println("For face = ", f, ", element e = ", e, ", gl point = ", n)
        println("The (x,y,z) coordinates are ")
        println((x[id⁻], y[id⁻], z[id⁻]))
        println("The normal vector is ")
        println(normal_vector)
        println("vmap⁻=", id⁻)
        println("vid⁻=", vid⁻)
        println("local flux=", local_flux)
        println("----------------------")
        x[vid⁻, e]
        normal_vector
        end
    end
end
=#
#=
##
v_flux_divergence[grid.vmap⁻]
s_flux_divergence[grid.vmap⁻]
scatter(x[interior], y[interior])
##
Φ = reshape(flux, (N+1, N+1, size(flux)[2], size(flux)[3]))
divΦ = reshape(v_flux_divergence, (N+1, N+1, size(x)[2]))
∮divΦ = reshape(s_flux_divergence, (N+1, N+1, size(x)[2]))
## grid.sgeo[grid.sMid, :, :, :]
Φ = reshape(flux, (N+1, N+1, N+1, size(flux)[2], size(flux)[3]))
divΦ = reshape(v_flux_divergence, (N+1, N+1, N+1, size(x)[2]))
∮divΦ = reshape(s_flux_divergence, (N+1, N+1, N+1, size(x)[2]))

##
Mr = reshape(M, (N+1,N+1,N+1))
Mir = reshape(Mi, (N+1,N+1,N+1))
i = 1
j = 1 
# in the single element case
for i in 1:N+1
    for j in 1:N+1
        println("------------------")
        println("for i = ", i , " and j = ", j)
        println("The local flux 3 should be ")
        println(Mr[i,j,:] .* Φ[i,j,:,1,3])
        println("The value should be")
        println(Mir[i,j,:] .* (grid.D[1]' * ( Mr[i,j,:] .* Φ[i,j,:,1,3] )))
        println("The volume term is ")
        println(divΦ[i,j,:])
        println("The surface term is ")
        println(∮divΦ[i,j,:])
        println("------------------")
    end
end
## more details
for i in 1:N+1
    for j in 1:N+1
        for k in 1:N+1
            println("------------------")
            println("for i = ", i , " and j = ", j, " and k = ", k)
            println("The local flux 3 should be ")
            println(Mr[i,j,k] .* Φ[i,j,k,1,3])
            println("The value should be")
            tmp = Mir[i,j,k] .* (grid.D[1]' * ( Mr[i,j,:] .* Φ[i,j,:,1,3] ))
            println(tmp[k])
            println("The volume term is ")
            println(divΦ[i,j,k])
            println("The surface term is ")
            println(∮divΦ[i,j,k])
            println("------------------")
        end
    end
end

##
# this is another way to check the answer
Φ = reshape(flux, (N+1, N+1, N+1, size(flux)[2], size(flux)[3]))
∂ˣΦ = zeros(N+1, N+1, N+1, size(flux)[2])
metricterm = grid.vgeo[:, 1, :, :]
for j in 1:(N+1)
    for k in 1:(N+1)
        for e in 1:size(flux)[2]
            ∂ˣΦ[:, j, k, e] = grid.D[1] * Φ[:, j, k, e, 1] * metricterm[1]
        end
    end
end
check = reshape(∂ˣΦ, ((N+1)^dim, size(flux)[2]))
check - analytic_flux_divergence
##
# The check
v_flux_divergence - s_flux_divergence
=#