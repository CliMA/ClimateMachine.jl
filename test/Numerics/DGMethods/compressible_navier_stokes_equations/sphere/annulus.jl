using GLMakie
using GaussQuadrature

# First construct the mapping for a spherical sector

# parameters:
nn = 12  # number of sectors
φ = π/nn # half angle of sector
r1 = 0.8 # inner radius
r2 = 1.0 # outer radius

# mappings
# annulus mappings
R(ξ¹; r1 = r1, r2 = r2) = (r2 - r1) * (ξ¹ + 1)/2 + r1
Θ(ξ²; θ1 = π/2-φ, θ2 = π/2+φ) = (θ2 - θ1) * (ξ² + 1)/2 + θ1

# annulus to cartesian mappings
xc(r, θ) = r * cos(θ)
yc(r, θ) = r * sin(θ)

# Now construct the discretization
# Construct Gauss-Lobatto points
N1 = 1
ξ1vec, ω1 = GaussQuadrature.legendre(N1 + 1, GaussQuadrature.both)
ω1 = reshape(ω1, (N1+1,1))
N2 = 2
ξ2vec, ω2 = GaussQuadrature.legendre(N2 + 1, GaussQuadrature.both)
ω2 = reshape(ω2, (1, N2+1))

# Construct Differentiation Matrices
vec = ξ1vec
ξp = [vec[i]^(j-1) for i in 1:length(vec), j in 1:length(vec)]
dξp = [(j-1) * vec[i]^(j-2) for i in 1:length(vec), j in 1:length(vec)]
∂ξ¹ = dξp / ξp

vec = ξ2vec
ξp = [vec[i]^(j-1) for i in 1:length(vec), j in 1:length(vec)]
dξp = [(j-1) * vec[i]^(j-2) for i in 1:length(vec), j in 1:length(vec)]
∂ξ² = dξp / ξp

# Calculate the grid points in polar coordinates
rvec = R.(ξ1vec)
θvec = Θ.(ξ2vec)
# convert to cartesian
x_positions = [xc(r, θ) for r in rvec, θ in θvec]
y_positions = [yc(r, θ) for r in rvec, θ in θvec]

# Now do the discrete Calculus
∂x∂ξ¹ = copy(x_positions)
∂y∂ξ¹ = copy(x_positions)
for j in 1:size(∂x∂ξ¹)[2]
    ∂x∂ξ¹[:, j] = ∂ξ¹ * x_positions[:, j]
    ∂y∂ξ¹[:, j] = ∂ξ¹ * y_positions[:, j]
end

∂x∂ξ² = copy(x_positions)
∂y∂ξ² = copy(x_positions)
for i in 1:size(∂x∂ξ²)[1]
    ∂x∂ξ²[i,:] = ∂ξ² * x_positions[i,:]
    ∂y∂ξ²[i,:] = ∂ξ² * y_positions[i,:]
end
jacobian = zeros(size(x_positions)..., (2,2)...)

# Construct convenient object for entries, The columns are the covariant vectors
jacobian[:,:,1,1] .= ∂x∂ξ¹
jacobian[:,:,1,2] .= ∂x∂ξ²
jacobian[:,:,2,1] .= ∂y∂ξ¹
jacobian[:,:,2,2] .= ∂y∂ξ²

detJ = [det(jacobian[i,j,:,:]) for i in 1:length(ξ1vec), j in 1:length(ξ2vec)]
M = detJ .* ω1 .* ω2
exact_area = (r2^2 - r1^2) * φ
approx_area = sum(M)

wrongness = (exact_area - approx_area) / exact_area
println("The error in computing the area is ", wrongness)

# Construct contravariant basis numerically, these are the face normals
ijacobian = copy(jacobian) # the rows are the contravariant vectors
for i in 1:length(ξ1vec), j in 1:length(ξ2vec)
    tmp = inv(jacobian[i,j,:,:])
    ijacobian[i,j,:,:] .= tmp
end
ijacobian[1,1,:,:]
x_positions[1,1], y_positions[1,1]

# face 2 is the linear side
approx_vec = ijacobian[1,1,2,:] ./ norm(ijacobian[1,1,2,:])
exact_vec = [-y_positions[1,1], x_positions[1,1]] ./ norm([y_positions[1,1], -x_positions[1,1]])
println("angle face error ", norm(approx_vec - exact_vec) / norm(exact_vec))
# face 1 is the curvy side
approx_vec = ijacobian[1,1,1,:] ./ norm(ijacobian[1,1,1,:])
exact_vec = [x_positions[1,1], y_positions[1,1]] ./ norm([x_positions[1,1], y_positions[1,1]])
println("radial face error ", norm(approx_vec - exact_vec) / norm(exact_vec))
##
fig = scatter(x_positions[:], y_positions[:], color = :red)
xlims!(fig.axis, (-1.1,1.1))
ylims!(fig.axis, (-1.1,1.1))
display(fig)


#=
for i in 1:(nn-1)
ξ1vec = -1:0.5:1
ξ2vec = -1:0.5:1
rvec = R.(ξ1vec)
θvec = Θ.(ξ2vec, θ1 = π/2 - φ + 2*φ * i, θ2 = π/2 + φ + 2*φ * i)

x_positions = [xc(r, θ) for r in rvec, θ in θvec]
y_positions = [yc(r, θ) for r in rvec, θ in θvec]
scatter!(fig.axis, x_positions[:], y_positions[:])
end
xlims!(fig.axis, (-1.1,1.1))
ylims!(fig.axis, (-1.1,1.1))
display(fig)
=#
