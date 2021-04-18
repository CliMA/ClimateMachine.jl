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
R(ξ¹; r_inner = r1, r_outer = r2) = (r_outer - r_inner) * (ξ¹ + 1)/2 + r_inner
Θ(ξ²; θ1 = π/2-φ, θ2 = π/2+φ) = (θ2 - θ1) * (ξ² + 1)/2 + θ1

# annulus to cartesian mappings
xc(r, θ) = r * sin(θ)
yc(r, θ) = r * cos(θ)

# Now construct the discretization
# Construct Gauss-Lobatto points
N1 = 4
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

# Construct convenient object for entries
jacobian[:,:,1,1] .= ∂x∂ξ¹
jacobian[:,:,1,2] .= ∂x∂ξ²
jacobian[:,:,2,1] .= ∂y∂ξ¹
jacobian[:,:,2,2] .= ∂y∂ξ²

detJ = [abs(det(jacobian[i,j,:,:])) for i in 1:length(ξ1vec), j in 1:length(ξ2vec)]
M = detJ .* ω1 .* ω2
exact_area = (r2^2 - r1^2) * φ
approx_area = sum(M)

wrongness = (exact_area - approx_area) / exact_area
println("The error in computing the area is ", wrongness)

##
fig = scatter(x_positions[:], y_positions[:])
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
