using ClimateMachine
ClimateMachine.init()

include("../boilerplate.jl") # most functions in vector fields, include("test/Numerics/DGMethods/compressible_navier_stokes_equations/boilerplate.jl") 

# Ω = Interval(0,1) × Interval(-1,1) × Interval(exp(1), π) 
Ω =  AtmosDomain(radius = 6e6, height = 3e3)

elements = (vertical = 10, horizontal = 10) # horizontal here means horizontal^2 * 6
polynomialorder = (vertical = 1, horizontal = 5)
grid = DiscontinuousSpectralElementGrid(
    Ω,
    elements = elements,
    polynomialorder = polynomialorder
)
h = min_node_distance(grid, HorizontalDirection())
x,y,z  = coordinates(grid)
# visualize(grid) # can only visualize grid if vertical = horizontal polynomial order, otherwise can't change face

## volume check
M = grid.vgeo[:, grid.Mid, :]
J = constructdeterminant(grid)
exact = 4π/3 * ( (Ω.height + Ω.radius)^3  - (Ω.radius)^3 )
volumeerror = ( sum(M) - exact ) / exact

ijk = 1 # ijk max is size(x)[1]
e = 1   # e max is size(x)[2]
r⃗ = [x,y,z]
position = getindex.(r⃗, Ref(ijk), Ref(e))

fullJ  = getjacobian(grid, ijk, e)
fulliJ = inv(getjacobian(grid, ijk, e))

gⁱᵏ = fullJ * fullJ'
display(gⁱᵏ / gⁱᵏ[1,1])

##
porders = (polynomialorder.horizontal, polynomialorder.horizontal, polynomialorder.vertical)
glnum = porders .+ 1
n_e = size(x)[2]
xreshape = reshape(x, (glnum..., n_e))
yreshape = reshape(y, (glnum..., n_e))
zreshape = reshape(z, (glnum..., n_e))

∂¹ =  grid.D[1]
∂² =  grid.D[2]
∂³ =  grid.D[3]

# derivative with respect to ξ1
x1ξ1 = copy(xreshape)
x2ξ1 = copy(xreshape)
x3ξ1 = copy(xreshape)
for j in 1:glnum[2], k in 1:glnum[3], e in 1:n_e
    x1ξ1[:,j,k,e] = ∂¹ * xreshape[:,j,k,e] 
    x2ξ1[:,j,k,e] = ∂¹ * yreshape[:,j,k,e] 
    x3ξ1[:,j,k,e] = ∂¹ * zreshape[:,j,k,e] 
end

# derivative with respect to ξ2
x1ξ2 = copy(xreshape)
x2ξ2 = copy(xreshape)
x3ξ2 = copy(xreshape)
for i in 1:glnum[1], k in 1:glnum[3], e in 1:n_e
    x1ξ2[i,:,k,e] = ∂² * xreshape[i,:,k,e] 
    x2ξ2[i,:,k,e] = ∂² * yreshape[i,:,k,e] 
    x3ξ2[i,:,k,e] = ∂² * zreshape[i,:,k,e] 
end

# derivative with respect to ξ3
x1ξ3 = copy(xreshape)
x2ξ3 = copy(xreshape)
x3ξ3 = copy(xreshape)
for i in 1:glnum[1], j in 1:glnum[2], e in 1:n_e
    x1ξ3[i,j,:,e] = ∂³ * xreshape[i,j,:,e] 
    x2ξ3[i,j,:,e] = ∂³ * yreshape[i,j,:,e] 
    x3ξ3[i,j,:,e] = ∂³ * zreshape[i,j,:,e] 
end

## Reshape and Store in matrix
x1ξ1 = reshape(x1ξ1, (prod(glnum), n_e))
x2ξ1 = reshape(x2ξ1, (prod(glnum), n_e))
x3ξ1 = reshape(x3ξ1, (prod(glnum), n_e))

x1ξ2 = reshape(x1ξ2, (prod(glnum), n_e))
x2ξ2 = reshape(x2ξ2, (prod(glnum), n_e))
x3ξ2 = reshape(x3ξ2, (prod(glnum), n_e))

x1ξ3 = reshape(x1ξ3, (prod(glnum), n_e))
x2ξ3 = reshape(x2ξ3, (prod(glnum), n_e))
x3ξ3 = reshape(x3ξ3, (prod(glnum), n_e))

metrics = [x1ξ1[ijk, e] x1ξ2[ijk, e]  x1ξ3[ijk, e];
           x2ξ1[ijk, e] x2ξ2[ijk, e]  x2ξ3[ijk, e];
           x3ξ1[ijk, e] x3ξ2[ijk, e]  x3ξ3[ijk, e];]
JJ = inv(metrics)
JJ * JJ'