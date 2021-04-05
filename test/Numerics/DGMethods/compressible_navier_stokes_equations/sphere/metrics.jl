include("../boilerplate.jl") # most functions in vector fields, include("test/Numerics/DGMethods/compressible_navier_stokes_equations/boilerplate.jl") 
ClimateMachine.init()
# Ω = Interval(0,1) × Interval(-1,1) × Interval(exp(1), π) 
Ω =  AtmosDomain(radius = 1, height = 0.02)

elements = (vertical = 1, horizontal = 4) # horizontal here means horizontal^2 * 6
polynomialorder = (vertical = 1, horizontal = 4)
grid = DiscontinuousSpectralElementGrid(
    Ω,
    elements = elements,
    polynomialorder = polynomialorder
)

x,y,z  = coordinates(grid)
# visualize(grid) # can only visualize grid if vertical = horizontal polynomial order, otherwise can't change face

## volume check
M = grid.vgeo[:, grid.Mid, :]
J = constructdeterminant(grid)
exact = 4π/3 * ( (Ω.height + Ω.radius)^3  - (Ω.radius)^3 )
volumeerror = ( sum(M) - exact ) / exact

ijk = 1 # ijk max is size(x)[1]
e = 1   # e max is size(x)[2]

fullJ  = getjacobian(grid, ijk, e)
fulliJ = inv(getjacobian(grid, ijk, e))
r = getposition(grid, ijk, e)

# comparison one
abs(J[ijk,e] -  det(fulliJ)) / J[ijk,e]

# check normal directions
r /= norm(r) # outward pointing normal
n̂ = fulliJ[:,3] / norm(fulliJ[:,3])
norm(r[:] - n̂)
# note that fulliJ[:,3] is the vertical component
fulliJ[:,3]' * fullJ[1,:]
fulliJ[:,3]' * fullJ[2,:]
fulliJ[:,3]' * fullJ[3,:]
# note that the vertical direction is not orthogonal to the two horizontal directions in general
# contravariant check
fullJ[3,:]' * fullJ[1,:]
fullJ[3,:]' * fullJ[2,:]

# covariant check
fulliJ[:,3]' * fulliJ[:, 1]
fulliJ[:,3]' * fulliJ[:, 2]

##
face = 5
element = 1
ij = 1
index = grid.vmap⁻[ij, face, element]
n̂ =  [grid.sgeo[i,ij,face,element] for i in 1:3]
(x[index], y[index], z[index])
fullJ[3,:] / norm(fullJ[3,:])
fulliJ[:,3] / norm(fulliJ[:,3])

##
# hand build jacobian
n_ijk, n_e = size(x)
handbuildJ = [1/det(getjacobian(grid, ijk, e)) for ijk in 1:n_ijk, e in 1:n_e]
ωx = reshape(grid.ω[1], (length(grid.ω[1]), 1, 1, 1))
ωy = reshape(grid.ω[2], (1, length(grid.ω[2]), 1, 1))
ωz = reshape(grid.ω[3], (1, 1, length(grid.ω[3]), 1))
ω = reshape(ωx .* ωy .* ωz, (size(M)[1],1) )
handbuildM = handbuildJ .* ω

norm(handbuildM - M) / norm(M)

exact = 4π/3 * ( (Ω.height + Ω.radius)^3  - (Ω.radius)^3 )
volumeerror = ( sum(handbuildM) -  exact) / exact

## Hand build metrics
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
## Check Metric Identity


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

(det(metrics) - J[ijk,e]) / det(metrics)

##
edges = []
faces = [1, 2, 3, 4, 5, 6]
oppositefaces = [2, 1, 4, 3, 6, 5]
for i in faces
    for j in setdiff(setdiff(faces, i), oppositefaces[i])
        intersection = intersect( grid.vmap⁻[: , i, :],  grid.vmap⁻[: , j, :])
        prune_zeros = setdiff(intersection, [0])
        edges = union(edges, prune_zeros)
    end
end

elist = [1, 23, 51, 76]
pts = [Point3f0(getposition(grid, ijk, e)[:]) for e in elist] 
ptdir1 = [Point3f0(getjacobian(grid, ijk, e)[1,:]/norm(getjacobian(grid, ijk, e)[1,:])) for e in elist]
ptdir2 = [Point3f0(getjacobian(grid, ijk, e)[2,:]/norm(getjacobian(grid, ijk, e)[2,:])) for e in elist] 
ptdir3 = [Point3f0(getjacobian(grid, ijk, e)[3,:]/norm(getjacobian(grid, ijk, e)[3,:])) for e in elist] 

tpts = [pts..., pts..., pts...]
tptdir = [ptdir1..., ptdir2..., ptdir3...]

ptdir4 = [Point3f0(inv(getjacobian(grid, ijk, e))[:,1]/norm(inv(getjacobian(grid, ijk, e))[:,1])) for e in elist]
ptdir5 = [Point3f0(inv(getjacobian(grid, ijk, e))[:,2]/norm(inv(getjacobian(grid, ijk, e))[:,2])) for e in elist] 
ptdir6 = [Point3f0(inv(getjacobian(grid, ijk, e))[:,3]/norm(inv(getjacobian(grid, ijk, e))[:,3])) for e in elist] 

tptdir2 = [ptdir4..., ptdir5..., ptdir6...]

scene = scatter(x[:], y[:], z[:], markersize = 0)
# scatter!(scene.figure[1,1], x[edges], y[edges], z[edges], color = :red, markersize = 30)
GLMakie.arrows!(scene.figure[1,1], tpts, tptdir, arrowsize = 0.05, linecolor = :blue, arrowcolor = :darkblue, linewidth = 10)
GLMakie.arrows!(scene.figure[1,1], tpts, tptdir2, arrowsize = 0.05, linecolor = :red, arrowcolor = :darkblue, linewidth = 10)
display(scene)

ptdir3 - ptdir6

##
v⃗ⁱ =  getcontravariant(grid, ijk, e)
v⃗ᵢ =  getcovariant(grid, ijk, e)

##
vector = [x[1], y[1], z[1]] 

contravariant = (v⃗ᵢ[1]' * vector) * v⃗ⁱ[1] + (v⃗ᵢ[2]' * vector) * v⃗ⁱ[2] + (v⃗ᵢ[3]' * vector) * v⃗ⁱ[3]
covariant = (v⃗ⁱ[1]' * vector) * v⃗ᵢ[1] + (v⃗ⁱ[2]' * vector) * v⃗ᵢ[2] + (v⃗ⁱ[3]' * vector) * v⃗ᵢ[3]

# in the covariant representation the velocity field components should be order 1 with respect to one another

v⃗ = vector
components(v⃗, grid, ijk, e, Contravariant())
v¹, v², v³ = components(vector, grid, ijk, e, Contravariant())
v⃗₁, v⃗₂, v⃗₃ = getcontravariant(grid, ijk, e) 

v₁, v₂, v₃ = components(vector, grid, ijk, e, Covariant())
v⃗¹, v⃗², v⃗³ = getcovariant(grid, ijk, e)

vʳ, vᶿ, vᵠ = components(vector, grid, ijk, e, Spherical())
r̂, θ̂, φ̂ = getspherical(grid, ijk, e)

## check the different representations
println(v⃗)
println(v¹ * v⃗₁ +  v² * v⃗₂  + v³ * v⃗₃ )
println(v₁ * v⃗¹ +  v₂ * v⃗²  + v₃ * v⃗³ )
println(vʳ * r̂ + vᶿ * θ̂  + vᵠ * φ̂ )

##
dot(v⃗₁, v⃗₃)/ (norm(v⃗₁) * norm( v⃗₃))
dot(v⃗₂, v⃗₃)/ (norm(v⃗₃) * norm( v⃗₃))
dot(v⃗¹, v⃗³)/ (norm(v⃗¹) * norm( v⃗³))
dot(v⃗², v⃗³)/ (norm(v⃗²) * norm( v⃗³))
dot(v⃗₃, v⃗³)/ (norm(v⃗₃) * norm( v⃗³))

## vector field
x,y,z = coordinates(grid)
ϕ⃗ = VectorField(data = (x, y, z), grid = grid, representation = Cartesian())

ϕ⃗ᵢ = VectorField(ϕ⃗, representation = Covariant())
ϕ⃗ⁱ = VectorField(ϕ⃗, representation = Contravariant())
ϕ⃗ˢ = VectorField(ϕ⃗, representation = Spherical())
##
ϕ⃗[1,1];
ϕ⃗ᵢ[1,1];
ϕ⃗ⁱ[1,1];
ϕ⃗ˢ[1,1];


ϕ⃗(Cartesian())[1,1]
ϕ⃗(Spherical())[1,1]
ϕ⃗(Covariant())[1,1]
ϕ⃗(Contravariant())[1,1]

##
function grabclosest(x,y,z, xgrid, ygrid, zgrid)
    distances = @. (x - xgrid)^2 + (y - ygrid)^2 + (z - zgrid)^2
    amin = argmin(distances)
    return  (amin[1], amin[2])
end

grabclosest(2,1,1, x, y, z)