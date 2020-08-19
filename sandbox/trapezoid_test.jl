using Plots
# physical coordinates
inner_radius = 6378100 # [m]
outer_radius = 6378100 + 100000 # [m] adding 100km for the atmosphere [about 1.6%] (its 3km for the ocean)
r1 = (inner_radius / outer_radius)
#

nr = 5
nh = 10
ξ¹ = reshape(collect(range(-1,1, length = nh)), (1,nh)) 
ξ² = reshape(collect(range(-1,1, length = nr)) , (nr,1)) 

r1 = 0.5 # inner radius
r2 = 1.0 # outer radius
npolygon = 5 # polygon approximation
φ = 2π/(2 * npolygon)
x¹ = @. ( (r2-r1)*sin(φ) * (ξ² + 1)/2 + r1 * sin(φ) ) * ξ¹ 
x² = @. (r2-r1)*cos(φ) * (ξ²+1)/2 + 0 * ξ¹ + r1*cos(φ)
p1 = scatter(x¹[:], x²[:], xlims = (-r2*1.1,r2*1.1), ylims = (-r2*1.1,r2*1.1), legend = false,aspectratio = 1)

##
for j in 1:1:(npolygon-1)
    θ = 2π/npolygon * j
    xrot = @. cos(θ) * x¹ - sin(θ) * x²
    yrot = @. sin(θ) * x¹ + cos(θ) * x²
    p1 = scatter!(xrot[:], yrot[:], xlims = (-r2*1.1,r2*1.1), ylims = (-r2*1.1,r2*1.1))
    display(p1)
    sleep(0.1)
end

plot(p1)

##
j=npolygon
θ = 2π/npolygon * j
xrot = @. cos(θ) * x¹ - sin(θ) * x²
yrot = @. sin(θ) * x¹ + cos(θ) * x²
p1 = scatter(xrot[:], yrot[:], xlims = (-r2*1.1,r2*1.1), ylims = (-r2*1.1,r2*1.1), aspectratio = 1, legend = false)
display(p1)
sleep(0.1)
scatter!(x¹[:], x²[:], xlims = (-r2*1.1,r2*1.1), ylims = (-r2*1.1,r2*1.1), legend = false,aspectratio = 1)

