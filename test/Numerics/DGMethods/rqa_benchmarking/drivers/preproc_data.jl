using NCDatasets
using Plots
using Dierckx
using MPI
using ClimateMachine.Topologies
using Interpolations


MPI.Init()

pyplot()

# Uses ClimateMachine.Topologies

data = NCDataset("/Users/asridhar/Research/Codes/ClimateMachine.jl/topodata.nc");
λ = (data["X"][:] .+ 180) .* π/180; # Longitude in radians
ϕ = reverse(data["Y"][:]) .* π/180; # Latitude in radians 
r = data["topo"][:]; # Elevation in meters [0 to Everest] No Bathmetry
r = reverse(r,dims=2);
const rp = Float64(6.371e6); # get_parameters(ClimaParameters...)
# Generate Spline
skip_var = 5;
get_elevation = Spline2D(λ[1:skip_var:end],ϕ[1:skip_var:end],r[1:skip_var:end,1:skip_var:end], kx = 4, ky=4);


# Testing

vert_range = grid1d(
    rp,
    rp + 10e3,
    nothing,
    nelem = 10,
)

# Parametric Sphere
X(r,λ,ϕ) = r * cos(λ) * cos(ϕ)
Y(r,λ,ϕ) = r * sin(λ) * cos(ϕ)
Z(r,λ,ϕ) = r * sin(ϕ)

# Plot Test
Λ = 0:π/180:2π
Φ = -π/2:π/90:π/2

function R(λ,ϕ) 
    r = get_elevation(λ,ϕ)
    r <= -0 ? 0 : r
    return r*10 .+ rp
end

xs = [X(R(λ,ϕ),λ,ϕ) for λ in Λ, ϕ in Φ]
ys = [Y(R(λ,ϕ),λ,ϕ) for λ in Λ, ϕ in Φ]
zs = [Z(R(λ,ϕ),λ,ϕ) for λ in Λ, ϕ in Φ]


# Trying reshape and Cartesian Spline :D 
xr = reshape(xs, length(xs), 1)
yr = reshape(ys, length(ys), 1)
zr = reshape(zs, length(zs), 1)


surface(xs,ys,zs)

x₀ = similar(xs)
y₀ = similar(ys)
z₀ = similar(zs)

for ii = 1:size(xs)[1]
    for jj = 1:size(ys)[2]
        x,y,z = xs[ii,jj], ys[ii,jj], zs[ii,jj]
        x₀[ii,jj],y₀[ii,jj],z₀[ii,jj] = cubed_sphere_unwarp(EquiangularCubedSphere(),
                                                            x,
                                                            y,
                                                            z)
    end
end

for ii = 1:size(xs)[1]
    for jj = 1:size(ys)[2]
        x,y,z = x₀[ii,jj], y₀[ii,jj], z₀[ii,jj]
        x₀[ii,jj],y₀[ii,jj],z₀[ii,jj] = earth_sphere_warp(EquiangularCubedSphere(),
                                                            x,
                                                            y,
                                                            z, 
                                                            max(abs(x),abs(y),abs(z)))
    end
end
