using ClimateMachine
ClimateMachine.init()
##
include("../boilerplate.jl")
include("../plotting/vizinanigans.jl")

##
domain = AtmosDomain(radius = 1, height = 1)

ddomain = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = 4),
    polynomial_order = (vertical = 1, horizontal = 4),
    overintegration_order = 0,
)

grid = ddomain.numerical
xC, yC, zC = cellcenters(grid)
x, y, z = coordinates(grid)
ihelper = InterpolationHelper(grid)

##
function lagrange_eval(f, newr, ihelper::InterpolationHelper)
    fnew = lagrange_eval(f, newr[1], newr[2], newr[3],  
                  ihelper.points[1], ihelper.points[2], ihelper.points[3], 
                  ihelper.quadrature[1], ihelper.quadrature[2], ihelper.quadrature[3])
    return fnew
end

##
x[1,e]
lagrange_eval(reshape(x[:,e], polynomialorders(grid) .+ 1), (-1,-1,-1), ihelper)
lagrange_eval(reshape(y[:,e], polynomialorders(grid) .+ 1), (-1,-1,-1), ihelper)
lagrange_eval(reshape(z[:,e], polynomialorders(grid) .+ 1), (-1,-1,-1), ihelper)

function closure_position(x,y,z,e,grid, ihelper)
    function position(ξ¹, ξ², ξ³)
        ξ⃗ = (ξ¹, ξ², ξ³)
        x¹ = lagrange_eval(reshape(x[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
        x² = lagrange_eval(reshape(y[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
        x³ = lagrange_eval(reshape(z[:,e], polynomialorders(grid) .+ 1), ξ⃗, ihelper)
        x⃗ = [x¹, x², x³]
        return x⃗
    end
end

position = closure_position(x,y,z,e,grid, ihelper)

position(1,1,1)
x[end,e]
y[end,e]
z[end,e]

##
fig = Figure()
lscene = fig[1:4, 2:4] = LScene(fig.scene)
e = 1
scatter!(lscene, x[:, e], y[:, e], z[:, e], color = :black, markersize = 100.0, strokewidth = 0)
scatter!(lscene, xC[e:e], yC[e:e], zC[e:e], color = :red, markersize = 100.0, strokewidth = 0)

xC[e]^2 +  yC[e]^2 + zC[e]^2



##

visualize(grid)

##
