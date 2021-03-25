include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("../sphere/sphere_helper_functions.jl")
include("gradient.jl")
ClimateMachine.init()

function pdofs(dofs, he)
    pe = floor(Int, dofs / he / 4) - 1
end

########
# Setup physical and numerical domains
########
hdof = 90 
he = 3
hp = pdofs(hdof, he)
domain = AtmosDomain(radius = 1, height = 0.02)

grid = DiscretizedDomain(
    domain;
    elements = (vertical = 1, horizontal = he),
    polynomial_order = (vertical = 1, horizontal = hp),
    overintegration_order = (vertical = 0, horizontal = 0),
)

##
x,y,z = coordinates(grid)
r = sqrt.(x .^2 .+ y .^2 .+ z .^2)
∇  =  Nabla(grid)
∇r =  ∇(r)
∇ϕ =  ∇(r .^(-1))
∇rᴱ = similar(∇r)

for i in eachindex(x)
    ∇rᴱ[i,1] = x[i] / r[i]
    ∇rᴱ[i,2] = y[i] / r[i]
    ∇rᴱ[i,3] = z[i] / r[i]
end

maximum(abs.(∇r - ∇rᴱ)) / maximum(abs.(∇rᴱ))
minimum(abs.(∇r - ∇rᴱ))
norm(∇r - ∇rᴱ) / norm(∇rᴱ)
sum(abs.(∇r - ∇rᴱ)) / sum(abs.(∇rᴱ))