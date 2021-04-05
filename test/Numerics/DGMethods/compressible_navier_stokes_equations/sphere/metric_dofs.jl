include("../boilerplate.jl")
include("../three_dimensional/ThreeDimensionalCompressibleNavierStokesEquations.jl")
include("sphere_helper_functions.jl")

ClimateMachine.init()

hdofs = [180] # # of total gridpoints on the equator
he_list = collect(6:20)
errorlist = []

function pdofs(dofs, he)
    pe = floor(Int, dofs / he / 4) - 1
end

########
# Setup physical and numerical domains
########
for hdof in hdofs, he in he_list 
    hp = pdofs(hdof, he)
    domain = AtmosDomain(radius = 1, height = 0.02)

    grid = DiscretizedDomain(
        domain;
        elements = (vertical = 1, horizontal = he),
        polynomial_order = (vertical = 1, horizontal = hp),
        overintegration_order = (vertical = 0, horizontal = 0),
    )
    println("he = ", he)
    println("hp = ", hp)
    # pick a location
    ijk = 1 
    e = 1  
    J = getjacobian(grid.numerical, ijk, e)
    x,y, z = coordinates(grid.numerical)
    r = [x[ijk,e], y[ijk,e], z[ijk,e]]
    r = r / norm(r)
    a¹ = J[1,:]
    a² = J[2,:]
    a³ = J[3,:]
    nr = J[3,:] ./ norm(J[3,:])
    norm(r - nr)
    iJ = inv(J)
    a₁ = iJ[:,1]
    a₂ = iJ[:,2]
    a₃ = iJ[:,3]
    inr = a₃ ./ norm(a₃)
    push!(errorlist, ((; horizontal_dofs = hdof, horizontal_elements = he), norm(nr - inr)))
    println(norm(nr - inr))
    println("--------")
end