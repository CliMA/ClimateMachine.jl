using JLD2
using GLMakie

filename = "/Users/ballen/Projects/Clima/CLIMA/output/vtk_bickley_2D/bickley_jet.jld2"
DOF = 32

f = jldopen(filename, "r+")
include("bigfileofstuff.jl")
include("vizinanigans.jl")
include("ScalarFields.jl")

dg_grid = f["grid"]
gridhelper = GridHelper(dg_grid)
x, y, z = coordinates(dg_grid)
xC, yC, zC = cellcenters(dg_grid)

ϕ = ScalarField(copy(x), gridhelper)

newx = range(-2π, 2π, length = DOF)
newy = range(-2π, 2π, length = DOF)
norm(f["100"])

##

Q = f["0"]
dof, nstates, nelems = size(Q)

states = Array{Float64, 3}[]
statenames = ["ρ", "ρu", "ρv", "ρθ"]

for i in 1:nstates
    state = zeros(length(newx), length(newy), 101)
    push!(states, state)
end

for i in 0:100
    println("interpolating step " * string(i))
    Qⁱ = f[string(i)]

    for j in 1:nstates
        ϕ .= Qⁱ[:, j, :]
        states[j][:, :, i + 1] .= ϕ(newx, newy, threads = true)
    end
end

# f["states"] = states
close(f)

##

scene = volumeslice(states, statenames = statenames)
