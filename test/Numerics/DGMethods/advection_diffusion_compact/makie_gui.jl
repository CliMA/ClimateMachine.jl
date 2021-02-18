include("makie_src/QuickVizExample.jl")
#using QV

function makie_gui(
    Ω,  # topl
    dgΩ, # grid
    rhs_DGsource, 
    rhs_anal,
    )

    # construct gridhelpers
    gridhelper = QV.GridHelper(dgΩ)
    x, y, z = QV.coordinates(dgΩ)
    ϕ = QV.ScalarField(copy(x), gridhelper)

    # construct new grid
    xnew = range(Ω[1][1], Ω[1][2], length = 128)
    ynew = range(Ω[2][1], Ω[2][2], length = 128)
    znew = range(Ω[3][1], Ω[3][1], length = 1)

    # load data
    u = rhs_DGsource
    v = rhs_anal

    # interpolate each moment in time to a new grid
    nt = length(u) 
    ut = zeros(length(xnew), length(ynew), nt)
    vt = zeros(length(xnew), length(ynew), nt)
    tic = time()
    trange = collect(1:nt)
    for i in 1:nt
        ϕ .= u[i]
        ut[:, :, i] .= view(ϕ(xnew, ynew, znew), :, :, 1)
        ϕ .= v[i]
        vt[:, :, i] .= view(ϕ(xnew, ynew, znew), :, :, 1)
    end
    toc = time()
    println("interpolation time is $(toc - tic) seconds")
    # visualize (z-axis is time here)
    states = [ut, vt]
    statenames = ["rhs_DGsource", "rhs_anal"]
    QV.volumeslice(states, statenames = statenames)
end