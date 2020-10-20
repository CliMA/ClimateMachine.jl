export curl

function curl(grid, Q)
    d1 = Diagnostics.VectorGradient(grid, Q[1], 1)
    d2 = Diagnostics.VectorGradient(grid, Q[2], 1)
    d3 = Diagnostics.VectorGradient(grid, Q[3], 1)
    vgrad = Diagnostics.VectorGradients(d1, d2, d3)
    vort = Diagnostics.Vorticity(grid, vgrad)
    return vort
end
