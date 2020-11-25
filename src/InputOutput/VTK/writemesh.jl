using WriteVTK

function vtk_connectivity_map_highorder(Nqi, Nqj = 1, Nqk = 1)
    connectivity = Array{Int, 1}(undef, Nqi * Nqj * Nqk)
    L = LinearIndices((1:Nqi, 1:Nqj, 1:Nqk))

    corners = (
        (1, 1, 1),
        (Nqi, 1, 1),
        (Nqi, Nqj, 1),
        (1, Nqj, 1),
        (1, 1, Nqk),
        (Nqi, 1, Nqk),
        (Nqi, Nqj, Nqk),
        (1, Nqj, Nqk),
    )
    edges = (
        (2:(Nqi - 1), 1:1, 1:1),
        (Nqi:Nqi, 2:(Nqj - 1), 1:1),
        (2:(Nqi - 1), Nqj:Nqj, 1:1),
        (1:1, 2:(Nqj - 1), 1:1, 1:1),
        (2:(Nqi - 1), 1:1, Nqk:Nqk),
        (Nqi:Nqi, 2:(Nqj - 1), Nqk:Nqk),
        (2:(Nqi - 1), Nqj:Nqj, Nqk:Nqk),
        (1:1, 2:(Nqj - 1), Nqk:Nqk),
        (1:1, 1:1, 2:(Nqk - 1)),
        (Nqi:Nqi, 1:1, 2:(Nqk - 1)),
        (1:1, Nqj:Nqj, 2:(Nqk - 1)),
        (Nqi:Nqi, Nqj:Nqj, 2:(Nqk - 1)),
    )
    faces = (
        (1:1, 2:(Nqj - 1), 2:(Nqk - 1)),
        (Nqi:Nqi, 2:(Nqj - 1), 2:(Nqk - 1)),
        (2:(Nqi - 1), 1:1, 2:(Nqk - 1)),
        (2:(Nqi - 1), Nqj:Nqj, 2:(Nqk - 1)),
        (2:(Nqi - 1), 2:(Nqj - 1), 1:1),
        (2:(Nqi - 1), 2:(Nqj - 1), Nqk:Nqk),
    )
    if Nqj == Nqk == 1
        @assert Nqi > 1
        corners = (corners[1:2]...,)
        edges = (edges[1],)
        faces = ()
    elseif Nqk == 1
        @assert Nqi > 1
        @assert Nqj > 1
        corners = (corners[1:4]...,)
        edges = (edges[1:4]...,)
        faces = (faces[5],)
    end

    # corners
    iter = 1
    for (i, j, k) in corners
        connectivity[iter] = L[i, j, k]
        iter += 1
    end
    # edges
    for (is, js, ks) in edges
        for k in ks, j in js, i in is
            connectivity[iter] = L[i, j, k]
            iter += 1
        end
    end
    # faces
    for (is, js, ks) in faces
        for k in ks, j in js, i in is
            connectivity[iter] = L[i, j, k]
            iter += 1
        end
    end
    # interior
    for k in 2:(Nqk - 1), j in 2:(Nqj - 1), i in 2:(Nqi - 1)
        connectivity[iter] = L[i, j, k]
        iter += 1
    end
    return connectivity
end

#=
This is the 1D WriteMesh routine
=#
function writemesh_highorder(
    base_name,
    x1;
    x2 = nothing,
    x3 = nothing,
    fields = (),
    realelems = 1:size(x1)[end],
)
    (Nqr, _) = size(x1)

    con = vtk_connectivity_map_highorder(Nqr)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, length(realelems))

    for (i, e) in enumerate(realelems)
        offset = (e - 1) * Nqr
        cells[i] = MeshCell(VTKCellTypes.VTK_LAGRANGE_CURVE, offset .+ con)
    end

    if x2 == nothing
        @assert isnothing(x3)
        vtkfile =
            vtk_grid("$(base_name)", @view(x1[:]), cells; compress = false)
    elseif x3 == nothing
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            cells;
            compress = false,
        )
    else
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            @view(x3[:]),
            cells;
            compress = false,
        )
    end
    for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 2D WriteMesh routine
=#
function writemesh_highorder(
    base_name,
    x1,
    x2;
    x3 = nothing,
    fields = (),
    realelems = 1:size(x1)[end],
)
    @assert size(x1) == size(x2)
    (Nqr, Nqs, _) = size(x1)
    @assert Nqr == Nqs
    con = vtk_connectivity_map_highorder(Nqr, Nqs)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, length(realelems))
    for (i, e) in enumerate(realelems)
        offset = (e - 1) * Nqr * Nqs
        cells[i] =
            MeshCell(VTKCellTypes.VTK_LAGRANGE_QUADRILATERAL, offset .+ con[:])
    end

    if x3 == nothing
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            cells;
            compress = false,
        )
    else
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            @view(x3[:]),
            cells;
            compress = false,
        )
    end
    for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 3D WriteMesh routine
=#
function writemesh_highorder(
    base_name,
    x1,
    x2,
    x3;
    fields = (),
    realelems = 1:size(x1)[end],
)
    (Nqr, Nqs, Nqt, _) = size(x1)
    @assert Nqr == Nqs == Nqt
    con = vtk_connectivity_map_highorder(Nqr, Nqs, Nqt)
    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, length(realelems))
    for (i, e) in enumerate(realelems)
        offset = (e - 1) * Nqr * Nqs * Nqt
        cells[i] = MeshCell(VTKCellTypes.VTK_LAGRANGE_HEXAHEDRON, offset .+ con)
    end

    vtkfile = vtk_grid(
        "$(base_name)",
        @view(x1[:]),
        @view(x2[:]),
        @view(x3[:]),
        cells;
        compress = false,
    )
    for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 1D WriteMesh routine
=#
function writemesh_raw(base_name, x1;
    x2 = nothing,
    x3 = nothing,
                       fields = (), realelems = 1:size(x1)[end])
    (Nqr, _) = size(x1)
    Nsubcells = (Nqr - 1)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, Nsubcells * length(realelems))
    for e in realelems
        offset = (e - 1) * Nqr
        for i in 1:(Nqr - 1)
            cells[i + (e - 1) * Nsubcells] =
                MeshCell(VTKCellTypes.VTK_LINE, offset .+ [i, i + 1])
        end
    end

    vtkfile = vtk_grid("$(base_name)", @view(x1[:]), cells; compress = false)
    for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 2D WriteMesh routine
=#
function writemesh_raw(
    base_name,
    x1,
    x2;
    x3 = nothing,
    fields = (),
    realelems = 1:size(x1)[end],
)
    @assert size(x1) == size(x2)
    (Nqr, Nqs, _) = size(x1)
    Nsubcells = (Nqr - 1) * (Nqs - 1)

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, Nsubcells * length(realelems))
    ind = LinearIndices((1:Nqr, 1:Nqs))
    for e in realelems
        offset = (e - 1) * Nqr * Nqs
        for j in 1:(Nqs - 1)
            for i in 1:(Nqr - 1)
                cells[i + (j - 1) * (Nqr - 1) + (e - 1) * Nsubcells] = MeshCell(
                    VTKCellTypes.VTK_PIXEL,
                    offset .+ ind[i:(i + 1), j:(j + 1)][:],
                )
            end
        end
    end

    if x3 == nothing
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            cells;
            compress = false,
        )
    else
        vtkfile = vtk_grid(
            "$(base_name)",
            @view(x1[:]),
            @view(x2[:]),
            @view(x3[:]),
            cells;
            compress = false,
        )
    end
    for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end

#=
This is the 3D WriteMesh routine
=#
function writemesh_raw(
    base_name,
    x1,
    x2,
    x3;
    fields = (),
    realelems = 1:size(x1)[end],
)
    (Nqr, Nqs, Nqt, _) = size(x1)
    (Nr, Ns, Nt) = (Nqr - 1, Nqs - 1, Nqt - 1)
    Nsubcells = Nr * Ns * Nt

    M = MeshCell{VTKCellTypes.VTKCellType, Array{Int, 1}}
    cells = Array{M, 1}(undef, Nsubcells * length(realelems))
    ind = LinearIndices((1:Nqr, 1:Nqs, 1:Nqt))
    for e in realelems
        offset = (e - 1) * Nqr * Nqs * Nqt
        for k in 1:Nt
            for j in 1:Ns
                for i in 1:Nr
                    cells[i + (j - 1) * Nr + (k - 1) * Nr * Ns + (e - 1) * Nsubcells] =
                        MeshCell(
                            VTKCellTypes.VTK_VOXEL,
                            offset .+ ind[i:(i + 1), j:(j + 1), k:(k + 1)][:],
                        )
                end
            end
        end
    end

    vtkfile = vtk_grid(
        "$(base_name)",
        @view(x1[:]),
        @view(x2[:]),
        @view(x3[:]),
        cells;
        compress = false,
    )
    for (name, v) in fields
        vtk_point_data(vtkfile, v, name)
    end
    outfiles = vtk_save(vtkfile)
end
