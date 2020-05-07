using WriteVTK

#=
This is the 1D WriteMesh routine
=#
function writemesh(base_name, x1; fields = (), realelems = 1:size(x1)[end])
    (Nqr, _) = size(x1)
    Nsubcells = (Nqr - 1)

    cells =
        Array{MeshCell{Array{Int, 1}}, 1}(undef, Nsubcells * length(realelems))
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
function writemesh(
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

    cells =
        Array{MeshCell{Array{Int, 1}}, 1}(undef, Nsubcells * length(realelems))
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
function writemesh(
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
    cells =
        Array{MeshCell{Array{Int, 1}}, 1}(undef, Nsubcells * length(realelems))
    ind = LinearIndices((1:Nqr, 1:Nqs, 1:Nqt))
    for e in realelems
        offset = (e - 1) * Nqr * Nqs * Nqt
        for k in 1:Nt
            for j in 1:Ns
                for i in 1:Nr
                    cells[i + (j - 1) * Nr + (k - 1) * Nr * Ns + (e - 1) *
                                                                 Nsubcells] =
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
