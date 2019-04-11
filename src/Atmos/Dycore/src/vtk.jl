using WriteVTK

#=
This is the 1D WriteMesh routine
=#
function writemesh(base_name, x; fields=(), realelems=1:size(x)[end])
  (Nqr, _) = size(x)
  Nsubcells = (Nqr-1)

  cells = Array{MeshCell{Array{Int,1}}, 1}(undef, Nsubcells * length(realelems))
  for e ∈ realelems
    offset = (e-1) * Nqr
    for i = 1:Nqr-1
        cells[i + (e-1)*Nsubcells] =
          MeshCell(VTKCellTypes.VTK_LINE, offset .+ [i, i+1])
    end
  end

  vtkfile = vtk_grid("$(base_name)", @view(x[:]), cells; compress=false)
  for (name, v) ∈ fields
    vtk_point_data(vtkfile, v, name)
  end
  outfiles = vtk_save(vtkfile)
end

#=
This is the 2D WriteMesh routine
=#
function writemesh(base_name, x, y; z=nothing, fields=(), realelems=1:size(x)[end])
  @assert size(x) == size(y)
  (Nqr, Nqs, _) = size(x)
  Nsubcells = (Nqr-1) * (Nqs-1)

  cells = Array{MeshCell{Array{Int,1}}, 1}(undef, Nsubcells * length(realelems))
  ind = LinearIndices((1:Nqr, 1:Nqs))
  for e ∈ realelems
    offset = (e-1) * Nqr * Nqs
    for j = 1:Nqs-1
      for i = 1:Nqr-1
        cells[i + (j-1)*(Nqr-1) + (e-1)*Nsubcells] =
        MeshCell(VTKCellTypes.VTK_PIXEL, offset .+ ind[i:i+1,j:j+1][:])
      end
    end
  end

  if z == nothing
    vtkfile = vtk_grid("$(base_name)", @view(x[:]), @view(y[:]), cells;
                       compress=false)
  else
    vtkfile = vtk_grid("$(base_name)", @view(x[:]), @view(y[:]), @view(z[:]),
                       cells; compress=false)
  end
  for (name, v) ∈ fields
    vtk_point_data(vtkfile, v, name)
  end
  outfiles = vtk_save(vtkfile)
end

#=
This is the 3D WriteMesh routine
=#
function writemesh(base_name, x, y, z; fields=(), realelems=1:size(x)[end])
  (Nqr, Nqs, Nqt, _) = size(x)
  (Nr, Ns, Nt) = (Nqr-1, Nqs-1, Nqt-1)
  Nsubcells = Nr * Ns * Nt
  cells = Array{MeshCell{Array{Int,1}}, 1}(undef, Nsubcells * length(realelems))
  ind = LinearIndices((1:Nqr, 1:Nqs, 1:Nqt))
  for e ∈ realelems
    offset = (e-1) * Nqr * Nqs * Nqt
    for k = 1:Nt
      for j = 1:Ns
        for i = 1:Nr
          cells[i + (j-1) * Nr + (k-1) * Nr * Ns + (e-1) * Nsubcells] =
            MeshCell(VTKCellTypes.VTK_VOXEL,
                     offset .+ ind[i:i+1, j:j+1, k:k+1][:])
        end
      end
    end
  end

  vtkfile = vtk_grid("$(base_name)", @view(x[:]), @view(y[:]), @view(z[:]),
                     cells; compress=false)
  for (name, v) ∈ fields
    vtk_point_data(vtkfile, v, name)
  end
  outfiles = vtk_save(vtkfile)
end
