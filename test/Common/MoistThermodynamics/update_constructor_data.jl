#=
This file is for updating the NCDataset database that stores
input values to the moist thermodynamic state constructors
which have caused convergence issues. Updating this database
allows us to optimize the convergence rate of the moist
thermodynamics constructor for a variety of realistic input
values.
=#

using NCDatasets
using DelimitedFiles

"""
    compress_targz(file)

Platform-independent file compression
"""
function compress_targz(file)
  if Sys.iswindows()
    error("Needs to be implemented")
  else
    run(`tar -zcvf $file $(readdir())`)
  end
end

const archive = true
const clean = true
const append = false
const create = true

# Error check
@assert !(append && create) # cannot create and append
@assert !(append && clean)  # cannot clean and append

folder = joinpath(@__DIR__,"MTConstructorData")
output_file = joinpath(@__DIR__,"MTConstructorDataZipped.tar.gz")
mkpath(folder)

constructors = Dict("PhaseEquil"=>(:e_int, :ρ, :q_tot),)
get_nc(k) = joinpath(folder,"test_data_$(k).nc")
get_data_to_append(k) = joinpath(@__DIR__,"test_data_$(k).csv")

if clean
  for k in keys(constructors)
    rm(get_nc(k); force=true)
  end
end
FT = Float64
if create || append
  for (k,v) in constructors
    if isfile(get_data_to_append(k))
      if append
        ds = Dataset(get_nc(k),"a")
      elseif create
        ds = Dataset(get_nc(k),"c")
      end
      data_all = readdlm(get_data_to_append(k), ',')
      for (i,_v) in enumerate(v)
        vardata = Array{FT}(data_all[2:end,i])
        s = string(_v)
        defVar(ds, s, vardata, ("datapoint",))
      end
      close(ds)
    end
  end
end

if archive
  cd(folder) do
    compress_targz(output_file)
  end
end

