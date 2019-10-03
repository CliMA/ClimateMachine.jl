#### Export functions based on primitive data

"""
    export_data(data::Array, filename::S, headers::Vector{S})

Export `data` array to filename `filename` with headers `headers`.
"""
function export_data(data::Array, filename::S, headers::Vector{S}) where S<:AbstractString
  s = size(data)
  open(filename, "w") do f
    write(f, join(headers, ",")*"\n")
    for i in 1:s[1], j in 1:s[2]
      if j==s[2]
        @printf(f, "%18.8f\n", data[i,j])
      else
        @printf(f, "%18.8f,", data[i,j])
      end
    end
  end
end

"""
    import_data!(data::Array, filename::S, headers::Vector{S})

Import `data` array from filename `filename` with headers `headers`, exported from `export_csv`.
"""
function import_data!(data::Array{DT}, filename::S, headers::Vector{S}) where {S<:AbstractString,DT}
  data_all = readdlm(filename, ',')
  data .= DT.(data_all[2:end,:]) # Remove headers
end
