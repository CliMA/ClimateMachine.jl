# graph_diagnostic.jl

using DataFrames, FileIO

include("diagnostic_vars.jl")

function usage()
    println("""
Usage:
    julia graph_diagnostic.jl <diagnostic_file.jld2> <diagnostic_name>""")
end

function start(args::Vector{String})
    data = load(args[1])
    println("data for $(length(data)) time steps in file")
    key1 = first(keys(data))
    Nqk = size(data[key1], 1)
    nvertelem = size(data[key1], 2)

    Z = zeros(Nqk * nvertelem)
    for ev in 1:nvertelem
        for k in 1:Nqk
            dv = diagnostic_vars(data[key1][k,ev])
            Z[k+(ev-1)*Nqk] = dv.z
        end
    end

    V = zeros(Nqk * nvertelem)
    for key in keys(data)
        for ev in 1:nvertelem
            for k in 1:Nqk
                dv = diagnostic_vars(data[key][k,ev])
                V[k+(ev-1)*Nqk] += getproperty(dv, Symbol(args[2]))
            end
        end
    end

    df = DataFrame(x = V, y = Z)

    # use graphing framework of choice to plot data in `df`

    # VegaLite:
    #p = df |>
    #@vlplot(:line,
    #        x={:x, title=args[2]},
    #        y={:y, title="Z"},
    #        title=args[2])
    #save("graph-$(args[2])_z.png", p)
end

if length(ARGS) != 3 || !endswith(ARGS[1], ".jld2")
    usage()
end
start(ARGS)

