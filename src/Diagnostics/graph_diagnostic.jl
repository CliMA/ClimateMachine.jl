using Plots; pyplot()
using DataFrames, FileIO

using CLIMA.VariableTemplates

include(joinpath("..","src","Diagnostics","diagnostic_vars.jl"))

function usage()
    println("""
Usage:
    julia graph_diagnostic.jl <diagnostic_file.jld2> <diagnostic_name>""")
end

function start(args::Vector{String})
    #data = load(args[1])

    # USER INPUTS:
    FT = Float64
    vars_diag = vars_diagnostic(FT)
    varnames_diag = fieldnames(vars_diag)
    out_vars = string.(varnames_diag)

    data_files = collect(filter(x->occursin(".jld2",x) && occursin("diagnostics",x), readdir("ck-TestDycoms")))
    data_files = map(x-> (mtime(joinpath("ck-TestDycoms",x)),x), data_files)
    data_file = last(first(sort(data_files,by=first, rev=true)))
    @show data_files
    @show data_file

    data = load(joinpath("ck-TestDycoms", data_file))

#    time = 0.0
     time = 0.05

    @show keys(data)
    println("data for $(length(data)) time steps in file")
    #time_data = first(keys(data))

    time_data = string(time)
    Nqk = size(data[time_data], 1)
    nvertelem = size(data[time_data], 2)
    all_vars = ntuple(i->zeros(Nqk * nvertelem), length(out_vars))

    t = time_data
    for ev in 1:nvertelem
        for k in 1:Nqk
            dv = diagnostic_vars(data[t][k,ev])
            for i in 1:length(out_vars)
              all_vars[i][k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[i]))
            end
        end
    end
    Z = all_vars[1]

    each_plot = []
    for i in 1:length(out_vars)
      push!(each_plot, plot(all_vars[i], Z,
                            linewidth=2,
                            # xaxis=(out_vars[i], (-0, 10), 0:1:10),
                            xaxis=(out_vars[i]),
                            yaxis=("Altitude[m]", (0, max(Z...))),
                            label=(out_vars[i]),
                            ))
    end

    f=font(11,"courier")
    time_str = string("t = ", ceil(time), " s")

    mkpath(joinpath("ck-TestDycoms","plots"))
    for (k,i) in var_groups(FT)
      all_plots = plot(each_plot[i]..., layout = (1,length(i)), titlefont=f, tickfont=f, legendfont=f, guidefont=f, title=time_str)
      plot!(size=(900,800))
      savefig(all_plots, joinpath("ck-TestDycoms","plots",string(k)*".png"))
    end

  return varnames_diag

end

#if length(ARGS) != 3 || !endswith(ARGS[1], ".jld2")
#    usage()
#end
results = start(ARGS)
nothing
