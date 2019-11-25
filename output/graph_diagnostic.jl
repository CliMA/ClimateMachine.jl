using Plots; pyplot()
using DataFrames, FileIO

using CLIMA.VariableTemplates

function vars_diagnostic(FT)
  @vars begin
    z::FT
    u::FT                  # 1
    v::FT                  # 2
    w::FT                  # 3
    q_tot::FT              # 4
    q_liq::FT              # 5
    e_tot::FT              # 6
    thd::FT                # 7
    thl::FT                # 8
    thv::FT                # 9
    e_int::FT              # 10
    h_m::FT                # 11
    h_t::FT                # 12
    qt_sgs::FT             # 13
    ht_sgs::FT             # 14
    vert_eddy_mass_flx::FT # 15
    vert_eddy_u_flx::FT    # 16
    vert_eddy_v_flx::FT    # 17
    vert_eddy_qt_flx::FT   # 18       #<w'q_tot'>
    vert_qt_flx::FT        # 19            #<w q_tot>
    vert_eddy_ql_flx::FT   # 20
    vert_eddy_qv_flx::FT   # 21
    vert_eddy_thd_flx::FT  # 22
    vert_eddy_thv_flx::FT  # 23
    vert_eddy_thl_flx::FT  # 24
    uvariance::FT          # 25
    vvariance::FT          # 26
    wvariance::FT          # 27
    wskew::FT              # 28
    TKE::FT                # 29
  end
end
num_diagnostic(FT) = varsize(vars_diagnostic(FT))
diagnostic_vars(array) = Vars{vars_diagnostic(eltype(array))}(array)

function usage()
    println("""
Usage:
    julia graph_diagnostic.jl <diagnostic_file.jld2> <diagnostic_name>""")
end

function start(args::Vector{String})
    #data = load(args[1])

    # USER INPUTS:
    fn = fieldnames(vars_diagnostic(Float64))
    @show fn
    # out_vars = string.(fn[2:end])
    out_vars = string.(fn)
    @show out_vars

    data_file = first(collect(filter(x->occursin(".jld2",x) && occursin("diagnostics",x), readdir("output"))))
    data = load(joinpath("output",data_file))

    time = 0.0
#     time = 0.05

    zvertical = 1600
#     Lv0 = 2.5008e6 #See PlanetParameters.jl
#     #
#     # END USER INPUTS:
#     #

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
                            yaxis=("Altitude[m]", (0, zvertical)),
                            label=(out_vars[i]),
                            ))
    end

    f=font(11,"courier")
    time_str = string("t = ", ceil(time), " s")

    var_groups = (
                  2:4, # velocity
                  5:6, # q_tot, q_liq
                  # 7:9, # thd, thl, thv
                  )
    for i in var_groups
      all_plots = plot(each_plot[i]..., layout = (1,length(i)), titlefont=f, tickfont=f, legendfont=f, guidefont=f, title=time_str)
      plot!(size=(900,800))
      savefig(all_plots, joinpath("output",join(out_vars[i])*".png"))
    end

  return fn

end

#if length(ARGS) != 3 || !endswith(ARGS[1], ".jld2")
#    usage()
#end
results = start(ARGS)
nothing
