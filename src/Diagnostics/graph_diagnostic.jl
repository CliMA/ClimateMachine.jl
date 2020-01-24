"""
    graph_diagnostic.jl
    
    #Output variables names:
     out_vars = ["ht_sgs",
                "qt_sgs",
                "h_m",
                "h_t",
                "vert_eddy_qt_flx",
                "q_tot",
                "q_liq",
                "wvariance",
                "wskew",
                "thd",
                "thv",
                "thl",
                "w",
                "uvariance",
                "vvariance",
                "vert_eddy_thv_flx",
                "u",
                "v"]
     
"""

using Plots; pyplot()
#using VegaLite
using DataFrames, FileIO
using DelimitedFiles
import PyPlot

clima_path = "/Users/simone/Work/CLIMA"
out_plot_dir = "/Users/simone/Work/CLIMA/output"
include(string(clima_path,"/src/Diagnostics/diagnostic_vars.jl"))

function usage()
    println("""
Usage:
    julia graph_diagnostic.jl <diagnostic_file.jld2> <diagnostic_name>""")
end

function start(args::Vector{String})
    #data = load(args[1])
    
    #
    # USER INPUTS:
    #
    user_specified_timestart = 0
    user_specified_timeend   = -12329.963523831071 # set it to -1 if you want the plotter to detect and show the last time step data only
    time_average = "y"
    isimex = "y"
    dh=40
    dv=20

    #
    # List the directories containing the JLD2 files to post-process:
    #
    gcloud_VM = ["yt-DYCOMS-MULTI-RATE-NEW-MOIST-THERMO"] #EDDY
        
    for gcloud in gcloud_VM

        
        SGS = "Smago"
        
        ode_str = "imex"
        
        #data = load(string(clima_path, "/output/EDDY/", gcloud, "/IMPL/diagnostics-2020-01-16T22:11:26.296.jld2")) #Dsub=0 component-wise Geostrophic forcing

        data = load(string(clima_path, "/output/EDDY/", gcloud, "/IMPL/diagnostics-2020-01-17T07:44:12.648.jld2")) # D=-3.7 
        
        
        
    info_str = string(dh,"X", dv, "-SGS-", ode_str)
        
    out_vars = ["ht_sgs",
                "qt_sgs",
                "h_m",
                "h_t",
                "vert_eddy_qt_flx",
                "q_tot",
                "q_liq",
                "wvariance",
                "wskew",
                "thd",
                "thv",
                "thl",
                "w",
                "uvariance",
                "vvariance",
                "vert_eddy_thv_flx",
                "u",
                "v"]
               
    
    zvertical = 1500
    #
    # END USER INPUTS:
    #


    #
    # Stevens et al. 2005 measurements:
    #
    qt_stevens  = readdlm(string(clima_path, "/output/Stevens2005Data/experimental_qt_stevens2005.csv"), ',', Float64)
    ql_stevens  = readdlm(string(clima_path, "/output/Stevens2005Data/experimental_ql_stevens2005.csv"), ',', Float64)
    thl_stevens = readdlm(string(clima_path, "/output/Stevens2005Data/experimental_thetal_stevens2005.csv"), ',', Float64)
    tkelower_stevens = readdlm(string(clima_path, "/output/Stevens2005Data/lower_limit_tke_time_stevens2005.csv"), ',', Float64)
    tkeupper_stevens = readdlm(string(clima_path, "/output/Stevens2005Data/upper_limit_tke_time_stevens2005.csv"), ',', Float64)
                    
    
    @show keys(data)
    println("data for $(length(data)) time steps in file")
    
    diff = 100

    times = parse.(Float64,keys(data))

    if user_specified_timestart < 0   
        timestart = minimum(times)
        @show(timestart)
    else
        timestart = user_specified_timestart
        @show(timestart)
    end
    if user_specified_timeend < 0   
        timeend = maximum(times)
        @show(timeend)
    else
        timeend = user_specified_timeend
        @show(timeend)
    end
       
    time_data = string(timeend)
    Nqk = size(data[time_data], 1)
    nvertelem = size(data[time_data], 2)
    Z = zeros(Nqk * nvertelem)
    for ev in 1:nvertelem
        for k in 1:Nqk
            dv = diagnostic_vars(data[time_data][k,ev])
            Z[k+(ev-1)*Nqk] = dv.z
        end
    end

     
    

    V1  = zeros(Nqk * nvertelem)
    V2  = zeros(Nqk * nvertelem)
    V3  = zeros(Nqk * nvertelem)
    V4  = zeros(Nqk * nvertelem)
    V5  = zeros(Nqk * nvertelem)
    V6  = zeros(Nqk * nvertelem)
    V7  = zeros(Nqk * nvertelem)
    V8  = zeros(Nqk * nvertelem)
    V9  = zeros(Nqk * nvertelem)
    V10 = zeros(Nqk * nvertelem)
    V11 = zeros(Nqk * nvertelem)
    V12 = zeros(Nqk * nvertelem)
    V13 = zeros(Nqk * nvertelem)
    V14 = zeros(Nqk * nvertelem)
    V15 = zeros(Nqk * nvertelem)
    V16 = zeros(Nqk * nvertelem)
    V17 = zeros(Nqk * nvertelem)
    V18 = zeros(Nqk * nvertelem)

    if time_average == "yes" || time_average == "y"
        time_average_str = "Tave"
        timestr = string(info_str, ". ", SGS, time_average_str, ".", ceil(timeend), " s")
        key = time_data #this is a string
        ntimes = 0
        for key in keys(data)
            if  parse.(Float64,key) > timestart
                for ev in 1:nvertelem
                    for k in 1:Nqk
                        dv = diagnostic_vars(data[key][k,ev])
                        V1[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[1]))
                        V2[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[2]))
                        V3[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[3]))
                        V4[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[4]))
                        V5[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[5]))
                        V6[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[6]))
                        V7[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[7]))
                        V8[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[8]))
                        V9[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[9]))
                        V10[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[10]))
                        V11[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[11]))
                        V12[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[12]))
                        V13[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[13]))
                        V14[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[14]))
                        V15[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[15]))
                        V16[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[16]))
                        V17[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[17]))
                        V18[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[18]))
                    end               
                end
                ntimes += 1
            end #end if timestart           
        end
    else
        time_average_str = "Tinst"
        key = time_data #this is a string
        timestr = string(info_str, ". ", SGS, ". At t= ", ceil(timeend), " s")
        ntimes = 1
        for ev in 1:nvertelem
            for k in 1:Nqk
                dv = diagnostic_vars(data[key][k,ev])
                V1[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[1]))
                V2[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[2]))
                V3[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[3]))
                V4[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[4]))
                V5[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[5]))
                V6[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[6]))
                V7[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[7]))
                V8[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[8]))
                V9[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[9]))
                V10[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[10]))
                V11[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[11]))
                V12[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[12]))
                V13[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[13]))
                V14[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[14]))
                V15[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[15]))
                V16[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[16]))
                V17[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[17]))
                V18[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[18]))
            end
        end
    end
    @show "Total times steps " ntimes

    p1 = plot(V1/ntimes, Z,
              linewidth=3,
              xaxis=("ht_sgs", (0, 250), 0:25:250),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("ht_sgs"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    p2 = plot(V2/ntimes, Z,
              linewidth=3,
              xaxis=("qt_sgs", (-5e-5, 2e-5)),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("qt_sgs"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    labels = ["h_m = e_i + gz + RmT" "h_t = e_t + RmT"]
    p3 = plot([V3/ntimes V4/ntimes], Z,
              linewidth=3,
              xaxis=("Moist and total enthalpies"), #(1.08e8, 1.28e8), 1.08e8:0.1e8:1.28e8),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")
    
    pwqt = plot(V5*1e+3/ntimes, Z,
              linewidth=3,
              xaxis=("<w' qt'> (m/s g/kg)"), #(0, 0.), 0:0.:1),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w qt>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    pqt = plot(V6*1e+3/ntimes, Z,
              linewidth=3,
              xaxis=("<qt>", (0, 12), 0:2:12),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<qt>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    qt_rf01 = qt_stevens[:,1]
    z_rf01  = qt_stevens[:,2]
    p5 = plot!(qt_rf01,z_rf01,seriestype=:scatter,
               markersize = 10,
               markercolor = :black,
               label=("<qt experimental>")
               )
##
    
    pql = plot(V7*1e+3/ntimes, Z,
              linewidth=3,
              xaxis=("<ql>"), #(-0.05, 0.5), -0.05:0.1:0.5),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<ql>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black], label="")

    ql_rf01 = ql_stevens[:,1]
    z_rf01  = ql_stevens[:,2]
    p6 = plot!(ql_rf01,z_rf01,seriestype=:scatter,
               markersize = 10,
               markercolor = :black,
               label=("<ql experimental>")
               )

    puu = plot(V14/ntimes, Z,
              linewidth=3,
              xaxis=("<u'u'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<u'u'>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    pvv = plot(V15/ntimes, Z,
              linewidth=3,
              xaxis=("<v'v'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<v'v'>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")


    pww = plot(V8/ntimes, Z,
              linewidth=3,
              xaxis=("<w'w'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w'w'>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")
  
   tke = 0.5*(V14.*V14 + V15.*V15 + V8.*V8)
   ptke = plot(tke/ntimes^2, Z,
              linewidth=3,
              xaxis=("TKE"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<TKE>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    tke_rf01 = tkelower_stevens[:,1]
    z_rf01   = tkelower_stevens[:,2]
   # ptke = plot!(tke_rf01,z_rf01,seriestype=:scatter,
   #            markersize = 10,
   #            markercolor = :black,
   #            label=("<lower limit tke experimental>"))

    
    pwww = plot(V9/ntimes, Z,
              linewidth=3,
              xaxis=("<w'w'w'>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w'w'w'>"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")
    
    data = [V10/ntimes V11/ntimes V12/ntimes]
    labels = ["θ" "θv" "θl"]
    pthl = plot(data, Z,
              linewidth=3,
              xaxis=("<θ>", (285, 310), 285:5:310),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
               )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    thl_rf01 = thl_stevens[:,1]
    z_rf01  = thl_stevens[:,2]
    pthl = plot!(thl_rf01,z_rf01,seriestype=:scatter,
               markersize = 10,
               markercolor = :black,
              label=("<thl experimental>"))

    pth = plot(data, Z,
              linewidth=3,
              xaxis=("<θ>", (285, 310), 285:5:310),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")
##

    B = 9.81*V16/290.4
    pB = plot(B/ntimes, Z,
              linewidth=3,
              xaxis=("g<w'θv>/θ_0"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("g<w'θv>/θ_0"),
               )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

  #=  vert_eddy_thl_flx = plot(V21/ntimes, Z,
              linewidth=3,
              xaxis=("<w'θ_l'>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w'θ_l'>"),
               )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")
=#
    pu = plot(V17/ntimes, Z,
              linewidth=3,
              xaxis=("u", (0, 10), 0:2:10),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("u (m/s)"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    pv = plot(V18/ntimes, Z,
              linewidth=3,
              xaxis=("v", (-10, 0), -10:2:0),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("v (m/s)"),
              )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    pw = plot(V13/ntimes, Z,
              linewidth=3,
              xaxis=("w"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("w (m/s)"),
               )
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    
    f=font(14,"courier")
    all_plots = plot(pu,  pv,  pw,   pthl,
                     pqt, pql, pwqt, pB,
                     puu, pww, pwww, ptke,
                     layout = (3,4), titlefont=f, tickfont=f, legendfont=f, guidefont=f) #, title=timestr)

    plot!(size=(2200,1200))
    outfile_name = string(info_str,"-", ode_str,"-", time_average_str, "-from-", timestart,"-to-", ceil(timeend),"s.png")
    savefig(all_plots, joinpath(string(out_plot_dir,"/plots/"), outfile_name))

   #= one_plot = plot(vert_eddy_thl_flx, titlefont=f, tickfont=f, legendfont=f, guidefont=f, title=timestr)
    outfile_name = string("eddy_THETAL_flx.", info_str,".", ode_str,".", time_average_str, ".t", ceil(timeend),"s.png")
    plot!(size=(2200,1000))
    savefig(one_plot, joinpath(string(out_plot_dir,"/plots/"), outfile_name))
=#
end
end

#if length(ARGS) != 3 || !endswith(ARGS[1], ".jld2")
#    usage()
#end
start(ARGS)
