"""
            graph_diagnostic.jl
         
"""

using Plots; pyplot()
#using VegaLite
using DataFrames, FileIO
using DelimitedFiles
using LaTeXStrings

import PyPlot

gr()

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
    user_specified_timestart = -7200
    user_specified_timeend   = -9973 # set it to -1 if you want the plotter to detect and show the last time step data onl
    time_average = "y"
    isimex = "y"
   
    #
    # List the directories containing the JLD2 files to post-process:
    #
    gcloud_VM = ["dycoms"] #EDDY
    
    for gcloud in gcloud_VM

        ode_str = "Multi-rate"
        #ode_str = "LSRK14"

        dx = 40
        dy = dx
        dz = 20
        Dt = 0.59
        SGS = "Smag"
        title_string = string(ode_str, ": Dt = ", Dt, " s. ",  SGS, ". (Dh, Dv) = (", dx, ", ", dz, ") m." )
        
        #data  = load(string(clima_path, "/output/EDDY/", gcloud, "/DYCOMS-AtmosDefault-2020-03-13T16.40.19.266.jld2")) #Multi-rate SMAGORINSKY 40x20m
        data  = load(string(clima_path, "/output/EDDY/", gcloud, "/DYCOMS-AtmosDefault-2020-03-16T17.23.48.495.jld2")) #Multi-rate Smago 40x20m
                
        info_str = string(dx," x ", dz, "-",SGS,"-", ode_str)

  #=      out_vars = ["ht_sgs",
                    "qt_sgs",
                    "h_m",
                    "h_t",
                    "vert_eddy_qt_flux",
                    "q_tot",
                    "q_liq",
                    "uvariance",
                    "vvariance",
                    "wvariance",
                    "wskew",
                    "thd",
                    "thv",
                    "thl",
                    "w",
                    "vert_eddy_thv_flux",
                    "u",
                    "v",
                    "TKE"]

        out_vars_symbols = ["ht_sgs",
                            "qt_sgs",
                            "h_m",
                            "h_t",
                            "<w'q_t>",
                            "<qt>",
                            "<ql>",
                            "<u'u'>",
                            "<v'v'>",
                            "<w'w'>",
                            "<w'w'w'>",
                            "\\theta_d",
                            "\\theta_v",
                            "\\theta_l",
                            "w",
                            "<w'thv>",
                            "u",
                            "v",
                            "TKE"]
        =#
        out_vars = ["vert_eddy_qt_flux",
                    "q_tot",
                    "q_liq",
                    "uvariance",
                    "vvariance",
                    "wvariance",
                    "wskew",
                    "thv",
                    "thl",
                    "w",
                    "vert_eddy_thv_flux",
                    "u",
                    "v"]
                    
        out_vars_symbols = ["<w'q_t>",
                            "<qt>",
                            "<ql>",
                            "<u'u'>",
                            "<v'v'>",
                            "<w'w'>",
                            "<w'w'w'>",
                            "\\theta_v",
                            "\\theta_l",
                            "w",
                            "<w'thv>",
                            "u",
                            "v"]
        

        nvars = length(out_vars_symbols)
        @show(length(out_vars_symbols))
        
        zvertical = 2500
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
        www_stevens = readdlm(string(clima_path, "/output/Stevens2005Data/experimental_www_stevens2005.csv"), ',', Float64)
        ww_stevens = readdlm(string(clima_path, "/output/Stevens2005Data/experimental_ww_stevens2005.csv"), ',', Float64)
        
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

        #Allocate:
        Var = zeros(Nqk * nvertelem, nvars)

        if time_average == "yes" || time_average == "y"
            time_average_str = "Tave"
            @show time_average_str
            key = time_data #this is a string
            ntimes = 0
            for key in keys(data)
                if  parse.(Float64,key) > timestart
                    for ev in 1:nvertelem
                        for k in 1:Nqk
                            dv = diagnostic_vars(data[key][k,ev])
                            if @isdefined data2
                                dv2 = diagnostic_vars(data2[key][k,ev])
                            end
                            for ivar = 1:nvars
                                Var[k+(ev-1)*Nqk, ivar] += getproperty(dv, Symbol(out_vars[ivar]))
                            end
                            if @isdefined data2
                                for ivar = 1:nvars
                                    Var2[k+(ev-1)*Nqk, ivar] += getproperty(dv2, Symbol(out_vars[ivar]))
                                end
                            end
                        end               
                    end
                    ntimes += 1
                end #end if timestart           
            end
        else
            time_average_str = "Tinst"
            @show time_average_str
            key = time_data #this is a string
            ntimes = 1
            for ev in 1:nvertelem
                for k in 1:Nqk
                    dv = diagnostic_vars(data[key][k,ev])
                    for ivar = 1:nvars
                        Var[k+(ev-1)*Nqk, ivar] = getproperty(dv, Symbol(out_vars[ivar]))
                    end
                    if @isdefined data2
                        for ivar = 1:nvars
                            Var2[k+(ev-1)*Nqk, ivar] += getproperty(dv2, Symbol(out_vars[ivar]))
                        end
                    end
                end
            end
        end
@show "Total times steps " ntimes

f=font(24,"Times")
ivar = 1
for outvar in out_vars
    #for ivar in [1, 2, 6, 7, 8, 12, 19]
    V         = Var[:,ivar]/ntimes
    #if ivar == 6 || ivar  == 7
    if outvar == "q_tot" || outvar == "q_liq"
        V .= V*1e+3
    end

    if @isdefined data2
        V2         = Var[:,ivar]/ntimes
        #if ivar == 6 || ivar  == 7
        if outvar == "q_tot" || outvar == "q_liq"
            V2 .= V2*1e+3
        end
    end
    
    axis_name = out_vars_symbols[ivar]
    p = plot(V, Z,
             linewidth=6,
             color=[:green],
             xaxis=(axis_name),
             yaxis=("Altitude (m)", (0, zvertical)),
             label=(axis_name),
             )

    if outvar == "q_tot"
        qt_rf01 = qt_stevens[:,1]
        z_rf01  = qt_stevens[:,2]
        plot!(qt_rf01,z_rf01,seriestype=:scatter,
              markersize = 10,
              markercolor = :black,
              label=("<qt> Exp."),
              xaxis=("<qt> (g/kg)"),)
        
    #elseif ivar == 7
    elseif outvar == "q_liq"
        ql_rf01 = ql_stevens[:,1]
        z_rf01  = ql_stevens[:,2]
        plot!(ql_rf01,z_rf01,seriestype=:scatter,
              markersize = 10,
              markercolor = :black,
              label=("<ql> Exp."),
              xaxis=("<ql> (g/kg)"),)

    #elseif ivar == 8
    elseif outvar == "wvariance"
        ww_rf01 = ww_stevens[:,1]
        z_rf01  = ww_stevens[:,2]
        p_ww_exp = plot!(ww_rf01,z_rf01,seriestype=:scatter,
                         xaxis=((0, 0.75), 0:0.1:0.75),
                         markersize = 10,
                         markercolor = :black,
                         label=("<w'w'> Exp.")
                         )

        if @isdefined data2
            plot!(ww,z_rf01,seriestype=:scatter,
                         markersize = 10,
                         markercolor = :black,
                         label=("<w'w'> Exp.")
                         )
        end
        
    #elseif ivar == 9
    elseif outvar == "wskew"
        #ww   .= Var[:,8]/ntimes
        www  = Var[:,ivar]/ntimes
        #ww3  .= ww .* ww .* ww
        Sw   = www # ./sqrt.(ww3) #<w'w'w'>/√(w'w')^3
        pwww = plot(Sw, Z,
                    linewidth=5,
                    xaxis=("<w'w'w'>/√<w'w'>^3"), #, (-0.15, 0.15), -0.15:0.05:0.15),
                    #xaxis=(" <w'w'w'>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
                    yaxis=("Altitude[m]", (0, zvertical)),
                    label=("Skewness"),
                    )
        hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")
        www_rf01 = www_stevens[:,1]
        z_rf01  = www_stevens[:,2]
        p_www_exp = plot!(www_rf01,z_rf01,seriestype=:scatter,
                          markersize = 10,
                          markercolor = :black,
                          label=("<Sw experimental>")
                          )
        
    #elseif ivar == 12
    elseif outvar == "thl"
        thl_rf01 = thl_stevens[:,1]
        z_rf01  = thl_stevens[:,2]
        plot!(thl_rf01,z_rf01,seriestype=:scatter,
              markersize = 10,
              markercolor = :black,
              label=("<\\theta_l > Exp."),
              xaxis=("<\\theta_l > (K)"),)
              
    end
    
    #Plot cloud base and top as dashed lines:
    hline!( [600, 840], width=[1,1], linestyle=[:dash, :dash], color=[:black],  label="")

    #Plot
    all_plots = plot(
        p,
        layout = (1,1),
        titlefont=f,
        tickfont=f,
        legendfont=f,
        guidefont=f,
        leg=false,
        fmt = :png,
        title=title_string,
        grid=false)
    
    plot!(size=(1050,900))
    outfile_name = string(out_vars[ivar], "_", info_str,"-", ode_str,"-", time_average_str, "-from-", timestart,"-to-", ceil(timeend),"s.png")
    savefig(all_plots, joinpath(string(out_plot_dir,"/plots/"), outfile_name))
    
    ivar += 1
    
end

end
end

#if length(ARGS) != 3 || !endswith(ARGS[1], ".jld2")
#    usage()
#end
start(ARGS)
