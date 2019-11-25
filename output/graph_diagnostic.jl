"""
    graph_diagnostic.jl
    
    #Output variables names:

    u
    v
    w
    q_tot
    e_tot
    q_liq
    thd
    thl
    thv
    e_int
    h_m
    h_t
    qt_sgs
    ht_sgs
    
    vert_eddy_mass_flx
    vert_eddy_u_flx
    vert_eddy_v_flx
    vert_eddy_qt_flx
    vert_qt_flx
    vert_eddy_ql_flx
    vert_eddy_qv_flx
    vert_eddy_thd_flx
    vert_eddy_thv_flx
    vert_eddy_thl_flx

    # variances
    uvariance
    vvariance
    wvariance

    # skewness
    wskew
"""

using Plots; pyplot()
using DataFrames, FileIO

include("/Users/simone/Work/CLIMA/src/Diagnostics/diagnostic_vars.jl")

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
    #data = load("nov21-subsidence-AS-results/diagnostics-2019-11-21T13_36_21.865.jld2") #AS
    #data = load("nov21-mine/diagnostics-2019-11-21T19:11:36.449.jld2") #smago
    #data = load("nov22-mine/diagnostics-2019-11-22T19:10:32.422.jld2")
    #data = load("nov23-mine/diagnostics-2019-11-23T16:47:30.956.jld2")
    #data = load("nov24/diagnostics-2019-11-24T12:12:15.426.jld2")
    data = load("nov25/diagnostics-2019-11-25T14:57:58.778.jld2")

    out_vars = ["u",                 #1
                "v",                 #2
                "w",                 #3
                "q_tot",             #4
                "q_liq",             #5
                "thd",               #6
                "thl",               #7
                "thv",               #8
                "e_tot",             #9
                "e_int",             #10
                "h_m",               #11
                "h_t",               #12
                "qt_sgs",            #13
                "ht_sgs",            #14
                "vert_eddy_mass_flx",#15
                "vert_eddy_u_flx",   #16
                "vert_eddy_v_flx",   #17
                "vert_eddy_qt_flx",  #18
                "vert_qt_flx",       #19
                "vert_eddy_ql_flx",  #20
                "vert_eddy_qv_flx",  #21
                "vert_eddy_thd_flx", #22
                "vert_eddy_thv_flx", #23
                "vert_eddy_thl_flx", #24
                "uvariance",         #25
                "vvariance",         #26
                "wvariance",         #27
                "wskew"]             #28

    time = 0.0
    time = 0.05

    zvertical = 1600
    Lv0 = 2.5008e6 #See PlanetParameters.jl
    #
    # END USER INPUTS:
    #

@show keys(data)
    println("data for $(length(data)) time steps in file")
    #time_data = first(keys(data))
    
    time_data = string(time)
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
    V19 = zeros(Nqk * nvertelem)
    V20 = zeros(Nqk * nvertelem)
    V21 = zeros(Nqk * nvertelem)
    V22 = zeros(Nqk * nvertelem)
    V23 = zeros(Nqk * nvertelem)
    V24 = zeros(Nqk * nvertelem)
    V25 = zeros(Nqk * nvertelem)
    V26 = zeros(Nqk * nvertelem)
    V27 = zeros(Nqk * nvertelem)
    V28 = zeros(Nqk * nvertelem)
    V29 = zeros(Nqk * nvertelem)
    V30 = zeros(Nqk * nvertelem)
    #for key in keys(data)
     t = time_data
        for ev in 1:nvertelem
            for k in 1:Nqk
                dv = diagnostic_vars(data[t][k,ev])
#                V[k+(ev-1)*Nqk] += getproperty(dv, Symbol(args[2]))
                V1[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[1]))
                V2[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[2]))
                V3[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[3]))
                V4[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[4]))
                V5[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[5]))
                V6[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[6]))
                V7[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[7]))
                V8[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[8]))
                V9[k+(ev-1)*Nqk]  += getproperty(dv, Symbol(out_vars[9]))
                V10[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[10]))
                V11[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[11]))
                V12[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[12]))
                V13[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[13]))
                V14[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[14]))
                V15[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[15]))
                V16[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[16]))
                V17[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[17]))
                V18[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[18]))
                V19[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[19]))
                V20[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[20]))
                V21[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[21]))
                V22[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[22]))
                V23[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[23]))
                V24[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[24]))
                V25[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[25]))
                V26[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[26]))
                V27[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[27]))
                V28[k+(ev-1)*Nqk] += getproperty(dv, Symbol(out_vars[28]))
            end
        end
    #end

#=   pSGS = plot(V30, Z,
              linewidth=2,
              xaxis=("SGS"),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("SGS"),
              )

   pRi = plot(V29, Z,
              linewidth=2,
              xaxis=("Richardson"),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("Ri"),
              )
=#
   pu = plot(V1, Z,
              linewidth=2,
              xaxis=("<u>", (-0, 10), 0:1:10),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<u_g>"),
              )

     pv = plot(V2, Z,
              linewidth=2,
              xaxis=("<v>", (-10, 0), -10:1:0),                                 
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<v_g>"),
              )

     pw = plot(V3, Z,
              linewidth=2,
              xaxis=("<w>"),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w>"),
              )

    
    labels = ["θl" "θ"]
    data = [V7 V6]
    pthl = plot(data, Z,
              linewidth=2,
              xaxis=("<θl, θ>", (285, 310), 285:5:310),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
              )
#=
    labels = ["θ"]
    pth = plot(V6, Z,
              linewidth=2,
              xaxis=("<θ>", (285, 310), 285:5:310),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
              )
=#
    pqt = plot(V4*1e+3, Z,
              linewidth=2,
              xaxis=("<qt>", (0, 15), 0:3:15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<qt>"),
              )


    pql = plot(V5*1e+3, Z,
              linewidth=2,
              xaxis=("<ql>"),# (0, 15), 0:3:15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<qt>"),
              )


#    pwqt = plot(V19*1e+3, Z,
    pwqt = plot(V18*1e+3, Z,
              linewidth=2,
              xaxis=("<w' qt'>Lv_0 (W/m2)"), #(0, 1e-4), 0:0.2e-4:1e-4),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w' qt'>"),
              )

    data = [V11 V12]
    labels = ["h_m = e_i + gz + RmT" "h_t = e_t + RmT"]
    phthm = plot(data, Z,
              linewidth=2,
              xaxis=("Moist and total enthalpies"), #(1.08e8, 1.28e8), 1.08e8:0.1e8:1.28e8),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=labels,
              )

#=    puu = plot(V25, Z,
              linewidth=2,
              xaxis=("<u'u'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<u'u'>"),
              )
=#
    θ0 = 289
    B = (9.81*(V23/Lv0)/θ0)
    pB = plot(B, Z,
              linewidth=2,
              xaxis=("g<w'θv'>/θ0"), #(-0.3, ), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("g<w'θv'>/θ0"),
              )

    pww = plot(V27, Z,
              linewidth=2,
              xaxis=("<w'w'>"),# (-0.1, 0.6), -0.1:0.1:0.6),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w'w'>"),
              )    
             
    pwww = plot(V28, Z,
              linewidth=2,
              xaxis=("<w'w'w'>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("<w'w'w'>"),
              )

    pwthl = plot(V24/Lv0, Z,
                 linewidth=2,
                 xaxis=("<w'th_l'>"), #, (-0.15, 0.15), -0.15:0.05:0.15),
                 yaxis=("Altitude[m]", (0, zvertical)),
                 label=("<w'th_l'>"),
              )
#=
    TKE = 0.5*(V25.^2 .+ V26.^2 .+ V27.^2)
    ptke = plot(TKE, Z,
              linewidth=2,
              xaxis=("TKE (m2/s2"), #, (-0.15, 0.15), -0.15:0.05:0.15),
              yaxis=("Altitude[m]", (0, zvertical)),
              label=("TKE"),
              )
  =#     
    f=font(11,"courier")
    time_str = string("t = ", ceil(time), " s")
    plot(pu, pv, pw, pthl, pqt, pql, pwqt, phthm, pB, pww, pwww, pwthl, layout = (3,4), titlefont=f, tickfont=f, legendfont=f, guidefont=f, title=time_str)
    plot!(size=(900,800))

end

#if length(ARGS) != 3 || !endswith(ARGS[1], ".jld2")
#    usage()
#end
start(ARGS)
