using NCDatasets
using Dates
using Statistics
using DelimitedFiles
using Plots
using NCDatasets
using Dates
using Plots.PlotMeasures
# fix these
start = DateTime(2016,04,01,0,30,0)
endtime = DateTime(2016, 07,01)
filepath = "data/lamont/arms_flux/sgparmbeatmC1.c1.20160101.003000.nc"
ds = Dataset(filepath)
times = ds["time"][:]
p = ((times .<=endtime) .+ (times .>= start)) .== 2
precip_rate = ds["precip_rate_sfc"][p] # mm/hr
lhf_baebbr = ds["latent_heat_flux_baebbr"][p]
times = ds["time"][p]


keep = ((typeof.(lhf_baebbr) .!= Missing) .+ (typeof.(precip_rate) .!= Missing)) .== 2
times_lhf = times[keep]
lhf = lhf_baebbr[keep]
precip_rate = precip_rate[keep]
Lv = 2.5008e6
ρ = 1e3
evap_rate = lhf ./ Lv ./ρ .*1000 .* 3600
foo = (times_lhf .-times_lhf[times_lhf .== start])./1000
seconds = [k.value for k in foo]
plot(1:1:2153, evap_rate, color = :red, xticks = :none, ylabel = "Evap (mm/hr)", yticks = ([0,0.4,0.8,1.2,1.6],["0","0.4","0.8","1.2","1.6"]), label = "", ylim = [0,1.6], right_margin = 50px, bottom_margin = 50px)
plot!(twinx(), -precip_rate, label = "", ylabel = "Precip (mm/hr)", yticks = ([0,-5,-10,-15,-20], ["0","5","10","15","20"]), ylim = [-20,0], right_margin = 50px)

plot!(xticks = ([1, 705, 1438, 2153], ["2016-4-1", "2016-5-1", "2016-6-1", "2016-7-1"]))
filepath = joinpath(@__DIR__, "data/lamont/arms_flux/precip_evap.png")
savefig(filepath)
