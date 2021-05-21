using NCDatasets
using Dates
using Statistics
using DelimitedFiles
# fix these
start = DateTime(2016,04,01)
endtime = DateTime(2016, 07,01)
filepath = "data/lamont/arms_flux/sgparmbeatmC1.c1.20160101.003000.nc"
ds = Dataset(filepath)
precip_rate = ds["precip_rate_sfc"][:] # mm/hr
lhf_baebbr = ds["latent_heat_flux_baebbr"][:]
times = ds["time"][:]
times_lhf = times[typeof.(lhf_baebbr) .!= Missing]
lhf = lhf_baebbr[typeof.(lhf_baebbr) .!= Missing]

times_precip = times[typeof.(precip_rate) .!= Missing]
precip_rate = precip_rate[typeof.(precip_rate) .!= Missing] ./1000 ./ 3600
Lv = 2.5008e6
ρ = 1e3
evap_rate = lhf ./ Lv ./ρ


lhf_qcecor = ds["latent_heat_flux_qcecor"][:]# allmmissing

