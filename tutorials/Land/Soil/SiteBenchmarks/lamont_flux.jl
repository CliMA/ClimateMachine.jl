using NCDatasets
using Dates
using Statistics
using DelimitedFiles
# fix these
start = DateTime(2005,06,22)
endtime = DateTime(2005,09,01)
filepath = "lamont/arms_flux/precip_rate_sfc"
ds = Dataset(filepath)
precip_rate = ds["precip_rate_sfc"][:] # mm/hr
lhf_baebbr = ds["latent_heat_flux_baebbr"][:]
times = ds["time"][:]
times_lhf = times[typeof.(lhf_baebbr) .!= Missing]
lhf = lhf_baebbr[typeof.(lhf_baebbr) .!= Missing]

times_precip = times[typeof.(precip_rate) .!= Missing]
precip_rate = precip_rate[typeof.(precip_rate) .!= Missing] ./1000 ./ 3600

lhf_qcecor = ds["latent_heat_flux_qcecor"][:]# allmmissing

