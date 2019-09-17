using Test
using CLIMA.Data.Soundings

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

gabersek = datadep"Gabersek sounding"

@test isfile(joinpath(gabersek, "sounding_gabersek.dat"))
