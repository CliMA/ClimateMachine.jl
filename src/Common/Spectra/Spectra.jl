module Spectra

export power_spectrum_1d, power_spectrum_2d, power_spectrum_3d

using FFTW
using MPI

using ..ConfigTypes
using ..Mesh.Grids
using ..MPIStateArrays

include("power_spectrum_les.jl")
include("power_spectrum_gcm.jl")

end
