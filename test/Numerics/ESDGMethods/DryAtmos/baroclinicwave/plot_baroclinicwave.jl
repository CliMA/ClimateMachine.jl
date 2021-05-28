include("baroclinicwave.jl")
include("../../diagnostics.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings
using LinearAlgebra
using ClimateMachine.VariableTemplates

const lonshift = 60

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

bw_diagnostic_vars(FT) = @vars(p::FT, lat::FT, lon::FT)
function bw_nodal_diagnostics!(atmos, diag::Vars, state::Vars, aux::Vars, coord)
  FT = eltype(state)
  _MSLP::FT = MSLP(param_set)
  _R_d::FT = R_d(param_set)
  _cp_d::FT = cp_d(param_set)

  x, y, z = coord
  
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ

  p = pressure(ρ, ρu, ρe, Φ) / 100
  r = norm(coord)
  lat = 180 / π * asin(z / r)
  lon = 180 / π * atan(y, x)
  lon = mod(lon + 180 - lonshift, 360) - 180
  diag.p = p
  diag.lat = lat
  diag.lon = lon
end

function bw_plot_surface_pressure(datadir)
  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("step", files)) || continue

    files = sort(files)
    datafile = files[end-7]
    data = load(joinpath(root, datafile))

    @show datafile

    dim = 3
    model = data["model"]
    N = data["N"]
    K = data["K"]
    vgeo = data["vgeo"]
    state_prognostic = data["state_prognostic"]
    state_auxiliary = data["state_auxiliary"]
    model = data["model"]
    time = data["time"]

    day = round(Int, time / (24 * 3600))
    @show day

    state_diagnostic = nodal_diagnostics(bw_nodal_diagnostics!, bw_diagnostic_vars,
                                         model, state_prognostic, state_auxiliary, vgeo)

    #x, z, δθ = interpolate_equidistant(state_diagnostic, vgeo, dim, N, K)
    
    Nq = N + 1


    p_surf = @view state_diagnostic[1:Nq^2, 1, 1:K[2]:end][:]
    lat = @view state_diagnostic[1:Nq^2, 2, 1:K[2]:end][:]
    lon = @view state_diagnostic[1:Nq^2, 3, 1:K[2]:end][:]
    
    mask = (lat .> 0) .& (lon .> -lonshift)
    lon = lon[mask]
    lat = lat[mask]
    p_surf = p_surf[mask]
   
    ioff()
    fig = figure(figsize=(14, 12))
    ax = gca()

    if day == 8
      levels = [960 + 5i for i in  0:11]
    else
      levels = [920 + 10i for i in  0:10]
    end

    ax.tricontour(lon, lat, p_surf; levels)
    ax.tricontourf(lon, lat, p_surf; levels)
    savefig(joinpath(root, "bw_test.pdf"))


    #levels = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    #fig = figure(figsize=(14, 12))
    #ax = gca()
    #xticks = range(0, 2000, length = 5)
    #ax.set_title("Potential temperature perturbation [K]")
    #ax.set_xlim([xticks[1], xticks[end]])
    #ax.set_ylim([xticks[1], xticks[end]])
    #ax.set_xticks(xticks)
    #ax.set_yticks(xticks)
    #ax.set_xlabel(L"x" * " [m]")
    #ax.set_ylabel(L"z" * " [m]")
    #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    #cset = ax.contourf(x', z', δθ', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
    #ax.contour(x', z', δθ', levels=levels, colors=("k",))
    #ax.set_aspect(1)
    #cbar = colorbar(cset)
    #tight_layout()
    #savefig(joinpath(root, "rtb_tht_perturbation.pdf"))
  end
end
