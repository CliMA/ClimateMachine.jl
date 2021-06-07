include("baroclinicwave.jl")
include("../../diagnostics.jl")

using FileIO
using JLD2: @load
using PyPlot
using Printf
using PGFPlotsX
using LaTeXStrings
using LinearAlgebra
using ClimateMachine.VariableTemplates
using Polynomials

using MPI
using ClimateMachine
using ClimateMachine.DGMethods: init_ode_state
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.Mesh.Topologies:
    StackedCubedSphereTopology, cubedshellwarp, grid1d

const lonshift = 60

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

#struct FullAtmosModel <: BalanceLaw end
#vars_state(::FullAtmosModel, ::Prognostic, FT) = @vars(ρ::FT, ρu::SVector{3, FT}, ρe::FT)
#vars_state(::FullAtmosModel, ::Auxiliary, FT) = @vars(coord::SVector{3, FT}, Φ::FT, ∇Φ::SVector{3, FT})

function bw_plots(datadir="esdg_output")
  #bw_panel(datadir)
  bw_timeseries(datadir)
end

function bw_timeseries(datadir)
  times = Dict()
  pmin = Dict()
  vmax = Dict()

  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("timeseries", files)) || continue
    datafile = filter(x -> occursin("timeseries", x), files)[1]
    
    data = load(joinpath(root, datafile))
    N = data["N"]
    K = data["K"]

    times[N] = data["times"]
    pmin[N] = data["pmin"]
    vmax[N] = data["vmax"]
  end

  @pgf begin
    ppmin3 = Plot({}, Coordinates(times[3], pmin[3]))
    axis = Axis(ppmin3)
    pgfsave("pmin.pdf", axis)
  
    pvmax3 = Plot({}, Coordinates(times[3], vmax[3]))
    axis = Axis(pvmax3)
    pgfsave("vmax.pdf", axis)
  end
end

bw_diagnostic_vars(FT) = @vars(p::FT, lat::FT, lon::FT, T::FT, ω::SVector{3, FT}, ωk::FT)
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

  p = pressure(ρ, ρu, ρe, Φ)
  r = norm(coord)
  lat = 180 / π * asin(z / r)
  lon = 180 / π * atan(y, x)
  T = p / (_R_d * ρ)
  k = coord / r


  diag.p = p / 100
  diag.lat = lat
  diag.lon = lon
  diag.T = T
  diag.ωk = k' * diag.ω / 1e-5
end

function bw_spherical_coordinates!(atmos, diag::Vars, state::Vars, aux::Vars, coord)
  x, y, z = coord
  r = norm(coord)
  lat = 180 / π * asin(z / r)
  lon = 180 / π * atan(y, x)

  diag.lat = lat
  diag.lon = lon
end

function compute_vorticity(N, K, state_prognostic)
  ClimateMachine.init()
  mpicomm = MPI.COMM_WORLD
  FT = Float64
  Nq = N + 1
  _planet_radius::FT = planet_radius(param_set)
  domain_height = FT(30e3)
  vert_range = grid1d(
      _planet_radius,
      FT(_planet_radius + domain_height),
      nelem = K[2],
  )
  topology = StackedCubedSphereTopology(mpicomm, K[1], vert_range)

  grid = DiscontinuousSpectralElementGrid(
      topology,
      FloatType = FT,
      DeviceArray = Array,
      polynomialorder = N,
      meshwarp = cubedshellwarp,
  )

  vort_model = VorticityModel()
  vort_dg = DGModel(
      vort_model,
      grid,
      RusanovNumericalFlux(),
      #CentralNumericalFluxFirstOrder(),
      CentralNumericalFluxSecondOrder(),
      CentralNumericalFluxGradient(),
  )
  ω = init_ode_state(vort_dg, FT(0))

  vort_dg.state_auxiliary.data .= @view state_prognostic[:, 1:4, :]
  vort_dg(ω, ω, nothing, FT(0))

  #filter = ExponentialFilter(grid, 0, 20)
  #Filters.apply!(ω, :, grid, filter)

  ω
end


function interpolate_to_pressure_surface(state_diagnostic, aux, N, K, Nqi; psurf=850)
  FT = eltype(aux)
  Nq = N + 1

  T850 = Array{FT}(undef, Nqi ^ 2, 6 * K[1] ^ 2)
  T850 .= NaN
  
  ωk850 = Array{FT}(undef, Nqi ^ 2, 6 * K[1] ^ 2)
  ωk850 .= NaN

  for eh in 1:6*K[1]^2
    for i in 1:Nqi
      for j in 1:Nqi
        for ev in 1:K[2]
          e = ev + (eh - 1) * K[2]
          pe = MVector{Nq, FT}(undef)
          Te = MVector{Nq, FT}(undef)
          ωke = MVector{Nq, FT}(undef)
          for k in 1:Nq
            ijk = i + Nqi * (j - 1 + Nqi * (k - 1))
            pe[k] = state_diagnostic[ijk, 1, e]
            Te[k] = state_diagnostic[ijk, 4, e]
            ωke[k] = state_diagnostic[ijk, end, e]
          end
          pmin, pmax = extrema(pe)
          ξ, _ = lglpoints(FT, N)
          V = vander(Polynomial{FT}, ξ, N)
          pl = Polynomial(V \ pe)
          ptest = pl(FT(-1))

          if pmin <= psurf <= pmax
            r = filter(isreal, roots(pl - psurf))
            r = real.(r)
            r = filter(x -> -1 <= x <= 1, r)
            @assert length(r) == 1
            I = interpolationmatrix(ξ, r)
            Tr = I * Te
            ωkr = I * ωke
            ij = i + Nqi * (j - 1)
            T850[ij, eh] = Tr[1]
            ωk850[ij, eh] = ωkr[1]
          end
        end
      end
    end
  end
  return T850, ωk850
end

function bw_panel(datadir)
  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("step", files)) || continue

    ioff()
    #fig = figure(figsize=(27, 10))
    fig, axs = subplots(3, 2, figsize=(27, 20))

    for datafile in files
      data = load(joinpath(root, datafile))
      
      time = data["time"]
      day = round(Int, time / (24 * 3600))
      day ∈ (8, 10) || continue
      @show day
      @show datafile

      dim = 3
      model = data["model"]
      N = data["N"]
      K = data["K"]
      vgeo = data["vgeo"]
      state_prognostic = data["state_prognostic"]
      state_auxiliary = data["state_auxiliary"]

      #model = FullAtmosModel()

      ω = compute_vorticity(N, K, state_prognostic)

      Nq = N + 1
      Nqi = Nq
      #Nqi = 2 * Nq

      #state_prognostic = interpolate_horz(state_prognostic, vgeo, dim, N, K; Nqi)
      #state_auxiliary = interpolate_horz(state_auxiliary, vgeo, dim, N, K; Nqi)
      #vgeo = interpolate_horz(vgeo, vgeo, dim, N, K; Nqi)
      #ω = interpolate_horz(ω, vgeo, dim, N, K; Nqi)

      state_diagnostic = create_diagnostic_state(bw_diagnostic_vars, state_prognostic)

      state_diagnostic[:, 5:7, :] .= ω

      nodal_diagnostics!(state_diagnostic,
                         bw_nodal_diagnostics!,
                         bw_diagnostic_vars,
                         model,
                         state_prognostic,
                         state_auxiliary,
                         vgeo)

      state_diagnostic = interpolate_horz(state_diagnostic, vgeo, dim, N, K; Nqi)
      vgeo = interpolate_horz(vgeo, vgeo, dim, N, K; Nqi)

      nodal_diagnostics!(state_diagnostic,
                         bw_spherical_coordinates!,
                         bw_diagnostic_vars,
                         model,
                         state_prognostic,
                         state_auxiliary,
                         vgeo)
      #state_diagnostic[:, 3, :] .= -180 .+ mod.(180 .+ state_diagnostic[:, 3, :], 360)


      T850, ωk850 = interpolate_to_pressure_surface(state_diagnostic, state_auxiliary, N, K, Nqi)
      T850 = T850[:]
      ωk850 = ωk850[:]

      p_surf = @view state_diagnostic[1:Nqi^2, 1, 1:K[2]:end][:]
      lat = @view state_diagnostic[1:Nqi^2, 2, 1:K[2]:end][:]
      lon = @view state_diagnostic[1:Nqi^2, 3, 1:K[2]:end][:]
      #ωk850 = @view state_diagnostic[1:Nqi^2, end, 1:K[2]:end][:]
      @show extrema(lon)
      @show extrema(lat)
      @show extrema(p_surf)
      
      lon = @. mod(lon + 180 - lonshift, 360) - 180
      mask = (lat .> 0) .& (lon .> -lonshift)
      lon = lon[mask]
      lat = lat[mask]
      T850 = T850[mask]
      ωk850 = ωk850[mask]
      p_surf = p_surf[mask]

      if day == 8
        levels = vcat([955], [960 + 5i for i in  0:11], [1020])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=955, vcenter=990, vmax=1025)
      else
        levels = vcat([920], [930 + 10i for i in  0:9], [1030])
        norm = matplotlib.colors.TwoSlopeNorm(vmin=920, vcenter=980, vmax=1040)
      end

      dayi = day == 8 ? 1 : 2
      cmap = ColorMap("nipy_spectral").copy()
      shrinkcb=0.7

      axs[1, dayi].tricontour(lon, lat, p_surf; levels, colors=("k",))
      cset = axs[1, dayi].tricontourf(lon, lat, p_surf; levels, cmap, norm, extend="neither")
      axs[1, dayi].set_title("Surface pressure", loc="left")
      axs[1, dayi].set_title("Day $day", loc="center")
      axs[1, dayi].set_title("hPa", loc="right")
      cbar = colorbar(cset,
                      orientation="horizontal",
                      ax = axs[1, dayi],
                      ticks=levels[1+dayi:2:end-1],
                      shrink=shrinkcb)
     
      @show extrema(T850)
      levels = vcat([220], [230 + 10i for i in 0:7], [310])
      norm = matplotlib.colors.TwoSlopeNorm(vmin=220, vcenter=270, vmax=320)
      axs[2, dayi].tricontour(lon, lat, T850; levels, colors=("k",))
      cset = axs[2, dayi].tricontourf(lon, lat, T850; levels, cmap, norm, extend="neither")
      axs[2, dayi].set_title("850 hPa Temperature", loc="left")
      axs[2, dayi].set_title("Day $day", loc="center")
      axs[2, dayi].set_title("K", loc="right")
      cbar = colorbar(cset,
                      orientation="horizontal",
                      ax = axs[2, dayi],
                      ticks=levels[2:end-1],
                      shrink=shrinkcb)
     
      cmap = ColorMap("seismic").copy()
      @show extrema(ωk850)
      levels = vcat([-10], [-5 + 5i for i in 0:6], [30])
      norm = matplotlib.colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=30)
      axs[3, dayi].tricontour(lon, lat, ωk850; levels, colors=("k",))
      cset = axs[3, dayi].tricontourf(lon, lat, ωk850; levels, cmap, norm, extend="neither")
      axs[3, dayi].set_title("850 hPa Vorticity", loc="left")
      axs[3, dayi].set_title("Day $day", loc="center")
      axs[3, dayi].set_title("1e-5/s", loc="right")
      cbar = colorbar(cset,
                      orientation="horizontal",
                      ax = axs[3, dayi],
                      ticks=levels[2:end-1],
                      shrink=shrinkcb)

      xticks = [-60, -30, 0, 30, 60, 90, 120, 150, 180]
      xticklabels = ["0", "30E", "60E", "90E", "120E", "150E", "180", "150W", "120W"]
      yticks = [0, 30, 60, 90]
      yticklabels = ["0", "30N", "60N", "90N"]
      for ax in axs[:]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlim([xticks[1], xticks[end]])
        ax.set_ylim([yticks[1], yticks[end]])
        ax.set_aspect(1)
      end
    end

    plt.subplots_adjust(wspace=0.05)
    savefig(joinpath(root, "bw_panel.pdf"))
    close(fig)
  end
end
