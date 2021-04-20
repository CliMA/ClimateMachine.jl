include("gravitywave.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings
using Printf
using ClimateMachine.Mesh.Elements: interpolationmatrix, lglpoints
using ClimateMachine.Mesh.Grids: _x1, _x2, _x3, _MH
using ClimateMachine.VariableTemplates

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

const SMALL_SIZE = 20
const MEDIUM_SIZE = 24
const BIGGER_SIZE = 32

rcParams["font.size"] = SMALL_SIZE 
rcParams["xtick.labelsize"] = SMALL_SIZE
rcParams["ytick.labelsize"] = SMALL_SIZE
rcParams["legend.fontsize"] = SMALL_SIZE
rcParams["figure.titlesize"] = BIGGER_SIZE
rcParams["axes.titlepad"] = 10
rcParams["axes.labelpad"] = 10

function plot_convergence(datadir=joinpath("esdg_output", "gravitywave"))
  for (root, dir, files) in walkdir(datadir)

    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("convergence", files)) || continue
    
    datafile = files[end]
    @show datafile

    @load joinpath(root, datafile) convergence_data

    @show convergence_data[2].l2_errors
    @show convergence_data[3].l2_errors

    plotsetup = @pgf {
              xlabel = "Δx [km]",
              grid = "major",
              ylabel = L"L_{2}" * " error",
              xmode = "log",
              ymode = "log",
              xticklabel="{\\pgfmathparse{exp(\\tick)/1000}\\pgfmathprintnumber[fixed,precision=3]{\\pgfmathresult}}",
              #xmax = 1,
              xtick = convergence_data[2].avg_dx,
              #ymin = 10. ^ -10 / 5,
              #ymax = 5,
              #ytick = 10. .^ -(0:2:10),
              legend_pos="south east",
            }
    axis = Axis(plotsetup)
    labels = []
    for N in keys(convergence_data)
      dxs = convergence_data[N].avg_dx
      l2s = convergence_data[N].l2_errors
      l2rate = convergence_data[N].l2_rates
      linfs = convergence_data[N].linf_errors
      linfrate = convergence_data[N].linf_rates
      coords = Coordinates(dxs, l2s)
      @pgf plot = PlotInc({}, coords)
      push!(axis, plot)
      push!(labels, "N$N " * @sprintf("(%.2f)", l2rate[end]))
    end
    @pgf legend = Legend(labels)
    push!(axis, legend)
    savepath = joinpath(root, "convergence_l2.pdf")
    pgfsave(savepath, axis)
  end
end

function plot_tht_perturbation(datadir=joinpath("esdg_output", "gravitywave"))
  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("gw", files)) || continue
    
    files = sort(files)
    datafile = files[end]
    data = load(joinpath(root, datafile))

    state_prognostic = data["state_prognostic"]
    state_auxiliary = data["state_auxiliary"]

    state_diagnostic = nodal_diagnostics(data)

    @show datafile

    # interpolate to an equidistant grid with the same number of DOFs
    # do not include interfaces
    N = data["N"]
    Nq = N + 1
    Np = Nq ^ 2
    vgeo = data["vgeo"]
    KH = data["Ne"][1]
    KV = data["Ne"][2]
    Ne = KH * KV
    FT = eltype(state_prognostic)
    ξsrc, _ = lglpoints(FT, N)

    Nqi = 4 * Nq
    Npi = Nqi ^ 2
    dxi = 2 / Nqi
    ξdst = [-1 + (j - 1 / 2) * dxi for j in 1:Nqi]
    I1d = interpolationmatrix(ξsrc, ξdst)
    I = kron(I1d, I1d)
    
    δθ = Array{FT}(undef, Nqi * KH, Nqi * KV)
    x = similar(δθ)
    z = similar(δθ)
    fill!(δθ, NaN)

    @show size(vgeo)

    dx1 = FT(-1)
    dx2 = FT(-2)
    #dx3 = FT(-3)
    @views for e in 1:Ne
      x1i = I * vgeo[:, _x1, e]
      x2i = I * vgeo[:, _x2, e]

      #@show x1i[:, 1]

      dx1 = x1i[2] - x1i[1]
      dx2 = x2i[Nqi + 1] - x2i[1]
      #dx3 = x3i[Nq ^ 2 + 1] - x3i[1]

      i1 = round.(Int, (x1i .+ dx1 / 2) ./ dx1)
      i2 = round.(Int, (x2i .+ dx2 / 2) ./ dx2)
      #@show extrema(i1)
      #@show extrema(i2)
      #i3 = round.(Int, (x3i .+ dx3 / 2) ./ dx3)
      
      ρi = I * (state_diagnostic[:, 1, e] .- state_auxiliary[:, 5, e])

      C = CartesianIndex.(i1, i2)
      for ijk in 1:Npi
        δθ[C[ijk]] = ρi[ijk]
        x[C[ijk]] = x1i[ijk]
        z[C[ijk]] = x2i[ijk]
      end
    end
    @show extrema(δθ)
   
    ioff()
    levels = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] .* 1e-5
    fig = figure(figsize=(14, 8))
    ax = gca()
    xticks = range(0, 300e3, length = 5)
    ax.set_title("Potential temperature perturbation [K]")
    ax.set_xlim([xticks[1], xticks[end]])
    #ax.set_ylim([xticks[1], xticks[end]])
    ax.set_xticks(xticks)
    #ax.set_yticks(xticks)
    ax.set_xlabel(L"x" * " [m]")
    ax.set_ylabel(L"z" * " [m]")
    #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    #cset = ax.contourf(x', z', δθ', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
    #levels = 10
    cset = ax.contourf(x', z', δθ', cmap=ColorMap("PuOr"), levels=levels)
    ax.contour(x', z', δθ', levels=levels, colors=("k",))
    ax.set_aspect(15)
    cbar = colorbar(cset)
    tight_layout()
    savefig(joinpath(root, "tht_perturbation.pdf"))
    close(fig)
  end
end

gravitywave_diagnostic_vars(FT) = @vars(T::FT)
function gravitywave_nodal_diagnostics!(atmos, diag::Vars, state::Vars, aux::Vars)
  FT = eltype(state)
  _MSLP::FT = MSLP(param_set)
  _R_d::FT = R_d(param_set)
  _cp_d::FT = cp_d(param_set)
  
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ

  p = pressure(ρ, ρu, ρe, Φ)
  T = p / (_R_d * ρ)
  #diag.θ = T * (_MSLP / p) ^ (_R_d / _cp_d)
  diag.T = T
end

function nodal_diagnostics(data)
  model = data["model"]
  state_prognostic = data["state_prognostic"]
  state_auxiliary = data["state_auxiliary"]
  N = data["N"]
  KH = data["Ne"][1]
  KV = data["Ne"][2]
  Nq = N + 1

  FT = eltype(state_prognostic)
  diagnostic_vars = gravitywave_diagnostic_vars(FT)
  num_state_diagnostic = varsize(diagnostic_vars)
  Np = Nq ^ 2
  Ne = KH * KV
  state_diagnostic = similar(state_prognostic,
                             (Np, num_state_diagnostic, Ne))

  num_state_prognostic = number_states(model, Prognostic())
  num_state_auxiliary = number_states(model, Auxiliary())

  local_state_diagnostic = MArray{Tuple{num_state_diagnostic}, FT}(undef)
  local_state_prognostic = MArray{Tuple{num_state_prognostic}, FT}(undef)
  local_state_auxiliary = MArray{Tuple{num_state_auxiliary}, FT}(undef)

  @inbounds @views for e in 1:Ne
    for ijk in 1:Np
       local_state_prognostic .= state_prognostic[ijk, :, e]
       local_state_auxiliary .= state_auxiliary[ijk, :, e]
       gravitywave_nodal_diagnostics!(
           model,
           Vars{diagnostic_vars}(
               local_state_diagnostic,
           ),
           Vars{vars_state(model, Prognostic(), FT)}(
               local_state_prognostic,
           ),
           Vars{vars_state(model, Auxiliary(), FT)}(
               local_state_auxiliary,
           ),
       )
       state_diagnostic[ijk, :, e] .= local_state_diagnostic
    end
  end
  
  state_diagnostic
end
