include("gravitywave.jl")
include("../../diagnostics.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings
using Printf
using ClimateMachine.VariableTemplates

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function gw_plots(datadir=joinpath("esdg_output", "gravitywave"))
  gw_plot_convergence(datadir)
  gw_plot_tht_perturbation(datadir)
end

gw_diagnostic_vars(FT) = @vars(δT::FT, w::FT)
function gw_nodal_diagnostics!(atmos, diag::Vars, state::Vars, aux::Vars, coord)
  FT = eltype(state)
  _MSLP::FT = MSLP(param_set)
  _R_d::FT = R_d(param_set)
  _cp_d::FT = cp_d(param_set)
  
  ρ = state.ρ
  ρu = state.ρu
  ρe = state.ρe
  Φ = aux.Φ

  w = ρu[2] / ρ

  p = pressure(ρ, ρu, ρe, Φ)
  T = p / (_R_d * ρ)
  #diag.θ = T * (_MSLP / p) ^ (_R_d / _cp_d)
  diag.δT = T - aux.ref_state.T
  diag.w = w
end

function gw_plot_convergence(datadir=joinpath("esdg_output", "gravitywave"))
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
    savepath = joinpath(root, "gw_convergence_l2.pdf")
    pgfsave(savepath, axis)
  end
end

function gw_plot_tht_perturbation(datadir)
  linedata = Dict()
  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("step", files)) || continue
    
    files = sort(files)
    datafile = files[end]
    data = load(joinpath(root, datafile))

    model = data["model"]
    state_prognostic = data["state_prognostic"]
    state_exact = data["state_exact"]
    state_auxiliary = data["state_auxiliary"]

    state_diagnostic = nodal_diagnostics(gw_nodal_diagnostics!,
                                         gw_diagnostic_vars,
                                         model,
                                         state_prognostic,
                                         state_auxiliary)
    
    state_diagnostic_exact = nodal_diagnostics(gw_nodal_diagnostics!,
                                               gw_diagnostic_vars,
                                               model,
                                               state_exact,
                                               state_auxiliary)

    @show datafile
    
    dim = 2
    N = data["N"]
    K = data["K"]
    vgeo = data["vgeo"]
    x, z, δT, w = interpolate_equidistant(state_diagnostic, vgeo, dim, N, K)
    x, z, δT_exact, w_exact = interpolate_equidistant(state_diagnostic_exact, vgeo, dim, N, K)

    # convert to km
    x ./= 1e3
    z ./= 1e3
   
    ioff()
    xticks = range(0, 300, length = 7)
    fig, ax = subplots(2, 1, figsize=(14, 14))
    
    for a in ax
      a.set_xlim([xticks[1], xticks[end]])
      a.set_xticks(xticks)
      a.set_xlabel(L"x" * " [km]")
      a.set_ylabel(L"z" * " [km]")
      a.set_aspect(15)
    end

    ll = 0.0036
    sl = 0.0006
    levels = vcat(-ll:sl:-sl, sl:sl:ll)
    ax[1].set_title("T perturbation [K]")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    cset = ax[1].contourf(x', z', δT', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
    ax[1].contour(x', z', δT_exact', levels=levels, colors=("k",))
    cbar = colorbar(cset, ax = ax[1])
    
    #levels = 10
    ax[2].set_title("w [m/s]")
    #norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    cset = ax[2].contourf(x', z', w', cmap=ColorMap("PuOr"), levels=levels)
    ax[2].contour(x', z', w_exact', levels=levels, colors=("k",))
    cbar = colorbar(cset, ax = ax[2])


    tight_layout()
    savefig(joinpath(root, "gw_T_perturbation.pdf"))
    close(fig)
    
    k = findfirst(z[1, :] .> 5)
    x_k = x[:, k]
    w_k = w[:, k]
    w_exact_k = w_exact[:, k]
    linedata[(N, K...)] = (x_k, w_k, w_exact_k)
  end


  @pgf begin

    fig = @pgf GroupPlot({group_style= {group_size="1 by 2", vertical_sep="1.5cm"},
                         xmin=0,
                         xmax= 300})
    x, w, w_exact = linedata[(3, 20, 5)]
    p1 = Plot({dashed}, Coordinates(x, w))
    p2 = Plot({}, Coordinates(x, w_exact))
    push!(fig, {xlabel="x [km]",
                ylabel="w [m/s]",
                width="10cm",
                height="5cm"},
               p1, p2)
    x, w, w_exact = linedata[(3, 40, 10)]
    p1 = Plot({dashed}, Coordinates(x, w))
    p2 = Plot({}, Coordinates(x, w_exact))
    push!(fig, {xlabel="x [km]",
                ylabel="w [m/s]",
                width="10cm",
                height="5cm"},
               p1, p2)
  end
  pgfsave(joinpath(datadir, "gw_line.pdf"), fig)
end
