include("risingbubble.jl")
include("../../diagnostics.jl")

using FileIO
using JLD2: @load
using PyPlot
using PGFPlotsX
using LaTeXStrings
using ClimateMachine.VariableTemplates

rcParams!(PyPlot.PyDict(PyPlot.matplotlib."rcParams"))

function rtb_plots(datadir=joinpath("esdg_output", "risingbubble"))
  rtb_plot_entropy_residual(datadir)
  rtb_plot_tht_perturbation(datadir)
end

rtb_diagnostic_vars(FT) = @vars(δθ::FT)
function rtb_nodal_diagnostics!(atmos, diag::Vars, state::Vars, aux::Vars)
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
  θ_ref = aux.ref_state.T * (_MSLP / aux.ref_state.p) ^ (_R_d / _cp_d) 
  diag.δθ = T * (_MSLP / p) ^ (_R_d / _cp_d) - θ_ref
end

function rtb_plot_entropy_residual(datadir)
  odesolver = "lsrk"
  ecpath = joinpath(datadir,
                    odesolver,
                    "EntropyConservative",
                    "4",
                    "10x10",
                    "rtb_entropy_residual.jld2")
  @load ecpath dη_timeseries 
  t_ec = first.(dη_timeseries)
  dη_ec = last.(dη_timeseries)
  
  matrixpath = joinpath(datadir,
                    odesolver,
                    "MatrixFlux",
                    "4",
                    "10x10",
                    "rtb_entropy_residual.jld2")
  @load matrixpath dη_timeseries 
  t_matrix = first.(dη_timeseries)
  dη_matrix = last.(dη_timeseries)

  t_ec = t_ec[1:10:end]
  dη_ec = dη_ec[1:10:end]
  t_matrix = t_matrix[1:10:end]
  dη_matrix = dη_matrix[1:10:end]

  @pgf begin
    plot300 = Plot({no_marks, dashed}, Coordinates([300, 300], [0, -1e-8]))
    plot_ec = Plot({mark="o", color="red"}, Coordinates(t_ec, dη_ec))
    plot_matrix = Plot({mark="x", color="blue"}, Coordinates(t_matrix, dη_matrix))
    legend = Legend("Entropy conservative flux", "Matrix dissipation flux")
    axis = Axis({
                 ylabel=L"(\eta - \eta_0) / |\eta_0|",
                 xlabel="time [s]",
                 legend_pos="south west",
                },
                L"\node[] at (320,-0.5e-8) {vanilla DGSEM};",
                L"\node[] at (270,-0.6e-8) {breaks here};",
                plot_ec,
                plot_matrix,
                plot300,
               legend)
    pgfsave(joinpath(datadir, odesolver, "rtb_entropy.pdf"), axis)
  end
end

function rtb_plot_tht_perturbation(datadir)
  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue

    any(occursin.("step", files)) || continue

    files = sort(files)
    datafile = files[end]
    data = load(joinpath(root, datafile))

    @show files
    @show datafile

    dim = 2
    model = data["model"]
    N = data["N"]
    K = data["K"]
    vgeo = data["vgeo"]
    state_prognostic = data["state_prognostic"]
    state_auxiliary = data["state_auxiliary"]

    state_diagnostic = nodal_diagnostics(rtb_nodal_diagnostics!, rtb_diagnostic_vars,
                                         model, state_prognostic, state_auxiliary)

    x, z, δθ = interpolate_equidistant(state_diagnostic, vgeo, dim, N, K)
   
    ioff()
    levels = [-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    fig = figure(figsize=(14, 12))
    ax = gca()
    xticks = range(0, 2000, length = 5)
    ax.set_title("Potential temperature perturbation [K]")
    ax.set_xlim([xticks[1], xticks[end]])
    ax.set_ylim([xticks[1], xticks[end]])
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xlabel(L"x" * " [m]")
    ax.set_ylabel(L"z" * " [m]")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=levels[1], vcenter=0, vmax=levels[end])
    cset = ax.contourf(x', z', δθ', cmap=ColorMap("PuOr"), levels=levels, norm=norm)
    ax.contour(x', z', δθ', levels=levels, colors=("k",))
    ax.set_aspect(1)
    cbar = colorbar(cset)
    tight_layout()
    savefig(joinpath(root, "rtb_tht_perturbation.pdf"))
  end
end
