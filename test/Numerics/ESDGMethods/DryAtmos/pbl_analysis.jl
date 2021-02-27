include("pbl_def.jl")

using FileIO
using FFTW
using Printf
using Statistics
using PGFPlotsX
using LaTeXStrings
using ClimateMachine.Mesh.Elements: interpolationmatrix, lglpoints
using ClimateMachine.VariableTemplates
using ClimateMachine.Mesh.Grids: _x1, _x2, _x3, _MH

function analyze_pbl(datadir)
  # get reference profiles
  refdir = "/home/mwarusz/data/pbl/plot/refdata"
  function getprof(file, normalize = false)
    Z = Float64[]
    V = Float64[]
    open(file, "r") do f
      lines = readlines(f)
      for line in lines
        v, z = parse.(Float64, split(line))
        push!(Z, z)
        push!(V, v)
      end
    end
    if normalize
      V .-= V[end]
      V ./= V[3]
    end
    Z, V
  end               

  wxθ_lab = getprof(joinpath(refdir, "hflux_lab.txt"), true)
  wxw_lab = getprof(joinpath(refdir, "w_lab.txt"))
  θxθ_lab = getprof(joinpath(refdir, "tht_lab.txt"))
  labdata = (wxθ = wxθ_lab, wxw = wxw_lab, θxθ = θxθ_lab)
  
  wxθ_ss = getprof(joinpath(refdir, "hflux_ss.txt"), true)
  wxw_ss = getprof(joinpath(refdir, "w_ss.txt"))
  θxθ_ss = getprof(joinpath(refdir, "tht_ss.txt"))
  ssdata = (wxθ = wxθ_ss, wxw = wxw_ss, θxθ = θxθ_ss)
  
  for (root, dir, files) in walkdir(datadir)
    files = filter(s->endswith(s, "jld2"), files)
    nfiles = length(files)
    nfiles == 0 && continue
    
    files = files[1:1]
    nfiles = length(files)

    z = nothing
    profs = nothing
    variances = nothing
    for (m, datafile) in enumerate(files)
      @show m
      data = load(joinpath(root, datafile))
      state_diagnostic = nodal_diagnostics(data)

      
      variance_pairs = ((:θ, :θ), (:w, :θ), (:w, :w))
      if m == 1
        z, profs, variances = profiles(variance_pairs, state_diagnostic, data)
      else
        _, p, v = profiles(variance_pairs, state_diagnostic, data)
        for k in keys(profs)
          profs[k] .+= p[k]
        end
        for k in keys(variances)
          variances[k] .+= v[k]
        end
      end
      
      spectra = get_spectra(joinpath(root, "spectra.pdf"),
                            state_diagnostic, data)
    end
    for k in keys(profs)
      profs[k] ./= nfiles
    end
    for k in keys(variances)
      variances[k] ./= nfiles
    end

    #write_profiles("test.txt", profile_data)
    println("finished $root")
    plot_profiles(joinpath(root, "profiles.pdf"),
                  (z, profs, variances),
                  labdata,
                  ssdata)
  end
end

pbl_diagnostic_vars(FT) = @vars(θ::FT, u::FT, v::FT, w::FT, ρ::FT)
function pbl_nodal_diagnostics!(atmos, diag::Vars, state::Vars, aux::Vars)
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
  diag.θ = T * (_MSLP / p) ^ (_R_d / _cp_d)
  diag.u = ρu[1] / ρ
  diag.v = ρu[2] / ρ
  diag.w = ρu[3] / ρ
  diag.ρ = ρ
end

function nodal_diagnostics(data)
  model = data["model"]
  state_prognostic = data["state_prognostic"]
  state_auxiliary = data["state_auxiliary"]
  N = data["N"]
  KH = data["KH"]
  KV = data["KV"]
  Nq = N + 1

  FT = eltype(state_prognostic)
  diagnostic_vars = pbl_diagnostic_vars(FT)
  num_state_diagnostic = varsize(diagnostic_vars)
  Np = Nq ^ 3
  Ne = KH ^ 2 * KV
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
       pbl_nodal_diagnostics!(
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

function profiles(variance_pairs, state_diagnostic, data; num_sample_points= (4 * data["N"]))
    FT = eltype(state_diagnostic)
    diagnostic_vars = pbl_diagnostic_vars(FT)
    num_state_diagnostic = varsize(diagnostic_vars)

    N = data["N"]
    Nq = N + 1
    Nqk = Nq
    vgeo = data["vgeo"]
    nvertelem = data["KV"]
    nhorzelem = data["KH"] ^ 2

    scaling = zeros(FT, Nqk, nvertelem) 
    z = zeros(FT, Nqk, nvertelem) 
    profs = zeros(FT, Nqk, nvertelem, num_state_diagnostic)
    
    _ρ = varsindex(diagnostic_vars, :ρ)[1]

    for ev in 1:nvertelem
        for eh in 1:nhorzelem
            e = ev + (eh - 1) * nvertelem
            for i in 1:Nq
                for j in 1:Nq
                    for k in 1:Nqk
                        ijk = i + Nq * ((j - 1) + Nqk * (k - 1))
                        z[k, ev] = vgeo[ijk, _x3, e]
                        ρ = state_diagnostic[ijk, _ρ, e] 
                        scaling[k, ev] += ρ * vgeo[ijk, _MH, e]
                        for var in fieldnames(diagnostic_vars)
                          s = varsindex(diagnostic_vars, var)
                          profs[k, ev, s] += ρ * state_diagnostic[ijk, s, e] * vgeo[ijk, _MH, e]
                        end
                    end
                end
            end
        end
    end
    
    for ev in 1:nvertelem
      for k in 1:Nqk
        for s in 1:num_state_diagnostic
          profs[k, ev, s] /= scaling[k, ev]
        end
      end
    end
    
    num_state_variance = length(variance_pairs)
    variances = zeros(FT, Nqk, nvertelem, num_state_variance)
    for ev in 1:nvertelem
        for eh in 1:nhorzelem
            e = ev + (eh - 1) * nvertelem
            for i in 1:Nq
                for j in 1:Nq
                    for k in 1:Nqk
                        ijk = i + Nq * ((j - 1) + Nqk * (k - 1))
                        ρ = state_diagnostic[ijk, _ρ, e] 
                        for (v, (var1, var2)) in enumerate(variance_pairs)
                          s1 = varsindex(diagnostic_vars, var1)[1]
                          s2 = varsindex(diagnostic_vars, var2)[1]
                          dv1 = state_diagnostic[ijk, s1, e] - profs[k, ev, s1]
                          dv2 = state_diagnostic[ijk, s2, e] - profs[k, ev, s2]
                          variances[k, ev, v] += ρ * dv1 * dv2 * vgeo[ijk, _MH, e]
                        end
                    end
                end
            end
        end
    end
    
    for ev in 1:nvertelem
      for k in 1:Nqk
        for s in 1:num_state_variance
          variances[k, ev, s] /= scaling[k, ev]
        end
      end
    end
  
    # interpolate profiles and variances to a fine uniform vertical grid
    ξsrc, _ = lglpoints(FT, N)
    ξdst = range(FT(-1); stop = FT(1), length = num_sample_points)
    I = interpolationmatrix(ξsrc, ξdst)

    z = I * z
    profs = ntuple(s -> I * (@view profs[:, :, s]), size(profs, 3))
    variances = ntuple(s -> I * (@view variances[:, :, s]), size(variances, 3))

    profs = (; zip(fieldnames(diagnostic_vars), profs)...)
    variances = (; zip((Symbol(v1, :x, v2) for (v1, v2) in variance_pairs),
                   variances)...)
    z, profs, variances
end

function write_profiles(file, profile_data)
  z, profs, variances = profile_data
  s = @sprintf "z θ w θxθ wxθ, wxw\n"
  for k in 1:length(profs.θ)
    s *= @sprintf("%.16e %.16e %.16e %.16e %.16e %.16e\n",
                  z[k],
                  profs.θ[k],
                  profs.w[k],
                  variances.θxθ[k],
                  variances.wxθ[k],
                  variances.wxw[k])
  end
  open(file, "w") do f
    write(f, s)
  end
end

function plot_profiles(name, profile_data, labdata, ssdata)
  z, profs, variances = profile_data

  # TODO: do not hardcode this values !
  g = 10.0
  tht_ref = 300.0
  H0 = 0.01

  z = z[:]
  hflux = variances.wxθ[:]
  var_w = variances.wxw[:]
  var_tht = variances.θxθ[:]

  _, kmin = findmin(hflux)
  zi = z[kmin]

  w_scale = (g / tht_ref * zi * H0) ^ (1 / 3)
  tht_scale = H0 / w_scale
  t_scale = zi / w_scale

  z ./= zi
  hflux ./= H0
  var_tht ./= tht_scale ^ 2
  var_w ./= w_scale ^ 2

  header = "zi = $zi, w_scale = $w_scale, tht_scale = $tht_scale, t_scale = $t_scale"
  println(header)

  axis = @pgf GroupPlot({group_style= {group_size="3 by 1",
                                       y_descriptions_at="edge left",
                                      },
                         ymin=0.0,
                         ymax=1.3,
                         width="6cm",
                         height="10cm",
                        })
  @pgf push!(
    axis,
    {xmin=-0.5, xmax=1.5, title=L"\langle \theta' w' \rangle / H_0"},
    Plot({}, Coordinates(hflux, z)),
    Plot({only_marks, mark="*", color="red"}, Coordinates(labdata.wxθ[2], labdata.wxθ[1])),
    Plot({only_marks, mark="x", color="blue"}, Coordinates(ssdata.wxθ[2], ssdata.wxθ[1]))
  )
  @pgf push!(
    axis,
    {xmin=0, xmax=32, title=L"\langle \theta' \theta' \rangle / T_*^2"},
    Plot({}, Coordinates(var_tht, z)),
    Plot({only_marks, mark="*", color="red"}, Coordinates(labdata.θxθ[2], labdata.θxθ[1])),
    Plot({only_marks, mark="x", color="blue"}, Coordinates(ssdata.θxθ[2], ssdata.θxθ[1]))
  )
  @pgf push!(
    axis,
    {xmin=0, xmax=0.5,title=L"\langle w' w' \rangle / w_*^2"},
    Plot({}, Coordinates(var_w, z)),
    Plot({only_marks, mark="*", color="red"}, Coordinates(labdata.wxw[2], labdata.wxw[1])),
    Plot({only_marks, mark="x", color="blue"}, Coordinates(ssdata.wxw[2], ssdata.wxw[1]))
  )

  pgfsave(name, axis)
end

function get_spectra(name, state_diagnostic, data)
    N = data["N"]
    KH = data["KH"]
    KV = data["KV"]
    vgeo = data["vgeo"]
    FT = eltype(vgeo)

    Ne = KH ^ 2 * KV
    Nq = N + 1
    Np = Nq ^ 3
    # interpolate to an equidistant grid with the same number of DOFs
    # do not include interfaces
    ξsrc, _ = lglpoints(FT, N)
    dx = 2 / Nq
    ξdst = [-1 + (j - 1 / 2) * dx for j in 1:Nq]
    #@show ξsrc, ξdst
    I1d = interpolationmatrix(ξsrc, ξdst)
    I = kron(I1d, I1d, I1d)
    
    S = Array{FT}(undef, Nq * KH, Nq * KH, Nq * KV)
    fill!(S, NaN)
    
    diagnostic_vars = pbl_diagnostic_vars(FT)
    _u, _v, _w = varsindices(diagnostic_vars, (:u, :v, :w))

    dx1 = FT(-1)
    dx2 = FT(-2)
    dx3 = FT(-3)
    @views for e in 1:Ne
      x1i = I * vgeo[:, _x1, e]
      x2i = I * vgeo[:, _x2, e]
      x3i = I * vgeo[:, _x3, e]

      dx1 = x1i[2] - x1i[1]
      dx2 = x2i[Nq + 1] - x2i[1]
      dx3 = x3i[Nq ^ 2 + 1] - x3i[1]

      i1 = round.(Int, (x1i .+ dx1 / 2) ./ dx1)
      i2 = round.(Int, (x2i .+ dx2 / 2) ./ dx2)
      i3 = round.(Int, (x3i .+ dx3 / 2) ./ dx3)

      x1i = I * vgeo[:, _x1, e]
      x2i = I * vgeo[:, _x2, e]
      x3i = I * vgeo[:, _x3, e]
      
      ui = I * state_diagnostic[:, _u, e]
      vi = I * state_diagnostic[:, _v, e]
      wi = I * state_diagnostic[:, _w, e]

      C = CartesianIndex.(i1, i2, i3)
      for ijk in 1:Np
        S[C[ijk]] = wi[ijk]
      end
    end

    spectrum_lev = round(Int, 673.3333333333333 / dx3)
    @show spectrum_lev
    S = @view S[:, :, spectrum_lev]

    N1d = Nq * KH
    @show div(N1d, 2) + 1

    Skx = rfft(S, (1,))
    Skx = abs.(Skx) .^ 2 ./ 2
    Skx = mean(Skx, dims=(2,))[:]

    Sky = rfft(S, (2,))
    Sky = abs.(Sky) .^ 2 ./ 2
    Sky = mean(Sky, dims=(1,))[:]

    k = rfftfreq(N1d)
    Sk = (Skx + Sky) / 2
    @pgf begin 
      plot = Plot({}, Coordinates(k, Sk))
      plot53 = Plot({color="red", dashed}, Coordinates(k, 0.05 * k .^ (-5 / 3)))
      axis = LogLogAxis({width="10cm",
                         height="10cm",
                         xlabel="k",
                         ylabel=L"E_w(k)",
                         ymax = maximum(Sk)+1},
                        plot,
                        plot53,
                        L"\node[] at (4e-1,0.4) {-5/3};")
      pgfsave(name, axis)
    end
end
