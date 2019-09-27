adv_eq_dir  = joinpath(output_root,"AdvectionEquation")

print_norms = false

@testset "Linear advection, ∂_t ϕ + ∇•(cϕ) = 0 ∈ ∂Ω, ϕ(t=0) = Gaussian(σ, μ), ConservativeForm, explicit Euler" begin
  dd = DomainDecomp(gm=1)
  dss = DomainSubSet(gm=true)
  σ, μ = .1, 0.5
  δz = 0.2
  Triangle(z) = μ-δz < z < μ+δz ? ( μ > z ? (z-(μ-δz))/δz : ((μ+δz)-z)/δz) : 0.0
  Square(z) = μ-δz < z < μ+δz ? 1 : 0.0
  Gaussian(z) = exp(-1/2*((z-μ)/σ)^2)
  for n_elems_real in (64, 128)
    grid = Grid(0.0, 1.0, n_elems_real)
    domain_range = over_elems_real(grid)
    x = [grid.zc[k] for k in domain_range]
    unknowns = ( (:ϕ, dss), )
    vars = ( (:ϕ_initial, dss), (:ϕ_error, dss), (:ϕ_analytic, dss), )
    q = StateVec(unknowns, grid, dd)
    tmp = StateVec(vars, grid, dd)
    rhs = deepcopy(q)
    CFL = 0.1
    Δt = CFL*grid.Δz
    T = 0.25
    maxiter = Int(T/Δt)
    for scheme in (
                   UpwindAdvective(),
                   UpwindCollocated(),
                   CenteredUnstable(),
                   )
      for distribution in (Triangle,
                           Square,
                           Gaussian)
        for wave_speed in (-1, 1)

          scheme_name = replace(string(scheme), "()"=>"")
          distribution_name = joinpath("LinearAdvection",string(distribution))
          directory = joinpath(adv_eq_dir, distribution_name, scheme_name)
          export_plots && mkpath(directory)
          print_norms && print("\n", directory, ", ", n_elems_real, ", ", wave_speed, ", ")

          for k in over_elems_real(grid)
            tmp[:ϕ_initial, k] = distribution(grid.zc[k])
            q[:ϕ, k] = tmp[:ϕ_initial, k]
          end
          amax_w = max([max(tmp[:ϕ_initial, k]) for k in over_elems_real(grid)]...)
          for i in 1:maxiter
            for k in over_elems_real(grid)
              ϕ = q[:ϕ, Cut(k)]
              ϕ_dual = q[:ϕ, Dual(k)]
              w = [wave_speed, wave_speed, wave_speed]
              w_dual = [wave_speed, wave_speed]
              rhs[:ϕ, k] = - advect_old(ϕ, ϕ_dual, w, w_dual, grid, scheme, Δt)
            end
            for k in over_elems(grid)
              q[:ϕ, k] += Δt*rhs[:ϕ, k]
            end
            apply_Dirichlet!(q, :ϕ, grid, 0.0, Zmax())
            apply_Dirichlet!(q, :ϕ, grid, 0.0, Zmin())
          end
          sol_analtyic = [distribution(grid.zc[k] - wave_speed*T) for k in over_elems(grid)]
          sol_error = [sol_analtyic[k] - q[:ϕ, k] for k in over_elems(grid)]
          L2_norm = sum(sol_error.^2)/length(sol_error)

          if !(scheme==CenteredUnstable())
            @test all(abs.(sol_error) .< 100*grid.Δz)
            @test all(L2_norm < 100*grid.Δz)
          end
          for k in over_elems(grid)
            tmp[:ϕ_error, k] = sol_error[k]
            tmp[:ϕ_analytic, k] = sol_analtyic[k]
          end

          print_norms && print("L2_norm(err) = ", L2_norm)

          name = string(n_elems_real)
          markershape = wave_speed==-1 ? :dtriangle : :utriangle
          if export_plots
            if wave_speed==-1
              y = [tmp[:ϕ_initial , k] for k in domain_range]; plot(y, x, label="initial",
                markercolor="blue", linecolor="blue", markershapes=markershape, markersize=2, legend=:topleft)
              y = [  q[:ϕ         , k] for k in domain_range]; plot!(y, x, label="numerical",
                markercolor="black", linecolor="black", markershapes=markershape, markersize=2, legend=:topleft)
              y = [tmp[:ϕ_error   , k] for k in domain_range]; plot!(y, x, label="error",
                markercolor="red", linecolor="red", markershapes=markershape, markersize=2, legend=:topleft)
              y = [tmp[:ϕ_analytic, k] for k in domain_range]; plot!(y, x, label="analytic",
                markercolor="green", linecolor="green", markershapes=markershape, markersize=2, legend=:topleft)
            else
              y = [  q[:ϕ         , k] for k in domain_range]; plot!(y, x, label="",
                markercolor="black", linecolor="black", markershapes=markershape, markersize=2, legend=:topleft)
              y = [tmp[:ϕ_error   , k] for k in domain_range]; plot!(y, x, label="",
                markercolor="red", linecolor="red", markershapes=markershape, markersize=2, legend=:topleft)
              y = [tmp[:ϕ_analytic, k] for k in domain_range]; plot!(y, x, label="",
                markercolor="green", linecolor="green", markershapes=markershape, markersize=2, legend=:topleft)
            end
            wave_speed==1 && savefig(joinpath(directory, name))
          end
        end
      end
    end
  end
end

@testset "Non-linear Bergers, ∂_t w + ∇•(ww) = 0 ∈ ∂Ω, u(t=0) = Gaussian(σ, μ), ConservativeForm, explicit Euler" begin
  dd = DomainDecomp(gm=1)
  dss = DomainSubSet(gm=true)
  σ, μ = .1, 0.5
  δz = 0.2
  Triangle(z, velocity_sign)   = μ-δz < z < μ+δz ? ( μ > z ? velocity_sign*(z-(μ-δz))/δz : velocity_sign*((μ+δz)-z)/δz  ) : 0.0
  Square(z, velocity_sign) = μ-δz < z < μ+δz ? velocity_sign : 0.0
  Gaussian(z, velocity_sign) = velocity_sign*exp(-1/2*((z-μ)/σ)^2)
  for n_elems_real in (64, 128)
    grid = Grid(0.0, 1.0, n_elems_real)
    domain_range = over_elems_real(grid)
    x = [grid.zc[k] for k in domain_range]
    unknowns = ( (:w, dss), )
    vars = ( (:w_initial, dss), (:w_error, dss), (:w_analytic, dss), )
    q = StateVec(unknowns, grid, dd)
    tmp = StateVec(vars, grid, dd)
    rhs = deepcopy(q)
    CFL = 0.1
    Δt = CFL*grid.Δz
    T = 0.25
    maxiter = Int(T/Δt)
    for scheme in (
                   UpwindAdvective(),
                   UpwindCollocated(),
                   CenteredUnstable(),
                   )
      for distribution in (Triangle,
                           Square,
                           Gaussian)
        for velocity_sign in (-1, 1)
          for k in over_elems_real(grid)
            tmp[:w_initial, k] = distribution(grid.zc[k], velocity_sign)
            q[:w, k] = tmp[:w_initial, k]
          end
          amax_w = max([max(tmp[:w_initial, k]) for k in over_elems_real(grid)]...)
          for i in 1:maxiter
            for k in over_elems_real(grid)
              w = q[:w, Cut(k)]
              w_dual = (w[1:2]+w[2:3])/2
              rhs[:w, k] = - advect_old(w, w_dual, w, w_dual, grid, scheme, Δt)
            end
            for k in over_elems(grid)
              q[:w, k] += Δt*rhs[:w, k]
            end
            apply_Dirichlet!(q, :w, grid, 0.0, Zmax())
            apply_Dirichlet!(q, :w, grid, 0.0, Zmin())
          end

          scheme_name = replace(string(scheme), "()"=>"")
          distribution_name = joinpath("BurgersEquation",string(distribution))
          directory = joinpath(adv_eq_dir, distribution_name, scheme_name)
          export_plots && mkpath(directory)
          name = string(n_elems_real)
          markershape = velocity_sign==-1 ? :dtriangle : :utriangle
          if export_plots
            if velocity_sign==-1
              y = [tmp[:w_initial , k] for k in domain_range]; plot(y, x, label="initial",
                markercolor="blue", linecolor="blue", markershapes=markershape, markersize=2, legend=:topleft)
              y = [  q[:w         , k] for k in domain_range]; plot!(y, x, label="numerical",
                markercolor="black", linecolor="black", markershapes=markershape, markersize=2, legend=:topleft)
            else
              y = [tmp[:w_initial , k] for k in domain_range]; plot!(y, x, label="",
                markercolor="blue", linecolor="blue", markershapes=markershape, markersize=2, legend=:topleft)
              y = [  q[:w         , k] for k in domain_range]; plot!(y, x, label="",
                markercolor="black", linecolor="black", markershapes=markershape, markersize=2, legend=:topleft)
            end
            velocity_sign==1 && savefig(joinpath(directory, name))
          end
        end
      end
    end
  end

  end