using Test, Printf, ForwardDiff

using CLIMA.TurbulenceConvection.FiniteDifferenceGrids

@testset "Grid interface" begin
  for DT in (Float32, Float64)
    for n_ghost in (1, 2)
      n_elems = 12
      n_elems_real = n_elems-2*n_ghost
      elem_indexes = 1:n_elems
      elem_indexes_real = elem_indexes[1+n_ghost:end-n_ghost]
      Δz = DT(1/n_elems_real)
      grid = Grid(DT(0.0), DT(1.0), n_elems_real, n_ghost)
      @test n_hat(Zmin()) == -1
      @test n_hat(Zmax()) == 1
      @test binary(Zmin()) == 0
      @test binary(Zmax()) == 1
      @test ghost_dual(Zmin()) == [1, 0]
      @test ghost_dual(Zmax()) == [0, 1]
      @test eltype(grid) == DT
      @test all(over_elems(grid) .== elem_indexes)
      @test all(over_elems_real(grid) .== elem_indexes_real)
      @test length(over_elems(grid)) == n_elems
      @test length(over_elems_real(grid)) == n_elems_real
      @test first_interior(grid, Zmax()) == n_elems-n_ghost
      @test first_interior(grid, Zmin()) == 1+n_ghost
      @test boundary(grid, Zmin()) == 1+n_ghost
      @test boundary(grid, Zmax()) == n_elems-n_ghost
      @test over_elems_ghost(grid) == [(1:n_ghost)..., (n_elems+1-n_ghost:n_elems)...]
      @test grid.zn_min ≈ DT(0.0)
      @test grid.zn_max  ≈ DT(1.0)
      sprint(show, grid)
    end
  end
end

function ∇(f, x::T) where {T}
    tag = typeof(ForwardDiff.Tag(f, T))
    y = f(ForwardDiff.Dual{tag}(x,one(x)))
    ForwardDiff.partials(tag, y, 1)
end

@testset "Grid operators" begin
  for DT in (Float32, Float64)
    n_ghost = 1
    n_elems = 12
    n_elems_real = n_elems-2*n_ghost
    elem_indexes = 1:n_elems
    elem_indexes_real = elem_indexes[1+n_ghost:end-n_ghost]
    Δz = DT(1/n_elems_real)
    grid = Grid(DT(0.0), DT(1.0), n_elems_real, n_ghost)
    f = DT[1, 5, 3]
    w = DT[2, 3, 8]
    @test grad(f[1:2], grid) ≈ DT(40)
    @test grad(f, grid) ≈ DT(10)
    @test_throws AssertionError grad(f[1], grid)
    @test_throws AssertionError advect(f[1:2], w[1:2], grid)
    @test advect(f, w, grid) ≈ DT(40)
    @test advect(f, -w, grid) ≈ DT(-20)
    @test ∇_pos(f, grid) ≈ DT(40)
    @test ∇_neg(f, grid) ≈ DT(-20)
  end
end

@testset "Grid operators convergence" begin
  z_min = 0.0
  z_max = 2.0*π
  tol = 1/2
  ϕ_(x) = 5+sin(x)
  ∇ϕ_(x) = cos(x) # computed manually, since ForwardDiff doesn't like gradient functions as inputs.
  ψ_(x) = 5+cos(x)
  w_(x) = 5+sin(x)
  K_(x) = 5+sin(x)
  ϕψ_(x) = ψ_(x)*ϕ_(x)
  wϕ_(x) = w_(x)*ϕ_(x)
  K∇ϕ_(x) = K_(x)*∇ϕ_(x)

  for n_elems_real in [2^k for k in 3:8]
    grid = Grid(z_min, z_max, n_elems_real)
    z = grid.zc

    ϕ     = ϕ_.(z)
    ψ     = ψ_.(z)
    ϕψ    = ϕψ_.(z)
    w     = w_.(z)
    K     = K_.(z)

    ∇ϕ  = ∇.(ϕ_, z)
    ∇ψ  = ∇.(ψ_, z)
    ∇ϕψ = ∇.(ϕψ_, z)
    ∇w  = ∇.(w_, z)
    ∇K∇ϕ = ∇.(K∇ϕ_, z)
    ∇ϕ  = ∇.(ϕ_, z)
    ∇wϕ  = ∇.(wϕ_, z)

    del_err = [abs(∇ϕ[k] - ∇_z_upwind(ϕ[k-1:k+1], w[k-1:k+1], grid)) for k in over_elems_real(grid)]
    @test all([e<tol*grid.Δz for e in del_err])

    del_err = [abs((∇ϕ[k-1]+∇ϕ[k])/2 - ∇_z_flux(ϕ[k-1:k], grid)) for k in over_elems_real(grid)]
    @test all([e<tol*grid.Δz^2 for e in del_err])
    del_err = [abs((∇ϕ[k]+∇ϕ[k+1])/2 - ∇_z_flux(ϕ[k:k+1], grid)) for k in over_elems_real(grid)]
    @test all([e<tol*grid.Δz^2 for e in del_err])

    del_err = [abs(∇ϕ[k] - ∇_z_centered(ϕ[k-1:k+1], grid)) for k in over_elems_real(grid)]
    @test all([e<tol*grid.Δz^2 for e in del_err])

    del_err = [abs((∇ϕ[k-1]+∇ϕ[k])/2 - ∇_z_dual(ϕ[k-1:k+1], grid)[1]) for k in over_elems_real(grid)]
    @test all([e<tol*grid.Δz^2 for e in del_err])
    del_err = [abs((∇ϕ[k]+∇ϕ[k+1])/2 - ∇_z_dual(ϕ[k-1:k+1], grid)[2]) for k in over_elems_real(grid)]
    @test all([e<tol*grid.Δz^2 for e in del_err])

    Lap_err = [abs(∇ψ[k] - Δ_z(ϕ[k-1:k+1], grid)) for k in over_elems_real(grid)]
    @test all([e<tol*grid.Δz^2 for e in Lap_err])

    Lap_err = [abs(∇K∇ϕ[k] - Δ_z(ϕ[k-1:k+1], grid, K[k-1:k+1])) for k in over_elems_real(grid)]
    @test all([e<2*tol*grid.Δz^2 for e in Lap_err])

    Lap_err = [abs(∇K∇ϕ[k] - Δ_z_dual(ϕ[k-1:k+1], grid, 0.5*(K[k-1:k].+K[k:k+1]))) for k in over_elems_real(grid)]
    @test all([e<2*tol*grid.Δz^2 for e in Lap_err])
    Δt = 1.0

    # ∇•(wϕ)
    for conservatove_scheme in (
                                UpwindCollocated(),
                                OneSidedUp(),
                                OneSidedDn(),
                                CenteredUnstable(),
                                )
      adv_err = [abs(∇wϕ[k] - advect_old(ϕ[k-1:k+1],
                                     0.5*(ϕ[k-1:k].+ϕ[k:k+1]),
                                     w[k-1:k+1],
                                     0.5*(w[k-1:k].+w[k:k+1]),
                                     grid, conservatove_scheme, Δt)) for k in over_elems_real(grid)]
      @test all([e<12*tol*grid.Δz for e in adv_err])
    end
    # w•∇ϕ
    for non_conservative_scheme in (UpwindAdvective(),)
      adv_err = [abs(w[k]*∇ϕ[k] - advect_old(ϕ[k-1:k+1],
                                         0.5*(ϕ[k-1:k].+ϕ[k:k+1]),
                                         w[k-1:k+1],
                                         0.5*(w[k-1:k].+w[k:k+1]),
                                         grid, non_conservative_scheme, Δt)) for k in over_elems_real(grid)]
      @test all([e<6*tol*grid.Δz for e in adv_err])
    end

  end
end
