using CLIMA.Mesh.Elements
using CLIMA.Mesh.Metrics
using Test


const VGEO2D = (x=1, y=2, J=3, ξx=4, ηx=5, ξy=6, ηy=7)
const SGEO2D = (sJ = 1, nx = 2, ny = 3)

const VGEO3D = (x = 1, y = 2, z = 3, J = 4, ξx = 5, ηx = 6, ζx = 7, ξy = 8,
                ηy = 9, ζy = 10, ξz = 11, ηz = 12, ζz = 13)
const SGEO3D = (sJ = 1, nx = 2, ny = 3, nz = 4)

@testset "1-D Metric terms" begin
  for T ∈ (Float32, Float64)
    #{{{
    let
      N = 4

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 1
      nfaces = 4
      e2c = Array{T, 3}(undef, 1, 2, 2)
      e2c[:, :, 1] = [-1 0]
      e2c[:, :, 2] = [ 0 10]
      nelem = size(e2c, 3)

      (x, ) = Metrics.creategrid1d(e2c, r)
      @test x[:, 1] ≈ (r .- 1) / 2
      @test x[:, 2] ≈ 5 * (r .+ 1)

      metric = Metrics.computemetric(x, D)
      @test metric.J[:, 1] ≈ ones(T, Nq) / 2
      @test metric.J[:, 2] ≈ 5 * ones(T, Nq)
      @test metric.ξx[:, 1] ≈ 2 * ones(T, Nq)
      @test metric.ξx[:, 2] ≈ ones(T, Nq) / 5
      @test metric.nx[1, 1, :] ≈ -ones(T, nelem)
      @test metric.nx[1, 2, :] ≈  ones(T, nelem)
      @test metric.sJ[1, 1, :] ≈  ones(T, nelem)
      @test metric.sJ[1, 2, :] ≈  ones(T, nelem)
    end
    #}}}
  end
end

@testset "2-D Metric terms" begin
  for T ∈ (Float32, Float64)
    # linear and rotation test
    #{{{
    let
      N = 2

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 2
      nfaces = 4
      e2c = Array{T, 3}(undef, 2, 4, 4)
      e2c[:, :, 1] = [0 2 0 2;0 0 2 2]
      e2c[:, :, 2] = [2 2 0 0;0 2 0 2]
      e2c[:, :, 3] = [2 0 2 0;2 2 0 0]
      e2c[:, :, 4] = [0 0 2 2;2 0 2 0]
      nelem = size(e2c, 3)

      x_exact = Array{Int, 3}(undef, 3, 3, 4)
      x_exact[:, :, 1] = [0 0 0; 1 1 1; 2 2 2]
      x_exact[:, :, 2] = rotr90(x_exact[:, :, 1])
      x_exact[:, :, 3] = rotr90(x_exact[:, :, 2])
      x_exact[:, :, 4] = rotr90(x_exact[:, :, 3])

      y_exact = Array{Int, 3}(undef, 3, 3, 4)
      y_exact[:, :, 1] = [0 1 2; 0 1 2; 0 1 2]
      y_exact[:, :, 2] = rotr90(y_exact[:, :, 1])
      y_exact[:, :, 3] = rotr90(y_exact[:, :, 2])
      y_exact[:, :, 4] = rotr90(y_exact[:, :, 3])

      J_exact = ones(Int, 3, 3, 4)

      ξx_exact = zeros(Int, 3, 3, 4)
      ξx_exact[:, :, 1] .= 1
      ξx_exact[:, :, 3] .= -1

      ξy_exact = zeros(Int, 3, 3, 4)
      ξy_exact[:, :, 2] .= 1
      ξy_exact[:, :, 4] .= -1

      ηx_exact = zeros(Int, 3, 3, 4)
      ηx_exact[:, :, 2] .= -1
      ηx_exact[:, :, 4] .= 1

      ηy_exact = zeros(Int, 3, 3, 4)
      ηy_exact[:, :, 1] .= 1
      ηy_exact[:, :, 3] .= -1

      sJ_exact = ones(Int, Nq, nfaces, nelem)

      nx_exact = zeros(Int, Nq, nfaces, nelem)
      nx_exact[:, 1, 1] .= -1
      nx_exact[:, 2, 1] .=  1
      nx_exact[:, 3, 2] .=  1
      nx_exact[:, 4, 2] .= -1
      nx_exact[:, 1, 3] .=  1
      nx_exact[:, 2, 3] .= -1
      nx_exact[:, 3, 4] .= -1
      nx_exact[:, 4, 4] .=  1

      ny_exact = zeros(Int, Nq, nfaces, nelem)
      ny_exact[:, 3, 1] .= -1
      ny_exact[:, 4, 1] .=  1
      ny_exact[:, 1, 2] .= -1
      ny_exact[:, 2, 2] .=  1
      ny_exact[:, 3, 3] .=  1
      ny_exact[:, 4, 3] .= -1
      ny_exact[:, 1, 4] .=  1
      ny_exact[:, 2, 4] .= -1

      vgeo = Array{T, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
      sgeo = Array{T, 4}(undef, Nq, nfaces, length(SGEO2D), nelem)
      Metrics.creategrid!(ntuple(j->(@view vgeo[:, :, j, :]), d)..., e2c, r)
      Metrics.computemetric!(ntuple(j->(@view vgeo[:, :, j, :]), length(VGEO2D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO2D))...,
      D)

      @test (@view vgeo[:,:,VGEO2D.x,:]) ≈ x_exact
      @test (@view vgeo[:,:,VGEO2D.y,:]) ≈ y_exact
      @test (@view vgeo[:, :, VGEO2D.J, :]) ≈ J_exact
      @test (@view vgeo[:, :, VGEO2D.ξx, :]) ≈ ξx_exact
      @test (@view vgeo[:, :, VGEO2D.ξy, :]) ≈ ξy_exact
      @test (@view vgeo[:, :, VGEO2D.ηx, :]) ≈ ηx_exact
      @test (@view vgeo[:, :, VGEO2D.ηy, :]) ≈ ηy_exact
      @test (@view sgeo[:, :, SGEO2D.sJ, :]) ≈ sJ_exact
      @test (@view sgeo[:, :, SGEO2D.nx, :]) ≈ nx_exact
      @test (@view sgeo[:, :, SGEO2D.ny, :]) ≈ ny_exact

      nothing
    end
    #}}}

    # Polynomial 2-D test
    #{{{
    let
      N = 4

      f(r,s) = (9 .* r - (1 .+ r) .* s.^2 + (r .- 1).^2 .* (1 .- s.^2 .+ s.^3),
                10 .* s .+ r.^4 .* (1 .- s) .+ r.^2 .* s .* (1 .+ s))
      fxr(r,s) = 7 .+ s.^2 .- 2 .* s.^3 .+ 2 .* r .* (1 .- s.^2 .+ s.^3)
      fxs(r,s) = -2 .* (1 .+ r) .* s .+ (-1 .+ r).^2 .* s .* (-2 .+ 3 .* s)
      fyr(r,s) = -4 .* r.^3 .* (-1 .+ s) .+ 2 .* r .* s .* (1 .+ s)
      fys(r,s) = 10 .- r.^4 .+ r.^2 .* (1 .+ 2 .* s)

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 2
      nfaces = 4
      e2c = Array{T, 3}(undef, 2, 4, 1)
      e2c[:, :, 1] = [-1 1 -1 1;-1 -1 1 1]
      nelem = size(e2c, 3)

      vgeo = Array{T, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
      sgeo = Array{T, 4}(undef, Nq, nfaces, length(SGEO2D), nelem)

      Metrics.creategrid!(ntuple(j->(@view vgeo[:, :, j, :]), d)..., e2c, r)
      x = @view vgeo[:, :, VGEO2D.x, :]
      y = @view vgeo[:, :, VGEO2D.y, :]

      (xr, xs, yr, ys) = (fxr(x, y), fxs(x,y), fyr(x,y), fys(x,y))
      J = xr .* ys - xs .* yr
      foreach(j->(x[j], y[j]) = f(x[j], y[j]), 1:length(x))

      Metrics.computemetric!(ntuple(j->(@view vgeo[:, :, j, :]), length(VGEO2D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO2D))...,
      D)
      @test J ≈ (@view vgeo[:, :, VGEO2D.J, :])
      @test (@view vgeo[:, :, VGEO2D.ξx, :]) ≈  ys ./ J
      @test (@view vgeo[:, :, VGEO2D.ηx, :]) ≈ -yr ./ J
      @test (@view vgeo[:, :, VGEO2D.ξy, :]) ≈ -xs ./ J
      @test (@view vgeo[:, :, VGEO2D.ηy, :]) ≈  xr ./ J

      # check the normals?
      nx = @view sgeo[:,:,SGEO2D.nx,:]
      ny = @view sgeo[:,:,SGEO2D.ny,:]
      sJ = @view sgeo[:,:,SGEO2D.sJ,:]
      @test hypot.(nx, ny) ≈ ones(T, size(nx))
      @test sJ[:,1,:] .* nx[:,1,:] ≈ -ys[ 1,:,:]
      @test sJ[:,1,:] .* ny[:,1,:] ≈  xs[ 1,:,:]
      @test sJ[:,2,:] .* nx[:,2,:] ≈  ys[Nq,:,:]
      @test sJ[:,2,:] .* ny[:,2,:] ≈ -xs[Nq,:,:]
      @test sJ[:,3,:] .* nx[:,3,:] ≈  yr[:, 1,:]
      @test sJ[:,3,:] .* ny[:,3,:] ≈ -xr[:, 1,:]
      @test sJ[:,4,:] .* nx[:,4,:] ≈ -yr[:,Nq,:]
      @test sJ[:,4,:] .* ny[:,4,:] ≈  xr[:,Nq,:]
    end
    #}}}

    #{{{
    let
      N = 4

      f(r,s) = (9 .* r - (1 .+ r) .* s.^2 + (r .- 1).^2 .* (1 .- s.^2 .+ s.^3),
                10 .* s .+ r.^4 .* (1 .- s) .+ r.^2 .* s .* (1 .+ s))
      fxr(r,s) = 7 .+ s.^2 .- 2 .* s.^3 .+ 2 .* r .* (1 .- s.^2 .+ s.^3)
      fxs(r,s) = -2 .* (1 .+ r) .* s .+ (-1 .+ r).^2 .* s .* (-2 .+ 3 .* s)
      fyr(r,s) = -4 .* r.^3 .* (-1 .+ s) .+ 2 .* r .* s .* (1 .+ s)
      fys(r,s) = 10 .- r.^4 .+ r.^2 .* (1 .+ 2 .* s)

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 2
      nfaces = 4
      e2c = Array{T, 3}(undef, 2, 4, 1)
      e2c[:, :, 1] = [-1 1 -1 1;-1 -1 1 1]
      nelem = size(e2c, 3)

      (x, y) = Metrics.creategrid2d(e2c, r)

      (xr, xs, yr, ys) = (fxr(x, y), fxs(x,y), fyr(x,y), fys(x,y))
      J = xr .* ys - xs .* yr
      foreach(j->(x[j], y[j]) = f(x[j], y[j]), 1:length(x))

      metric = Metrics.computemetric(x, y, D)
      @test J ≈ metric.J
      @test metric.ξx ≈  ys ./ J
      @test metric.ηx ≈ -yr ./ J
      @test metric.ξy ≈ -xs ./ J
      @test metric.ηy ≈  xr ./ J

      # check the normals?
      nx = metric.nx
      ny = metric.ny
      sJ = metric.sJ
      @test hypot.(nx, ny) ≈ ones(T, size(nx))
      @test sJ[:,1,:] .* nx[:,1,:] ≈ -ys[ 1,:,:]
      @test sJ[:,1,:] .* ny[:,1,:] ≈  xs[ 1,:,:]
      @test sJ[:,2,:] .* nx[:,2,:] ≈  ys[Nq,:,:]
      @test sJ[:,2,:] .* ny[:,2,:] ≈ -xs[Nq,:,:]
      @test sJ[:,3,:] .* nx[:,3,:] ≈  yr[:, 1,:]
      @test sJ[:,3,:] .* ny[:,3,:] ≈ -xr[:, 1,:]
      @test sJ[:,4,:] .* nx[:,4,:] ≈ -yr[:,Nq,:]
      @test sJ[:,4,:] .* ny[:,4,:] ≈  xr[:,Nq,:]
    end
    #}}}
  end

  # Constant preserving test
  #{{{
  let
    N = 4
    T = Float64

    f(r,s) = (9 .* r - (1 .+ r) .* s.^2 + (r .- 1).^2 .* (1 .- s.^2 .+ s.^3),
              10 .* s .+ r.^4 .* (1 .- s) .+ r.^2 .* s .* (1 .+ s))
    fxr(r,s) = 7 .+ s.^2 .- 2 .* s.^3 .+ 2 .* r .* (1 .- s.^2 .+ s.^3)
    fxs(r,s) = -2 .* (1 .+ r) .* s .+ (-1 .+ r).^2 .* s .* (-2 .+ 3 .* s)
    fyr(r,s) = -4 .* r.^3 .* (-1 .+ s) .+ 2 .* r .* s .* (1 .+ s)
    fys(r,s) = 10 .- r.^4 .+ r.^2 .* (1 .+ 2 .* s)

    r, w = Elements.lglpoints(T, N)
    D = Elements.spectralderivative(r)
    Nq = N + 1

    d = 2
    nfaces = 4
    e2c = Array{T, 3}(undef, 2, 4, 1)
    e2c[:, :, 1] = [-1 1 -1 1;-1 -1 1 1]
    nelem = size(e2c, 3)

    vgeo = Array{T, 4}(undef, Nq, Nq, length(VGEO2D), nelem)
    sgeo = Array{T, 4}(undef, Nq, nfaces, length(SGEO2D), nelem)

    Metrics.creategrid!(ntuple(j->(@view vgeo[:, :, j, :]), d)..., e2c, r)
    x = @view vgeo[:, :, VGEO2D.x, :]
    y = @view vgeo[:, :, VGEO2D.y, :]

    foreach(j->(x[j], y[j]) = f(x[j], y[j]), 1:length(x))

    Metrics.computemetric!(ntuple(j->(@view vgeo[:, :, j, :]), length(VGEO2D))...,
                          ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO2D))...,
                          D)

    (Cx, Cy) = (zeros(T, Nq, Nq), zeros(T, Nq, Nq))

    J  = @view vgeo[:,:,VGEO2D.J ,:]
    ξx = @view vgeo[:,:,VGEO2D.ξx,:]
    ξy = @view vgeo[:,:,VGEO2D.ξy,:]
    ηx = @view vgeo[:,:,VGEO2D.ηx,:]
    ηy = @view vgeo[:,:,VGEO2D.ηy,:]

    e = 1
    for n = 1:Nq
      Cx[:, n] += D * (J[:, n, e] .* ξx[:, n, e])
      Cx[n, :] += D * (J[n, :, e] .* ηx[n, :, e])

      Cy[:, n] += D * (J[:, n, e] .* ξx[:, n, e])
      Cy[n, :] += D * (J[n, :, e] .* ηx[n, :, e])
    end
    @test maximum(abs.(Cx)) ≤ 1000 * eps(T)
    @test maximum(abs.(Cy)) ≤ 1000 * eps(T)
  end
  #}}}
end

@testset "3-D Metric terms" begin
  # linear test
    #{{{
    for T ∈ (Float32, Float64)
    let
      N = 2

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 2)
      e2c[:, :, 1] = [0 2 0 2 0 2 0 2;
                      0 0 2 2 0 0 2 2;
                      0 0 0 0 2 2 2 2]
      e2c[:, :, 2] = [2 2 0 0 2 2 0 0;
                      0 2 0 2 0 2 0 2;
                      0 0 0 0 2 2 2 2]

      nelem = size(e2c, 3)

      x_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
      x_exact[1, :, :, 1] .= 0
      x_exact[2, :, :, 1] .= 1
      x_exact[3, :, :, 1] .= 2
      x_exact[:, 1, :, 2] .= 2
      x_exact[:, 2, :, 2] .= 1
      x_exact[:, 3, :, 2] .= 0

      ξx_exact = zeros(Int, 3, 3, 3, nelem)
      ξx_exact[:, :, :, 1] .= 1

      ξy_exact = zeros(Int, 3, 3, 3, nelem)
      ξy_exact[:, :, :, 2] .= 1

      ηx_exact = zeros(Int, 3, 3, 3, nelem)
      ηx_exact[:, :, :, 2] .= -1

      ηy_exact = zeros(Int, 3, 3, 3, nelem)
      ηy_exact[:, :, :, 1] .= 1

      ζz_exact = ones(Int, 3, 3, 3, nelem)

      y_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
      y_exact[:, 1, :, 1] .= 0
      y_exact[:, 2, :, 1] .= 1
      y_exact[:, 3, :, 1] .= 2
      y_exact[1, :, :, 2] .= 0
      y_exact[2, :, :, 2] .= 1
      y_exact[3, :, :, 2] .= 2

      z_exact = Array{Int, 4}(undef, 3, 3, 3, nelem)
      z_exact[:, :, 1, 1:2] .= 0
      z_exact[:, :, 2, 1:2] .= 1
      z_exact[:, :, 3, 1:2] .= 2

      J_exact = ones(Int, 3, 3, 3, nelem)

      sJ_exact = ones(Int, Nq, Nq, nfaces, nelem)

      nx_exact = zeros(Int, Nq, Nq, nfaces, nelem)
      nx_exact[:, :, 1, 1] .= -1
      nx_exact[:, :, 2, 1] .=  1
      nx_exact[:, :, 3, 2] .=  1
      nx_exact[:, :, 4, 2] .= -1

      ny_exact = zeros(Int, Nq, Nq, nfaces, nelem)
      ny_exact[:, :, 3, 1] .= -1
      ny_exact[:, :, 4, 1] .=  1
      ny_exact[:, :, 1, 2] .= -1
      ny_exact[:, :, 2, 2] .=  1

      nz_exact = zeros(Int, Nq, Nq, nfaces, nelem)
      nz_exact[:, :, 5, 1:2] .= -1
      nz_exact[:, :, 6, 1:2] .=  1

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 4}(undef, Nq^2, nfaces, length(SGEO3D), nelem)
      Metrics.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      Metrics.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO3D))...,
      D)
      sgeo = reshape(sgeo, Nq, Nq, nfaces, length(SGEO3D), nelem)

      @test (@view vgeo[:,:,:,VGEO3D.x,:]) ≈ x_exact
      @test (@view vgeo[:,:,:,VGEO3D.y,:]) ≈ y_exact
      @test (@view vgeo[:,:,:,VGEO3D.z,:]) ≈ z_exact
      @test (@view vgeo[:,:,:,VGEO3D.J,:]) ≈ J_exact
      @test (@view vgeo[:,:,:,VGEO3D.ξx,:]) ≈ ξx_exact
      @test (@view vgeo[:,:,:,VGEO3D.ξy,:]) ≈ ξy_exact
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.ξz,:])) ≤ 10 * eps(T)
      @test (@view vgeo[:,:,:,VGEO3D.ηx,:]) ≈ ηx_exact
      @test (@view vgeo[:,:,:,VGEO3D.ηy,:]) ≈ ηy_exact
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.ηz,:])) ≤ 10 * eps(T)
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.ζx,:])) ≤ 10 * eps(T)
      @test maximum(abs.(@view vgeo[:,:,:,VGEO3D.ζy,:])) ≤ 10 * eps(T)
      @test (@view vgeo[:,:,:,VGEO3D.ζz,:]) ≈ ζz_exact
      @test (@view sgeo[:,:,:,SGEO3D.sJ,:]) ≈ sJ_exact
      @test (@view sgeo[:,:,:,SGEO3D.nx,:]) ≈ nx_exact
      @test (@view sgeo[:,:,:,SGEO3D.ny,:]) ≈ ny_exact
      @test (@view sgeo[:,:,:,SGEO3D.nz,:]) ≈ nz_exact
    end
  end
  #}}}

  # linear test with rotation
  #{{{
  for T ∈ (Float32, Float64)
    θ1 = 2 * T(π) * T( 0.9 )
    θ2 = 2 * T(π) * T(-0.56)
    θ3 = 2 * T(π) * T( 0.33)
    #=
    θ1 = 2 * T(π) * rand(T)
    θ2 = 2 * T(π) * rand(T)
    θ3 = 2 * T(π) * rand(T)
    =#
    #=
    θ1 = T(π) / 6
    θ2 = T(π) / 12
    θ3 = 4 * T(π) / 5
    =#
    Q  = [cos(θ1) -sin(θ1) 0; sin(θ1) cos(θ1) 0; 0 0 1]
    Q *= [cos(θ2) 0 -sin(θ2); 0 1 0; sin(θ2) 0 cos(θ2)]
    Q *= [1 0 0; 0 cos(θ3) -sin(θ3); 0 sin(θ3) cos(θ3)]
    let
      N = 2

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 1)
      e2c[:, :, 1] = [0 2 0 2 0 2 0 2;
                      0 0 2 2 0 0 2 2;
                      0 0 0 0 2 2 2 2]
      @views (x, y, z) = (e2c[1, :, 1], e2c[2, :, 1], e2c[3, :, 1])
      for i = 1:length(x)
        (x[i], y[i], z[i]) = Q * [x[i]; y[i]; z[i]]
      end

      nelem = size(e2c, 3)

      xe = Array{T, 4}(undef, 3, 3, 3, nelem)
      xe[1, :, :, 1] .= 0
      xe[2, :, :, 1] .= 1
      xe[3, :, :, 1] .= 2

      ye = Array{T, 4}(undef, 3, 3, 3, nelem)
      ye[:, 1, :, 1] .= 0
      ye[:, 2, :, 1] .= 1
      ye[:, 3, :, 1] .= 2

      ze = Array{T, 4}(undef, 3, 3, 3, nelem)
      ze[:, :, 1, 1] .= 0
      ze[:, :, 2, 1] .= 1
      ze[:, :, 3, 1] .= 2

      for i = 1:length(xe)
        (xe[i], ye[i], ze[i]) = Q * [xe[i]; ye[i]; ze[i]]
      end

      Je = ones(Int, 3, 3, 3, nelem)

      # By construction
      # Q = [xr xs ys; yr ys yt; zr zs zt] = [ξx ηx ζx; ξy ηy ζy; ξz ηz ζz]
      rxe = fill(Q[1,1], 3, 3, 3, nelem)
      rye = fill(Q[2,1], 3, 3, 3, nelem)
      rze = fill(Q[3,1], 3, 3, 3, nelem)
      sxe = fill(Q[1,2], 3, 3, 3, nelem)
      sye = fill(Q[2,2], 3, 3, 3, nelem)
      sze = fill(Q[3,2], 3, 3, 3, nelem)
      txe = fill(Q[1,3], 3, 3, 3, nelem)
      tye = fill(Q[2,3], 3, 3, 3, nelem)
      tze = fill(Q[3,3], 3, 3, 3, nelem)

      sJe = ones(Int, Nq, Nq, nfaces, nelem)

      nxe = zeros(T, Nq, Nq, nfaces, nelem)
      nye = zeros(T, Nq, Nq, nfaces, nelem)
      nze = zeros(T, Nq, Nq, nfaces, nelem)

      fill!(@view(nxe[:,:,1,:]), -Q[1,1])
      fill!(@view(nxe[:,:,2,:]),  Q[1,1])
      fill!(@view(nxe[:,:,3,:]), -Q[1,2])
      fill!(@view(nxe[:,:,4,:]),  Q[1,2])
      fill!(@view(nxe[:,:,5,:]), -Q[1,3])
      fill!(@view(nxe[:,:,6,:]),  Q[1,3])
      fill!(@view(nye[:,:,1,:]), -Q[2,1])
      fill!(@view(nye[:,:,2,:]),  Q[2,1])
      fill!(@view(nye[:,:,3,:]), -Q[2,2])
      fill!(@view(nye[:,:,4,:]),  Q[2,2])
      fill!(@view(nye[:,:,5,:]), -Q[2,3])
      fill!(@view(nye[:,:,6,:]),  Q[2,3])
      fill!(@view(nze[:,:,1,:]), -Q[3,1])
      fill!(@view(nze[:,:,2,:]),  Q[3,1])
      fill!(@view(nze[:,:,3,:]), -Q[3,2])
      fill!(@view(nze[:,:,4,:]),  Q[3,2])
      fill!(@view(nze[:,:,5,:]), -Q[3,3])
      fill!(@view(nze[:,:,6,:]),  Q[3,3])

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 4}(undef, Nq^2, nfaces, length(SGEO3D), nelem)
      Metrics.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      Metrics.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO3D))...,
      D)
      sgeo = reshape(sgeo, Nq, Nq, nfaces, length(SGEO3D), nelem)

      @test (@view vgeo[:,:,:,VGEO3D.x,:]) ≈ xe
      @test (@view vgeo[:,:,:,VGEO3D.y,:]) ≈ ye
      @test (@view vgeo[:,:,:,VGEO3D.z,:]) ≈ ze
      @test (@view vgeo[:,:,:,VGEO3D.J,:]) ≈ Je
      @test (@view vgeo[:,:,:,VGEO3D.ξx,:]) ≈ rxe
      @test (@view vgeo[:,:,:,VGEO3D.ξy,:]) ≈ rye
      @test (@view vgeo[:,:,:,VGEO3D.ξz,:]) ≈ rze
      @test (@view vgeo[:,:,:,VGEO3D.ηx,:]) ≈ sxe
      @test (@view vgeo[:,:,:,VGEO3D.ηy,:]) ≈ sye
      @test (@view vgeo[:,:,:,VGEO3D.ηz,:]) ≈ sze
      @test (@view vgeo[:,:,:,VGEO3D.ζx,:]) ≈ txe
      @test (@view vgeo[:,:,:,VGEO3D.ζy,:]) ≈ tye
      @test (@view vgeo[:,:,:,VGEO3D.ζz,:]) ≈ tze
      @test (@view sgeo[:,:,:,SGEO3D.sJ,:]) ≈ sJe
      @test (@view sgeo[:,:,:,SGEO3D.nx,:]) ≈ nxe
      @test (@view sgeo[:,:,:,SGEO3D.ny,:]) ≈ nye
      @test (@view sgeo[:,:,:,SGEO3D.nz,:]) ≈ nze
    end
  end
  #}}}

  # Polynomial 3-D test
  #{{{
  for T ∈ (Float32, Float64)
    f(r, s, t) = @.( (s + r*t - (r^2*s^2*t^2)/4,
                      t - ((r*s*t)/2 + 1/2)^3 + 1,
                      r + (r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^6))

    fxr(r, s, t) = @.(t - (r*s^2*t^2)/2)
    fxs(r, s, t) = @.(1 - (r^2*s*t^2)/2)
    fxt(r, s, t) = @.(r - (r^2*s^2*t)/2)
    fyr(r, s, t) = @.(-(3*s*t*((r*s*t)/2 + 1/2)^2)/2)
    fys(r, s, t) = @.(-(3*r*t*((r*s*t)/2 + 1/2)^2)/2)
    fyt(r, s, t) = @.(1 - (3*r*s*((r*s*t)/2 + 1/2)^2)/2)
    fzr(r, s, t) = @.(3*(r/2 + 1/2)^5*(s/2 + 1/2)^6*(t/2 + 1/2)^6 + 1)
    fzs(r, s, t) = @.(3*(r/2 + 1/2)^6*(s/2 + 1/2)^5*(t/2 + 1/2)^6)
    fzt(r, s, t) = @.(3*(r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^5)

    let
      N = 9

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 1)
      e2c[:, :, 1] = [-1  1 -1  1 -1  1 -1  1;
                      -1 -1  1  1 -1 -1  1  1;
                      -1 -1 -1 -1  1  1  1  1]

      nelem = size(e2c, 3)

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 4}(undef, Nq^2, nfaces, length(SGEO3D), nelem)
      Metrics.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      x = @view vgeo[:, :, :, VGEO3D.x, :]
      y = @view vgeo[:, :, :, VGEO3D.y, :]
      z = @view vgeo[:, :, :, VGEO3D.z, :]

      (xr, xs, xt,
       yr, ys, yt,
       zr, zs, zt) = (fxr(x,y,z), fxs(x,y,z), fxt(x,y,z),
                      fyr(x,y,z), fys(x,y,z), fyt(x,y,z),
                      fzr(x,y,z), fzs(x,y,z), fzt(x,y,z))
      J = (xr .* (ys .* zt - yt .* zs) +
           yr .* (zs .* xt - zt .* xs) +
           zr .* (xs .* yt - xt .* ys))

      ξx =  (ys .* zt - yt .* zs) ./ J
      ξy =  (zs .* xt - zt .* xs) ./ J
      ξz =  (xs .* yt - xt .* ys) ./ J
      ηx =  (yt .* zr - yr .* zt) ./ J
      ηy =  (zt .* xr - zr .* xt) ./ J
      ηz =  (xt .* yr - xr .* yt) ./ J
      ζx =  (yr .* zs - ys .* zr) ./ J
      ζy =  (zr .* xs - zs .* xr) ./ J
      ζz =  (xr .* ys - xs .* yr) ./ J

      foreach(j->(x[j], y[j], z[j]) = f(x[j], y[j], z[j]), 1:length(x))

      Metrics.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO3D))...,
      D)
      sgeo = reshape(sgeo, Nq, Nq, nfaces, length(SGEO3D), nelem)

      @test (@view vgeo[:,:,:,VGEO3D.J,:]) ≈ J
      @test (@view vgeo[:,:,:,VGEO3D.ξx,:]) ≈ ξx
      @test (@view vgeo[:,:,:,VGEO3D.ξy,:]) ≈ ξy
      @test (@view vgeo[:,:,:,VGEO3D.ξz,:]) ≈ ξz
      @test (@view vgeo[:,:,:,VGEO3D.ηx,:]) ≈ ηx
      @test (@view vgeo[:,:,:,VGEO3D.ηy,:]) ≈ ηy
      @test (@view vgeo[:,:,:,VGEO3D.ηz,:]) ≈ ηz
      @test (@view vgeo[:,:,:,VGEO3D.ζx,:]) ≈ ζx
      @test (@view vgeo[:,:,:,VGEO3D.ζy,:]) ≈ ζy
      @test (@view vgeo[:,:,:,VGEO3D.ζz,:]) ≈ ζz
      nx = @view sgeo[:,:,:,SGEO3D.nx,:]
      ny = @view sgeo[:,:,:,SGEO3D.ny,:]
      nz = @view sgeo[:,:,:,SGEO3D.nz,:]
      sJ = @view sgeo[:,:,:,SGEO3D.sJ,:]
      @test hypot.(nx, ny, nz) ≈ ones(T, size(nx))
      @test ([sJ[:,:,1,:] .* nx[:,:,1,:], sJ[:,:,1,:] .* ny[:,:,1,:],
              sJ[:,:,1,:] .* nz[:,:,1,:]] ≈
             [-J[ 1,:,:,:] .* ξx[ 1,:,:,:], -J[ 1,:,:,:] .* ξy[ 1,:,:,:],
              -J[ 1,:,:,:] .* ξz[ 1,:,:,:]])
      @test ([sJ[:,:,2,:] .* nx[:,:,2,:], sJ[:,:,2,:] .* ny[:,:,2,:],
              sJ[:,:,2,:] .* nz[:,:,2,:]] ≈
             [ J[Nq,:,:,:] .* ξx[Nq,:,:,:],  J[Nq,:,:,:] .* ξy[Nq,:,:,:],
               J[Nq,:,:,:] .* ξz[Nq,:,:,:]])
      @test sJ[:,:,3,:] .* nx[:,:,3,:] ≈ -J[:, 1,:,:] .* ηx[:, 1,:,:]
      @test sJ[:,:,3,:] .* ny[:,:,3,:] ≈ -J[:, 1,:,:] .* ηy[:, 1,:,:]
      @test sJ[:,:,3,:] .* nz[:,:,3,:] ≈ -J[:, 1,:,:] .* ηz[:, 1,:,:]
      @test sJ[:,:,4,:] .* nx[:,:,4,:] ≈  J[:,Nq,:,:] .* ηx[:,Nq,:,:]
      @test sJ[:,:,4,:] .* ny[:,:,4,:] ≈  J[:,Nq,:,:] .* ηy[:,Nq,:,:]
      @test sJ[:,:,4,:] .* nz[:,:,4,:] ≈  J[:,Nq,:,:] .* ηz[:,Nq,:,:]
      @test sJ[:,:,5,:] .* nx[:,:,5,:] ≈ -J[:,:, 1,:] .* ζx[:,:, 1,:]
      @test sJ[:,:,5,:] .* ny[:,:,5,:] ≈ -J[:,:, 1,:] .* ζy[:,:, 1,:]
      @test sJ[:,:,5,:] .* nz[:,:,5,:] ≈ -J[:,:, 1,:] .* ζz[:,:, 1,:]
      @test sJ[:,:,6,:] .* nx[:,:,6,:] ≈  J[:,:,Nq,:] .* ζx[:,:,Nq,:]
      @test sJ[:,:,6,:] .* ny[:,:,6,:] ≈  J[:,:,Nq,:] .* ζy[:,:,Nq,:]
      @test sJ[:,:,6,:] .* nz[:,:,6,:] ≈  J[:,:,Nq,:] .* ζz[:,:,Nq,:]
    end
  end
  #}}}

  #{{{
  for T ∈ (Float32, Float64)
    f(r, s, t) = @.( (s + r*t - (r^2*s^2*t^2)/4,
                      t - ((r*s*t)/2 + 1/2)^3 + 1,
                      r + (r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^6))

    fxr(r, s, t) = @.(t - (r*s^2*t^2)/2)
    fxs(r, s, t) = @.(1 - (r^2*s*t^2)/2)
    fxt(r, s, t) = @.(r - (r^2*s^2*t)/2)
    fyr(r, s, t) = @.(-(3*s*t*((r*s*t)/2 + 1/2)^2)/2)
    fys(r, s, t) = @.(-(3*r*t*((r*s*t)/2 + 1/2)^2)/2)
    fyt(r, s, t) = @.(1 - (3*r*s*((r*s*t)/2 + 1/2)^2)/2)
    fzr(r, s, t) = @.(3*(r/2 + 1/2)^5*(s/2 + 1/2)^6*(t/2 + 1/2)^6 + 1)
    fzs(r, s, t) = @.(3*(r/2 + 1/2)^6*(s/2 + 1/2)^5*(t/2 + 1/2)^6)
    fzt(r, s, t) = @.(3*(r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^5)

    let
      N = 9

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 1)
      e2c[:, :, 1] = [-1  1 -1  1 -1  1 -1  1;
                      -1 -1  1  1 -1 -1  1  1;
                      -1 -1 -1 -1  1  1  1  1]

      nelem = size(e2c, 3)

      (x,y,z) = Metrics.creategrid3d(e2c, r)

      (xr, xs, xt,
       yr, ys, yt,
       zr, zs, zt) = (fxr(x,y,z), fxs(x,y,z), fxt(x,y,z),
                      fyr(x,y,z), fys(x,y,z), fyt(x,y,z),
                      fzr(x,y,z), fzs(x,y,z), fzt(x,y,z))
      J = (xr .* (ys .* zt - yt .* zs) +
           yr .* (zs .* xt - zt .* xs) +
           zr .* (xs .* yt - xt .* ys))

      ξx =  (ys .* zt - yt .* zs) ./ J
      ξy =  (zs .* xt - zt .* xs) ./ J
      ξz =  (xs .* yt - xt .* ys) ./ J
      ηx =  (yt .* zr - yr .* zt) ./ J
      ηy =  (zt .* xr - zr .* xt) ./ J
      ηz =  (xt .* yr - xr .* yt) ./ J
      ζx =  (yr .* zs - ys .* zr) ./ J
      ζy =  (zr .* xs - zs .* xr) ./ J
      ζz =  (xr .* ys - xs .* yr) ./ J

      foreach(j->(x[j], y[j], z[j]) = f(x[j], y[j], z[j]), 1:length(x))

      metric = Metrics.computemetric(x, y, z, D)

      @test metric.J ≈ J
      @test metric.ξx ≈ ξx
      @test metric.ξy ≈ ξy
      @test metric.ξz ≈ ξz
      @test metric.ηx ≈ ηx
      @test metric.ηy ≈ ηy
      @test metric.ηz ≈ ηz
      @test metric.ζx ≈ ζx
      @test metric.ζy ≈ ζy
      @test metric.ζz ≈ ζz
      ind = LinearIndices((1:Nq, 1:Nq))
      nx = metric.nx
      ny = metric.ny
      nz = metric.nz
      sJ = metric.sJ
      @test hypot.(nx, ny, nz) ≈ ones(T, size(nx))
      @test ([sJ[ind,1,:] .* nx[ind,1,:], sJ[ind,1,:] .* ny[ind,1,:],
              sJ[ind,1,:] .* nz[ind,1,:]] ≈
             [-J[ 1,:,:,:] .* ξx[ 1,:,:,:], -J[ 1,:,:,:] .* ξy[ 1,:,:,:],
              -J[ 1,:,:,:] .* ξz[ 1,:,:,:]])
      @test ([sJ[ind,2,:] .* nx[ind,2,:], sJ[ind,2,:] .* ny[ind,2,:],
              sJ[ind,2,:] .* nz[ind,2,:]] ≈
             [ J[Nq,:,:,:] .* ξx[Nq,:,:,:],  J[Nq,:,:,:] .* ξy[Nq,:,:,:],
               J[Nq,:,:,:] .* ξz[Nq,:,:,:]])
      @test sJ[ind,3,:] .* nx[ind,3,:] ≈ -J[:, 1,:,:] .* ηx[:, 1,:,:]
      @test sJ[ind,3,:] .* ny[ind,3,:] ≈ -J[:, 1,:,:] .* ηy[:, 1,:,:]
      @test sJ[ind,3,:] .* nz[ind,3,:] ≈ -J[:, 1,:,:] .* ηz[:, 1,:,:]
      @test sJ[ind,4,:] .* nx[ind,4,:] ≈  J[:,Nq,:,:] .* ηx[:,Nq,:,:]
      @test sJ[ind,4,:] .* ny[ind,4,:] ≈  J[:,Nq,:,:] .* ηy[:,Nq,:,:]
      @test sJ[ind,4,:] .* nz[ind,4,:] ≈  J[:,Nq,:,:] .* ηz[:,Nq,:,:]
      @test sJ[ind,5,:] .* nx[ind,5,:] ≈ -J[:,:, 1,:] .* ζx[:,:, 1,:]
      @test sJ[ind,5,:] .* ny[ind,5,:] ≈ -J[:,:, 1,:] .* ζy[:,:, 1,:]
      @test sJ[ind,5,:] .* nz[ind,5,:] ≈ -J[:,:, 1,:] .* ζz[:,:, 1,:]
      @test sJ[ind,6,:] .* nx[ind,6,:] ≈  J[:,:,Nq,:] .* ζx[:,:,Nq,:]
      @test sJ[ind,6,:] .* ny[ind,6,:] ≈  J[:,:,Nq,:] .* ζy[:,:,Nq,:]
      @test sJ[ind,6,:] .* nz[ind,6,:] ≈  J[:,:,Nq,:] .* ζz[:,:,Nq,:]
    end
  end
  #}}}


  # Constant preserving test
  #{{{
  for T ∈ (Float32, Float64)
    f(r, s, t) = @.( (s + r*t - (r^2*s^2*t^2)/4,
                      t - ((r*s*t)/2 + 1/2)^3 + 1,
                      r + (r/2 + 1/2)^6*(s/2 + 1/2)^6*(t/2 + 1/2)^6))
    let
      N = 5

      r, w = Elements.lglpoints(T, N)
      D = Elements.spectralderivative(r)
      Nq = N + 1

      d = 3
      nfaces = 6
      e2c = Array{T, 3}(undef, d, 8, 1)
      e2c[:, :, 1] = [-1  1 -1  1 -1  1 -1  1;
                      -1 -1  1  1 -1 -1  1  1;
                      -1 -1 -1 -1  1  1  1  1]

      nelem = size(e2c, 3)

      vgeo = Array{T, 5}(undef, Nq, Nq, Nq, length(VGEO3D), nelem)
      sgeo = Array{T, 4}(undef, Nq^2, nfaces, length(SGEO3D), nelem)
      Metrics.creategrid!(ntuple(j->(@view vgeo[:, :, :, j, :]), d)..., e2c, r)
      x = @view vgeo[:, :, :, VGEO3D.x, :]
      y = @view vgeo[:, :, :, VGEO3D.y, :]
      z = @view vgeo[:, :, :, VGEO3D.z, :]

      foreach(j->(x[j], y[j], z[j]) = f(x[j], y[j], z[j]), 1:length(x))

      Metrics.computemetric!(ntuple(j->(@view vgeo[:, :, :, j, :]), length(VGEO3D))...,
                            ntuple(j->(@view sgeo[:, :, j, :]), length(SGEO3D))...,
      D)
      sgeo = reshape(sgeo, Nq, Nq, nfaces, length(SGEO3D), nelem)

      (Cx, Cy, Cz) = (zeros(T, Nq, Nq, Nq), zeros(T, Nq, Nq, Nq),
                      zeros(T, Nq, Nq, Nq))

      J  = @view vgeo[:,:,:,VGEO3D.J ,:]
      ξx = @view vgeo[:,:,:,VGEO3D.ξx,:]
      ξy = @view vgeo[:,:,:,VGEO3D.ξy,:]
      ξz = @view vgeo[:,:,:,VGEO3D.ξz,:]
      ηx = @view vgeo[:,:,:,VGEO3D.ηx,:]
      ηy = @view vgeo[:,:,:,VGEO3D.ηy,:]
      ηz = @view vgeo[:,:,:,VGEO3D.ηz,:]
      ζx = @view vgeo[:,:,:,VGEO3D.ζx,:]
      ζy = @view vgeo[:,:,:,VGEO3D.ζy,:]
      ζz = @view vgeo[:,:,:,VGEO3D.ζz,:]

      e = 1
      for m = 1:Nq
        for n = 1:Nq
          Cx[:, n, m] += D * (J[:, n, m, e] .* ξx[:, n, m, e])
          Cx[n, :, m] += D * (J[n, :, m, e] .* ηx[n, :, m, e])
          Cx[n, m, :] += D * (J[n, m, :, e] .* ζx[n, m, :, e])

          Cy[:, n, m] += D * (J[:, n, m, e] .* ξx[:, n, m, e])
          Cy[n, :, m] += D * (J[n, :, m, e] .* ηx[n, :, m, e])
          Cy[n, m, :] += D * (J[n, m, :, e] .* ζx[n, m, :, e])

          Cz[:, n, m] += D * (J[:, n, m, e] .* ξx[:, n, m, e])
          Cz[n, :, m] += D * (J[n, :, m, e] .* ηx[n, :, m, e])
          Cz[n, m, :] += D * (J[n, m, :, e] .* ζx[n, m, :, e])
        end
      end
      @test maximum(abs.(Cx)) ≤ 1000 * eps(T)
      @test maximum(abs.(Cy)) ≤ 1000 * eps(T)
      @test maximum(abs.(Cz)) ≤ 1000 * eps(T)
    end
  end
  #}}}
end
