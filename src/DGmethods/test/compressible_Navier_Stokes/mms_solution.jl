# This file generates the solution used in method of manufactured solutions
using LinearAlgebra, SymPy, Printf

@syms x y z t real=true
μ = 1 // 100
γ = 7 // 5

output = open("mms_solution_generated.jl", "w")

@printf output "const γ_exact = %s\n" γ
@printf output "const μ_exact = %s\n" μ

for dim = 2:3
  if dim == 3
    ρ = cos(π * t) * sin(π * x) * cos(π * y) * cos(π * z) + 3
    U = cos(π * t) * ρ * sin(π * x) * cos(π * y) * cos(π * z)
    V = cos(π * t) * ρ * sin(π * x) * cos(π * y) * cos(π * z)
    W = cos(π * t) * ρ * sin(π * x) * cos(π * y) * sin(π * z)
    E = cos(π * t) * sin(π * x) * cos(π * y) * cos(π * z) + 100
  else
    ρ = cos(π * t) * sin(π * x) * cos(π * y) + 3
    U = cos(π * t) * ρ * sin(π * x) * cos(π * y)
    V = cos(π * t) * ρ * sin(π * x) * cos(π * y)
    W = cos(π * t) * 0
    E = cos(π * t) * sin(π * x) * cos(π * y) + 100
  end


  P = (γ-1)*(E - (U^2 + V^2 + W^2) / 2ρ)

  u, v, w = U / ρ, V / ρ, W / ρ

  dudx, dudy, dudz = diff(u, x), diff(u, y), diff(u, z)
  dvdx, dvdy, dvdz = diff(v, x), diff(v, y), diff(v, z)
  dwdx, dwdy, dwdz = diff(w, x), diff(w, y), diff(w, z)

  ϵ11 = dudx
  ϵ22 = dvdy
  ϵ33 = dwdz
  ϵ12 = (dudy + dvdx) / 2
  ϵ13 = (dudz + dwdx) / 2
  ϵ23 = (dvdz + dwdy) / 2

  τ11 = 2μ * (ϵ11 - (ϵ11 + ϵ22 + ϵ33) / 3)
  τ22 = 2μ * (ϵ22 - (ϵ11 + ϵ22 + ϵ33) / 3)
  τ33 = 2μ * (ϵ33 - (ϵ11 + ϵ22 + ϵ33) / 3)
  τ12 = τ21 = 2μ * ϵ12
  τ13 = τ31 = 2μ * ϵ13
  τ23 = τ32 = 2μ * ϵ23

  Fx_x = diff.([U;
                u * U + P - τ11;
                u * V     - τ12;
                u * W     - τ13;
                u * (E + P) - u * τ11 - v * τ12 - w * τ13], x)
  Fy_y = diff.([V;
                v * U     - τ21;
                v * V + P - τ22;
                v * W     - τ23;
                v * (E + P) - u * τ21 - v * τ22 - w * τ23], y)
  Fz_z = diff.([W;
                w * U     - τ31;
                w * V     - τ32;
                w * W + P - τ33;
                w * (E + P) - u * τ31 - v * τ32 - w * τ33], z)

  dρdt = simplify(Fx_x[1] + Fy_y[1] + Fz_z[1] + diff(ρ, t))
  dUdt = simplify(Fx_x[2] + Fy_y[2] + Fz_z[2] + diff(U, t))
  dVdt = simplify(Fx_x[3] + Fy_y[3] + Fz_z[3] + diff(V, t))
  dWdt = simplify(Fx_x[4] + Fy_y[4] + Fz_z[4] + diff(W, t))
  dEdt = simplify(Fx_x[5] + Fy_y[5] + Fz_z[5] + diff(E, t))


  @printf output "ρ_g(t, x, y, z, ::Val{%d}) = %s\n" dim ρ
  @printf output "U_g(t, x, y, z, ::Val{%d}) = %s\n" dim U
  @printf output "V_g(t, x, y, z, ::Val{%d}) = %s\n" dim V
  @printf output "W_g(t, x, y, z, ::Val{%d}) = %s\n" dim W
  @printf output "E_g(t, x, y, z, ::Val{%d}) = %s\n" dim E
  @printf output "Sρ_g(t, x, y, z, ::Val{%d}) = %s\n" dim dρdt
  @printf output "SU_g(t, x, y, z, ::Val{%d}) = %s\n" dim dUdt
  @printf output "SV_g(t, x, y, z, ::Val{%d}) = %s\n" dim dVdt
  @printf output "SW_g(t, x, y, z, ::Val{%d}) = %s\n" dim dWdt
  @printf output "SE_g(t, x, y, z, ::Val{%d}) = %s\n" dim dEdt
end

close(output)
