#### UpdraftVars

export Cloud,
       SurfaceBC,
       UpdraftVar,
       compute_cloud_base_top_cover!

mutable struct Cloud{FT}
  base::FT
  top::FT
  cover::FT
end

mutable struct SurfaceBC{FT}
  a::FT
  w::FT
  θ_liq::FT
  q_tot::FT
end

struct UpdraftVar{FT}
  cloud::Cloud{FT}
  surface_bc::SurfaceBC{FT}
  surface_scalar_coeff::FT
  function UpdraftVar(i::I, surface_area::FT, n_updrafts::I) where {FT, I}
    a_s = surface_area/FT(n_updrafts)
    c = Cloud{FT}(0,0,0)
    s_bc = SurfaceBC{FT}(0,0,0,0)
    surface_scalar_coeff = 1.7072226094205676
    # TODO: FIXME:
    # surface_scalar_coeff = percentile_bounds_mean_norm(1-surface_area + i    *a_s,
    #                                                    1-surface_area + (i+1)*a_s , 1000)
    return new{FT}(c, s_bc, surface_scalar_coeff)
  end
end

function compute_cloud_base_top_cover!(UpdVar, grid::Grid{FT}, q::StateVec, tmp::StateVec) where FT
  gm, en, ud, sd, al = allcombinations(q)
  k_2 = first_interior(grid, Zmax())
  @inbounds for i in ud
    UpdVar[i].cloud.base = grid.zc[k_2]
    UpdVar[i].cloud.top = FT(0)
    UpdVar[i].cloud.cover = FT(0)
    @inbounds for k in over_elems_real(grid)
      a_ik = q[:a, k, i]
      z_k = grid.zc[k]
      if tmp[:q_liq, k, i] > FT(1e-8) && a_ik > FT(1e-3)
        UpdVar[i].cloud.base  = min(UpdVar[i].cloud.base, z_k)
        UpdVar[i].cloud.top   = max(UpdVar[i].cloud.top, z_k)
        UpdVar[i].cloud.cover = max(UpdVar[i].cloud.cover, a_ik)
      end
    end
  end
end
