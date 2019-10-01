#### UpdraftVars

export Cloud,
       SurfaceBC,
       UpdraftVar,
       compute_cloud_base_top_cover!

mutable struct Cloud{DT}
  base::DT
  top::DT
  cover::DT
end

mutable struct SurfaceBC{DT}
  a::DT
  w::DT
  θ_liq::DT
  q_tot::DT
end

struct UpdraftVar{DT}
  cloud::Cloud{DT}
  surface_bc::SurfaceBC{DT}
  surface_scalar_coeff::DT
  function UpdraftVar(i::I, surface_area::DT, n_updrafts::I) where {DT, I}
    a_s = surface_area/n_updrafts
    c = Cloud{DT}(0,0,0)
    s_bc = SurfaceBC{DT}(0,0,0,0)
    surface_scalar_coeff = 1.7072226094205676
    # TODO: FIXME:
    # surface_scalar_coeff = percentile_bounds_mean_norm(1-surface_area + i    *a_s,
    #                                                    1-surface_area + (i+1)*a_s , 1000)
    return new{DT}(c, s_bc, surface_scalar_coeff)
  end
end

function compute_cloud_base_top_cover!(UpdVar, grid::Grid{DT}, q::StateVec, tmp::StateVec) where DT
  gm, en, ud, sd, al = allcombinations(DomainIdx(q))
  k_2 = first_interior(grid, Zmax())
  @inbounds for i in ud
    UpdVar[i].cloud.base = grid.zc[k_2]
    UpdVar[i].cloud.top = DT(0)
    UpdVar[i].cloud.cover = DT(0)
    @inbounds for k in over_elems_real(grid)
      a_ik = q[:a, k, i]
      z_k = grid.zc[k]
      if tmp[:q_liq, k, i] > DT(1e-8) && a_ik > DT(1e-3)
        UpdVar[i].cloud.base  = min(UpdVar[i].cloud.base, z_k)
        UpdVar[i].cloud.top   = max(UpdVar[i].cloud.top, z_k)
        UpdVar[i].cloud.cover = max(UpdVar[i].cloud.cover, a_ik)
      end
    end
  end
end
