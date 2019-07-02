module SubgridScaleTurbulence

# Module dependencies
using CLIMA.PlanetParameters: grav, cp_d, cv_d

# Module exported functions 
export compute_strainrate_tensor
export standard_smagorinsky
export buoyancy_correction

  """
  This module addresses §4 (Subgrid-Scale Models) of the CLiMA-atmos documentation.
  
  We define the strain-rate tensor in terms of the velocity gradient components
  ϵ = (∇u) + (∇u)ᵀ where ᵀ represents the transpose operator. 

  
  Model constants.
  C_ss takes typical values of 0.14 - 0.23 (flow dependent empirical coefficient) 

  article{doi:10.1063/1.869251,
  author = {Canuto,V. M.  and Cheng,Y. },
  title = {Determination of the Smagorinsky–Lilly constant CS},
  journal = {Physics of Fluids},
  volume = {9},
  number = {5},
  pages = {1368-1378},
  year = {1997},
  doi = {10.1063/1.869251},
  URL = {https://doi.org/10.1063/1.869251},
  eprint = {https://doi.org/10.1063/1.869251}
  }
  """
  const C_ss = 0.23
  const Prandtl_turb = 1 // 3
  const Prandtl = 71 // 100

  """
  Smagorinsky model coefficient for anisotropic grids.
  Given a description of the grid in terms of Δ1, Δ2, Δ3
  and polynomial order Npoly, computes the anisotropic equivalent grid
  coefficient such that the Smagorinsky coefficient is modified as follows

  Eddy viscosity          ν_e
  Smagorinsky coefficient C_ss
  Δeq                     Equivalent anisotropic grid

  ν_e = (C_ss Δeq)^2 * sqrt(2 * SijSij) 
  
  @article{doi:10.1063/1.858537,
  author = {Scotti,Alberto  and Meneveau,Charles  and Lilly,Douglas K. },
  title = {Generalized Smagorinsky model for anisotropic grids},
    journal = {Physics of Fluids A: Fluid Dynamics},
    volume = {5},
    number = {9},
    pages = {2306-2308},
    year = {1993},
    doi = {10.1063/1.858537},
    URL = {https://doi.org/10.1063/1.858537},
    eprint = {https://doi.org/10.1063/1.858537}
  }

  In addition, simple alternative methods of computing the geometric average
  are also included (in accordance with Deardorff's methods).
  """
  function anisotropic_coefficient_sgs3D(Δ1, Δ2, Δ3)
    # Arguments are the lengthscales in each of the coordinate directions
    # For a cube: this is the edge length
    # For a sphere: the arc length provides one approximation of many
    Δ = cbrt(Δ1 * Δ2 * Δ3)
    Δ_sorted = sort([Δ1, Δ2, Δ3])  
    # Get smallest two dimensions
    Δ_s1 = Δ_sorted[1]
    Δ_s2 = Δ_sorted[2]
    a1 = Δ_s1 / max(Δ1,Δ2,Δ3) 
    a2 = Δ_s2 / max(Δ1,Δ2,Δ3) 
    # In 3D we compute a scaling factor for anisotropic grids
    f_anisotropic = 1 + 2/27 * ((log(a1))^2 - log(a1)*log(a2) + (log(a2))^2)
    Δ = Δ*f_anisotropic
    Δsqr = Δ * Δ
    return Δsqr
  end
  
  function anisotropic_coefficient_sgs2D(Δ1, Δ3)
    # Order of arguments does not matter.
    Δ = min(Δ1, Δ3)
    Δsqr = Δ * Δ
    return Δsqr
  end
  
  function standard_coefficient_sgs3D(Δ1,Δ2,Δ3)
    Δ = cbrt(Δ1 * Δ2 * Δ3) 
    Δsqr = Δ * Δ
    return Δsqr
  end
  
  function standard_coefficient_sgs2D(Δ1, Δ2)
    Δ = sqrt(Δ1 * Δ2)
    Δsqr = Δ * Δ
    return Δsqr
  end
  
  """
  Compute components of strain-rate tensor 
  Dij = ∇u .................................................. [1]
  Sij = 1/2 (∇u + (∇u)ᵀ) .....................................[2]
  τij = 2 * ν_e * Sij ........................................[3]
  """
  function compute_strainrate_tensor(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
    # Assemble components of the strain-rate tensor 
    S11, = dudx
    S12  = (dudy + dvdx) / 2
    S13  = (dudz + dwdx) / 2
    S22  = dvdy
    S23  = (dvdz + dwdy) / 2
    S33  = dwdz
    SijSij = S11^2 + S22^2 + S33^2 + 2 * (S12^2 + S13^2 + S23^2)  
    return (S11, S22, S33, S12, S13, S23, SijSij)
  end

  """
  Smagorinksy-Lilly SGS Turbulence
  --------------------------------
  The constant coefficient Standard Smagorinsky Model model for 
  (1) eddy viscosity ν_e 
  (2) and eddy diffusivity D_e 

  The resolved scale stress tensor is calculated as in [3]
  where Sij represents the components of the resolved
  scale rate of strain tensor. ν_t is the unknown eddy
  viscosity which is computed here using the assumption
  that subgrid turbulence production and dissipation are 
  balanced.

  article{doi:10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2,
  author = {Smagorinksy, J.},
  title = {General circulation experiments with the primitive equations},
  journal = {Monthly Weather Review},
  volume = {91},
  number = {3},
  pages = {99-164},
  year = {1963},
  doi = {10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2},
  URL = {https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2},
  eprint = {https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2}
  }
  """
  function standard_smagorinsky(SijSij, Δsqr)
    # Eddy viscosity is a function of the magnitude of the strain-rate tensor
    # This is for use on both spherical and cartesian grids. 
    ν_e::eltype(SijSij) = sqrt(2.0 * SijSij) * C_ss * C_ss * Δsqr
    D_e::eltype(SijSij) = ν_e / Prandtl_turb 
    return (ν_e, D_e)
  end

  """
  Buoyancy adjusted Smagorinsky coefficient for stratified flows
  
  Ri = N² / (2*SijSij)
  Ri = gravity / ρ * ∂ρ∂z / 2 |S_{ij}|
  article{doi:10.1111/j.2153-3490.1962.tb00128.x,
  author = {LILLY, D. K.},
  title = {On the numerical simulation of buoyant convection},
  journal = {Tellus},
  volume = {14},
  number = {2},
  pages = {148-172},
  doi = {10.1111/j.2153-3490.1962.tb00128.x},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/j.2153-3490.1962.tb00128.x},
  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.2153-3490.1962.tb00128.x},
  year = {1962}
  }
  """
  function buoyancy_correction(SijSij, ρ, dρdz)
    N2 = grav / ρ * dρdz 
    Richardson = N2 / (2 * SijSij + eps(SijSij))
    buoyancy_factor = N2 <=0 ? 1 : sqrt(max(0.0, 1 - Richardson/Prandtl_turb))
    return buoyancy_factor
  end

end
