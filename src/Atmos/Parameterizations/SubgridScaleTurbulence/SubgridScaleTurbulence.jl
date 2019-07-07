"""
  Parameterisations and models for subgrid scale turbulent stresses. Based on 
  physical energy cascade (Alternatives not included here apply 
  artificial hyperviscosity for stabilisation)

Scope of module 
---------------
This module addresses §4 (Subgrid-Scale Models) of the CLiMA-atmos documentation.
(1-4) are eddy-viscosity, eddy-diffusivity models. Structure function based models
are presently not considered. Tools for computing local lengthscales based on pointwise
inputs by the user are included. A density stratification correction is also included. 

1) Standard Smagorinsky Model (SSM) 
2) Anisotropic Smagorinsky Model (ASM)
Future additions
3) Anisotropic Minimum Dissipation (AMD)
4) Residual-based Dynamic SGS Model (DYN-SGS)

"""

module SubgridScaleTurbulence

# Module dependencies
using CLIMA.PlanetParameters: grav, cp_d, cv_d
using ..SubgridScaleParameters

# Module exported functions 
export anisotropic_lengthscale_3D
export anisotropic_lengthscale_2D
export geo_mean_lengthscale_3D
export geo_mean_lengthscale_2D
export strainrate_tensor_components
export anisotropic_smagorinsky
export standard_smagorinsky
export buoyancy_correction

  """
  
  We define the strain-rate tensor in terms of the velocity gradient components
  ϵ = (∇u) + (∇u)ᵀ where ᵀ represents the transpose operator. 

  
  Model constants.
  C_smag takes typical values of 0.14 - 0.23 (flow dependent empirical coefficient) 

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
  const Prandtl = 71 // 100

  """
    anisotropic_lengthscale_3D(Δ1, Δ2, Δ3) 
    return Δ², the equivalent anisotropic lengthscale squared

  Given a description of the grid in terms of three lengthscales Δ1,Δ2,Δ3,
  computes the anisotropic equivalent grid coefficient described by Scotti et. al.
  for the generalised Smagorinsky model for anisotropic grids. 
  
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
  """
  function anisotropic_lengthscale_3D(Δ1, Δ2, Δ3)
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
  
  """
    anisotropic_lengthscale_2D(Δ1, Δ2)
    return Δ², the equivalent anisotropic lengthscale squared

  For a 2-D problem, compute the anisotropic length-scale
  given the two local grid dimensions. For example, edge length
  for cube meshes and arc length for curved grids. The local element
  length can be computed within the auxiliary state initialisation kernel
  and passed to the auxiliary state in a manner similar to the coordinate
  terms.
  """

  function anisotropic_lengthscale_2D(Δ1, Δ2)
    # Order of arguments does not matter.
    Δ = min(Δ1, Δ2)
    Δsqr = Δ * Δ
    return Δsqr
  end
  
  """
    geo_mean_lengthscale_3D(Δ1, Δ2, Δ3)
    return Δ², the geometric mean lengthscale squared

  For a 3-D problem, compute the standard, geometric mean
  lengthscale based on the three coordinate dimensions 
  Δ1, Δ2, Δ3. The local element length can be computed 
  within the auxiliary state initialisation kernel
  by passing in the metric terms as arguments and 
  stored therein for use throughout the driver.
  """
  function standard_lengthscale_3D(Δ1,Δ2,Δ3)
    Δ = cbrt(Δ1 * Δ2 * Δ3) 
    Δsqr = Δ * Δ
    return Δsqr
  end
  
  """
    geo_mean_lengthscale_2D(Δ1, Δ2)
    return Δ², the geometric mean lengthscale squared

  For a 2-D problem, compute the standard, geometric mean
  lengthscale based on the three coordinate dimensions 
  Δ1, Δ2, Δ3. The local element length can be computed 
  within the auxiliary state initialisation kernel
  by passing in the metric terms as arguments and 
  stored therein for use throughout the driver.
  """
  function geo_mean_lengthscale_2D(Δ1, Δ2)
    Δ = sqrt(Δ1 * Δ2)
    Δsqr = Δ * Δ
    return Δsqr
  end

  """
    strainrate_tensor_components(dudx, dudy, dudz, dvdx, dvdy, dvdz [, dwdx, dwdy, dwdz])
    return (S11, S22, S33, S12, S13, S23, SijSij)
    Sij are components of the strain-rate tensor, and SijSij is the tensor-inner-product

  Compute components of strain-rate tensor from velocity gradient terms provided in 
  driver. Note that the gradient terms are computed in Cartesian space even on the spherical
  grid, with velocities and other key variables projected onto spherical coordinates for 
  visualisation 

  Dij = ∇u .................................................. [1]
  Sij = 1/2 (∇u + (∇u)ᵀ) .....................................[2]
  τij = 2 * ν_e * Sij ........................................[3]
  """
  function strainrate_tensor_components(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
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
    standard_smagorinsky(SijSij, Δsqr)
    return (ν_e, D_e), the eddy viscosity and diffusivity respectively

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
    DF = eltype(SijSij)
    ν_e::DF = sqrt(2.0 * SijSij) * C_smag^2 * Δsqr
    D_e::DF = ν_e / Prandtl_turb 
    return (ν_e, D_e)
  end

  """
    anisotropic_smagorinsky(SijSij, Δ1, Δ2[, Δ3=0])
    return (ν_1, ν_2, ν_3, D_1, D_2, D_3), 
    directional viscosity and diffusivity

  Simple extension of the Standard Smagorinsky Model to accommodate
  anisotropic viscosity dependent on the characterstic lengthscale
  in each coordinate direction 
  """
  function anisotropic_smagorinsky(SijSij, Δ1, Δ2, Δ3=0)
    # Order of arguments is irrelevant as long as self-consistency
    # with governing equations is maintained.
    DF = eltype(SijSij)
    ν_1::DF = sqrt(2.0 * SijSij) * C_smag^2 * Δ1^2
    ν_2::DF = sqrt(2.0 * SijSij) * C_smag^2 * Δ2^2
    ν_3::DF = sqrt(2.0 * SijSij) * C_smag^2 * Δ3^2
    D_1::DF = ν_1 / Prandtl_turb 
    D_2::DF = ν_2 / Prandtl_turb 
    D_3::DF = ν_3 / Prandtl_turb 
    return (ν_1, ν_2, ν_3, D_1, D_2, D_3)
  end
  
  """
    buoyancy_correction(SijSij, ρ, dρdz)
    return buoyancy_factor, scaling coefficient for Standard Smagorinsky Model
    in stratified flows

  Compute the buoyancy adjustment coefficient for stratified flows 
  given the strain rate tensor inner product SijSij, local density 
  ρ and the vertical density gradient dρdz

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
    # Brunt-Vaisala frequency
    N2 = grav / ρ * dρdz 
    # Richardson number
    Richardson = N2 / (2 * SijSij + eps(SijSij))
    # Buoyancy correction factor
    buoyancy_factor = N2 <=0 ? 1 : sqrt(max(0.0, 1 - Richardson/Prandtl_turb))
    return buoyancy_factor
  end

end
