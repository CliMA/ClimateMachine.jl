####
#### Sponges
####

const (xmin, xmax) = (-30000,30000)
const (ymin, ymax) = (0,  5000)
const (zmin, zmax) = (0, 24000)

"""
    init_sponge!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
Compute sponge in aux state, for momentum equation.
"""
function init_sponge!(m::AtmosModel, aux::Vars, geom::LocalGeometry)
  DT = eltype(aux)
  #Sponge
  csleft  = DT(0)
  csright = DT(0)
  csfront = DT(0)
  csback  = DT(0)
  ctop    = DT(0)

  cs_left_right = DT(0.0)
  cs_front_back = DT(0.0)
  ct            = DT(0.9)

  #BEGIN  User modification on domain parameters.
  #Only change the first index of brickrange if your axis are
  #oriented differently:
  #x, y, z = aux[_a_x], aux[_a_y], aux[_a_z]
  #TODO z is the vertical coordinate
  #
  domain_left  = xmin
  domain_right = xmax

  domain_front = ymin
  domain_back  = ymax

  domain_bott  = zmin
  domain_top   = zmax

  #END User modification on domain parameters.
  z = aux.orientation.Φ/grav

  # Define Sponge Boundaries
  xc       = DT(0.5) * (domain_right + domain_left)
  yc       = DT(0.5) * (domain_back  + domain_front)
  zc       = DT(0.5) * (domain_top   + domain_bott)

  sponge_type = 2
  if sponge_type == 1

      bc_zscale   = DT(7000.0)
      top_sponge  = DT(0.85) * domain_top
      zd          = domain_top - bc_zscale
      xsponger    = domain_right - DT(0.15) * (domain_right - xc)
      xspongel    = domain_left  + DT(0.15) * (xc - domain_left)
      ysponger    = domain_back  - DT(0.15) * (domain_back - yc)
      yspongel    = domain_front + DT(0.15) * (yc - domain_front)

      #x left and right
      #xsl
      if x <= xspongel
          csleft = cs_left_right * (sinpi(1/2 * (x - xspongel)/(domain_left - xspongel)))^4
      end
      #xsr
      if x >= xsponger
          csright = cs_left_right * (sinpi(1/2 * (x - xsponger)/(domain_right - xsponger)))^4
      end
      #y left and right
      #ysl
      if y <= yspongel
          csfront = cs_front_back * (sinpi(1/2 * (y - yspongel)/(domain_front - yspongel)))^4
      end
      #ysr
      if y >= ysponger
          csback = cs_front_back * (sinpi(1/2 * (y - ysponger)/(domain_back - ysponger)))^4
      end

      #Vertical sponge:
      if z >= top_sponge
          ctop = ct * (sinpi(1/2 * (z - top_sponge)/(domain_top - top_sponge)))^4
      end

  elseif sponge_type == 2


      alpha_coe = DT(0.5)
      bc_zscale = DT(7500.0)
      zd        = domain_top - bc_zscale
      xsponger  = domain_right - DT(0.15) * (domain_right - xc)
      xspongel  = domain_left  + DT(0.15) * (xc - domain_left)
      ysponger  = domain_back  - DT(0.15) * (domain_back - yc)
      yspongel  = domain_front + DT(0.15) * (yc - domain_front)

      #
      # top damping
      # first layer: damp lee waves
      #
      ctop = DT(0.0)
      ct   = DT(0.5)
      if z >= zd
          zid = (z - zd)/(domain_top - zd) # normalized coordinate
          if zid >= 0.0 && zid <= 0.5
              abstaud = alpha_coe*(DT(1) - cos(zid*pi))
          else
              abstaud = alpha_coe*(DT(1) + cos((zid - 1/2)*pi) )
          end
          ctop = ct*abstaud
      end

  end #sponge_type

  beta  = DT(1) - (DT(1) - ctop) #*(1.0 - csleft)*(1.0 - csright)*(1.0 - csfront)*(1.0 - csback)
  aux.sponge  = min(beta, DT(1))
end
