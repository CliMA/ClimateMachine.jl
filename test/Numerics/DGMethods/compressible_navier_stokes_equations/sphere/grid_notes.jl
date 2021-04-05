# the grid data structure contains 37 members
# (6) Surface structures
# sgeo, and the 5 ids
# (17) Volume structures
# vgeo, and the 16 ids
# (2) Volume to Surface index
# vmap⁻, vmap⁺
# (1) Topology
# topology
# (3) Operators
# grid.D (differentiation matrix), grid.Imat (integration matrix?), grid.ω (quadrature weights)
# information (1)
# activedofs (active degrees of freedom)
# Domain Boundary Information (1)
# elemtobndy (element to boundary)
# MPI stuff (6)
# interiorelems, exteriorelems, nabrtovmaprecv, nabrtovmapsend, vmaprecv, vmapsendv 

# sgeo, the first index is for the ids
# the second index are the gauss-lobatto entries
# the third index is the face
# the last index is the element
# the ids are: (1, n1id), (2 n2id), (3 n3id) (4, sMid), (5, vMIid)
# grid.sgeo[1,:,:,:] are normals in the n1 direction
# norm(grid.vgeo[:, grid.MIid, :][grid.vmap⁻] - grid.sgeo[5,:,:,:]) is zero
# x[grid.vmap⁻[10]] is connected to x[grid.vmap⁺[10]]

# vgeo, first index is guass-lobatto points
# second index is ids
# third index is elements
# the are are 16 ids
# 1-9 are metric terms
# 10 is the mass matrix
# 11 is the inverse mass matrix
# 12 is MHid (horizontal mass matrix?)
# 13-15 are x1 x2 x3
# 16 is the vertical volume jacobian
# grid.vgeo[:, grid.ξ1x1id, :]