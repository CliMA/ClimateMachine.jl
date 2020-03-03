module BrickMesh

export brickmesh, centroidtocode, connectmesh, partition, mappings, commmapping

using MPI

"""
    linearpartition(n, p, np)

Partition the range `1:n` into `np` pieces and return the `p`th piece as a
range.

This will provide an equal partition when `n` is divisible by `np` and
otherwise the ranges will have lengths of either `floor(Int, n/np)` or
`ceil(Int, n/np)`.
"""
linearpartition(n, p, np) = range(div((p-1)*n, np) + 1, stop=div(p*n, np))

"""
    hilbertcode(Y::AbstractArray{T}; bits=8sizeof(T)) where T

Given an array of axes coordinates `Y` stored as `bits`-bit integers
the function returns the Hilbert integer `H`.

The encoding of the Hilbert integer is best described by example.
If 5-bits are used from each of 3 coordinates then the function performs

     X[2]|                       H[0] = A B C D E
         | /X[1]       ------->  H[1] = F G H I J
    axes |/                      H[2] = K L M N O
         0------ X[0]                   high low

where the 15-bit Hilbert integer = `A B C D E F G H I J K L M N O` is stored
in `H`

This function is based on public domain code from John Skilling which can be
found in <https://doi.org/10.1063/1.1751381>.
"""
function hilbertcode(Y::AbstractArray{T}; bits=8sizeof(T)) where T
  # Below is Skilling's AxestoTranspose
  X = deepcopy(Y)
  n = length(X)
  M = one(T) << (bits-1)

  Q = M
  for j = 1:bits-1
    P = Q - one(T)
    for i = 1:n
      if X[i] & Q != zero(T)
        X[1] ⊻= P
      else
        t = (X[1] ⊻ X[i]) & P
        X[1] ⊻= t
        X[i] ⊻= t
      end
    end
    Q >>>= one(T)
  end

  for i = 2:n
    X[i] ⊻= X[i - 1]
  end

  t = zero(T)
  Q = M
  for j = 1:bits-1
    if X[n] & Q != zero(T)
      t ⊻= Q - one(T)
    end
    Q >>>= one(T)
  end

  for i = 1:n
    X[i] ⊻= t
  end

  # Below we transpose X and store it in H, i.e.:
  #
  #   X[0] = A D G J M               H[0] = A B C D E
  #   X[1] = B E H K N   <------->   H[1] = F G H I J
  #   X[2] = C F I L O               H[2] = K L M N O
  #
  # The 15-bit Hilbert integer is then = A B C D E F G H I J K L M N O
  H = zero(X)
  for i = 0:n-1, j = 0:bits-1
    k = i * bits + j
    bit = (X[n - mod(k,n)] >>> div(k,n)) & one(T)
    H[n - i] |= (bit << j)
  end

  return H
end

"""
    centroidtocode(comm::MPI.Comm, elemtocorner; coortocode, CT)

Returns a code for each element based on its centroid.

These element codes can be used to determine a linear ordering for the
partition function.

The communicator `comm` is used to calculate the bounding box for representing
the centroids in coordinates of type `CT`, defaulting to `CT=UInt64`.  These
integer coordinates are converted to a code using the function `coortocode`,
which defaults to `hilbertcode`.

The array containing the element corner coordinates, `elemtocorner`, is used
to compute the centroids.  `elemtocorner` is a dimension by number of corners
by number of elements array.
"""
function centroidtocode(comm::MPI.Comm, elemtocorner; coortocode=hilbertcode,
                        CT=UInt64)
  (d, nvert, nelem) = size(elemtocorner)

  centroids = sum(elemtocorner, dims=2) ./ nvert
  T = eltype(centroids)

  centroidmin = (nelem > 0) ? minimum(centroids, dims=3) : fill(typemax(T),d)
  centroidmax = (nelem > 0) ? maximum(centroids, dims=3) : fill(typemin(T),d)

  centroidmin = MPI.Allreduce(centroidmin, MPI.MIN, comm)
  centroidmax = MPI.Allreduce(centroidmax, MPI.MAX, comm)
  centroidsize = centroidmax - centroidmin

  # Fix centroidsize to be nonzero.  It can be zero for a couple of reasons.
  # For example, it will be zero if we have just one element.
  if iszero(centroidsize)
    centroidsize = ones(T, d)
  else
    for i = 1:d
      if iszero(centroidsize[i])
        centroidsize[i] = maximum(centroidsize)
      end
    end
  end

  code = Array{CT}(undef, d, nelem)
  for e = 1:nelem
    c = (centroids[:,1,e] .- centroidmin) ./ centroidsize
    X = CT.(floor.(typemax(CT).*BigFloat.(c, 16sizeof(CT))))
    code[:,e] = coortocode(X)
  end

  code
end

"""
    brickmesh(x, periodic; part=1, numparts=1; boundary)

Generate a brick mesh with coordinates given by the tuple `x` and the
periodic dimensions given by the `periodic` tuple.

The brick can optionally be partitioned into `numparts` and this returns
partition `part`.  This is a simple Cartesian partition and further
partitioning (e.g, based on a space-filling curve) should be done before the
mesh is used for computation.

By default boundary faces will be marked with a one and other faces with a
zero.  Specific boundary numbers can also be passed for each face of the brick
in `boundary`.  This will mark the nonperiodic brick faces with the given
boundary number.

# Examples

We can build a 3 by 2 element two-dimensional mesh that is periodic in the
\$x_2\$-direction with
```jldoctest brickmesh
julia> (elemtovert, elemtocoord, elemtobndy, faceconnections) =
        brickmesh((2:5,4:6), (false,true); boundary=((1,2), (3,4)));
```
This returns the mesh structure for

             x_2

              ^
              |
             6-  9----10----11----12
              |  |     |     |     |
              |  |  4  |  5  |  6  |
              |  |     |     |     |
             5-  5-----6-----7-----8
              |  |     |     |     |
              |  |  1  |  2  |  3  |
              |  |     |     |     |
             4-  1-----2-----3-----4
              |
              +--|-----|-----|-----|--> x_1
                 2     3     4     5

The (number of corners by number of elements) array `elemtovert` gives the
global vertex number for the corners of each element.
```jldoctest brickmesh
julia> elemtovert
4×6 Array{Int64,2}:
 1  2  3   5   6   7
 2  3  4   6   7   8
 5  6  7   9  10  11
 6  7  8  10  11  12
```
Note that the vertices are listed in Cartesian order.

The (dimension by number of corners by number of elements) array `elemtocoord`
gives the coordinates of the corners of each element.
```jldoctes brickmesh
julia> elemtocoord
2×4×6 Array{Int64,3}:
[:, :, 1] =
 2  3  2  3
 4  4  5  5

[:, :, 2] =
 3  4  3  4
 4  4  5  5

[:, :, 3] =
 4  5  4  5
 4  4  5  5

[:, :, 4] =
 2  3  2  3
 5  5  6  6

[:, :, 5] =
 3  4  3  4
 5  5  6  6

[:, :, 6] =
 4  5  4  5
 5  5  6  6
```

The (number of faces by number of elements) array `elemtobndy` gives the
boundary number for each face of each element.  A zero will be given for
connected faces.
```jldoctest brickmesh
julia> elemtobndy
4×6 Array{Int64,2}:
 1  0  0  1  0  0
 0  0  2  0  0  2
 0  0  0  0  0  0
 0  0  0  0  0  0
```
Note that the faces are listed in Cartesian order.

Finally, the periodic face connections are given in `faceconnections` which is a
list of arrays, one for each connection.
Each array in the list is given in the format `[e, f, vs...]` where
 - `e`  is the element number;
 - `f`  is the face number; and
 - `vs` is the global vertices that face associated with.
I the example
```jldoctest brickmesh
julia> faceconnections
3-element Array{Array{Int64,1},1}:
 [4, 4, 1, 2]
 [5, 4, 2, 3]
 [6, 4, 3, 4]
```
we see that face `4` of element `5` is associated with vertices `[2 3]` (the
vertices for face `1` of element `2`).
"""
function brickmesh(x, periodic; part=1, numparts=1,
                   boundary=ntuple(j->(1,1), length(x)))
  if boundary isa Matrix
    boundary = tuple(mapslices(x -> tuple(x...), boundary, dims=1)...)
  end
  
  @assert length(x) == length(periodic)
  @assert length(x) >= 1
  @assert 1 <= part <= numparts

  T = promote_type(eltype.(x)...)
  d = length(x)
  nvert = 2^d
  nface = 2d

  nelemdim = length.(x).-1
  elemlocal = linearpartition(prod(nelemdim), part, numparts)

  elemtovert = Array{Int}(undef, nvert, length(elemlocal))
  elemtocoord = Array{T}(undef, d, nvert, length(elemlocal))
  elemtobndy = zeros(Int, nface, length(elemlocal))
  faceconnections = Array{Array{Int, 1}}(undef, 0)

  verts = LinearIndices(ntuple(j->1:length(x[j]), d))
  elems = CartesianIndices(ntuple(j->1:length(x[j])-1, d))

  p = reshape(1:nvert, ntuple(j->2, d))
  fmask = hcat((p[ntuple(j->(j==div(f-1,2)+1) ? (mod(f-1,2)+1:mod(f-1,2)+1) :
                         (:), d)...,][:]
                for f=1:nface)...)


  for (e, ec) = enumerate(elems[elemlocal])
    corners = CartesianIndices(ntuple(j->ec[j]:ec[j]+1, d))
    for (v, vc) = enumerate(corners)
      elemtovert[v,e] = verts[vc]

      for j = 1:d
        elemtocoord[j,v,e] = x[j][vc[j]]
      end
    end

    for i=1:d
      if !periodic[i] && ec[i]==1
        elemtobndy[2(i-1)+1,e] = boundary[i][1]
      end
      if !periodic[i] && ec[i]==nelemdim[i]
        elemtobndy[2(i-1)+2,e] = boundary[i][2]
      end
    end

    for i=1:d
      if periodic[i] && ec[i]==nelemdim[i]
        neighcorners = CartesianIndices(ntuple(j->(i==j) ?
                                               (1:2) : (ec[j]:ec[j]+1), d))
        push!(faceconnections,
              vcat(e, 2i, verts[neighcorners[fmask[:,2i-1]]]))
      end
    end
  end

  (elemtovert, elemtocoord, elemtobndy, faceconnections)
end

"""
    parallelsortcolumns(comm::MPI.Comm, A;
                        alg::Base.Sort.Algorithm=Base.Sort.DEFAULT_UNSTABLE,
                        lt=isless,
                        by=identity,
                        rev::Union{Bool,Nothing}=nothing)

Sorts the columns of the distributed matrix `A`.

See the documentation of `sort!` for a description of the keyword arguments.

This function assumes `A` has the same number of rows on each MPI rank but can
have a different number of columns.
"""
function parallelsortcolumns(comm::MPI.Comm, A;
                             alg::Base.Sort.Algorithm=Base.Sort.DEFAULT_UNSTABLE,
                             lt=isless,
                             by=identity,
                             rev::Union{Bool,Nothing}=nothing)

  m, n = size(A)
  T = eltype(A)

  csize = MPI.Comm_size(comm)
  crank = MPI.Comm_rank(comm)
  croot = 0

  A = sortslices(A, dims=2, alg=alg, lt=lt, by=by, rev=rev)

  npivots = clamp(n, 0, csize)
  pivots = T[A[i, div(n*p,npivots)+1] for i=1:m, p=0:npivots-1]
  pivotcounts = MPI.Allgather(Cint(length(pivots)), comm)
  pivots = MPI.Allgatherv(pivots, pivotcounts, comm)
  pivots = reshape(pivots, m, div(length(pivots),m))
  pivots = sortslices(pivots, dims=2, alg=alg, lt=lt, by=by, rev=rev)

  # if we don't have any pivots then we must have zero columns
  if size(pivots) == (m, 0)
    return A
  end

  pivots =
    [pivots[i, div(div(length(pivots),m)*r,csize)+1] for i=1:m, r=0:csize-1]

  cols = map(i->view(A,:,i), 1:n)
  sendstarts = [(i<=csize) ? (searchsortedfirst(cols, pivots[:,i], lt=lt,
                                                by=by, rev=rev)-1)*m+1 : n*m+1
                for i=1:csize+1]
  sendcounts = [Cint(sendstarts[i+1]-sendstarts[i]) for i=1:csize]

  B = []
  for r = 0:csize-1
    counts = MPI.Allgather(sendcounts[r+1], comm)
    c = MPI.Gatherv(view(A, sendstarts[r+1]:sendstarts[r+2]-1), counts, r,
                    comm)
    if r == crank
      B = c
    end
  end
  B = reshape(B, m, div(length(B),m))

  sortslices(B, dims=2, alg=alg, lt=lt, by=by, rev=rev)
end

"""
    getpartition(comm::MPI.Comm, elemtocode)

Returns an equally weighted partition of a distributed set of elements by
sorting their codes given in `elemtocode`.

The codes for each element, `elemtocode`, are given as an array with a single
entry per local element or as a matrix with a column for each local element.

The partition is returned as a tuple three parts:

 - `partsendorder`: permutation of elements into sending order
 - `partsendstarts`: start entries in the send array for each rank
 - `partrecvstarts`: start entries in the receive array for each rank

Note that both `partsendstarts` and `partrecvstarts` are of length
`MPI.Comm_size(comm)+1` where the last entry has the total number of elements
to send or receive, respectively.
"""
getpartition(comm::MPI.Comm, elemtocode::AbstractVector) =
  getpartition(comm, reshape(elemtocode, 1, length(elemtocode)))

function getpartition(comm::MPI.Comm, elemtocode::AbstractMatrix)
  (ncode, nelem) = size(elemtocode)

  csize = MPI.Comm_size(comm)
  crank = MPI.Comm_rank(comm)

  CT = eltype(elemtocode)

  A = CT[elemtocode;                                 # code
         collect(CT, 1:nelem)';                      # original element number
         fill(CT(MPI.Comm_rank(comm)), (1, nelem));  # original rank
         fill(typemax(CT), (1, nelem))]              # new rank
  m, n = size(A)

  # sort by just code
  A = parallelsortcolumns(comm, A)

  # count the distribution of A
  counts = MPI.Allgather(last(size(A)), comm)
  starts = ones(Int, csize+1)
  for i=1:csize
    starts[i+1] = counts[i] + starts[i]
  end

  # loop to determine new rank
  j = range(starts[crank+1], stop=starts[crank+2]-1)
  for r = 0:csize-1
    k = linearpartition(starts[end]-1, r+1, csize)
    o = intersect(k,j) .- (starts[crank+1]-1)
    A[ncode+3,o] .= r
  end

  # sort by original rank and code
  A = sortslices(A, dims=2, by=x->x[[ncode+2,(1:ncode)...]])

  # count number of elements that are going to be sent
  sendcounts = zeros(Cint, csize)
  for i = 1:last(size(A))
    sendcounts[A[ncode+2,i]+1] += m
  end
  sendstarts = ones(Int, csize+1)
  for i=1:csize
    sendstarts[i+1] = sendcounts[i] + sendstarts[i]
  end

  # communicate columns of A to original rank
  B = []
  for r = 0:csize-1
    rcounts = MPI.Allgather(sendcounts[r+1], comm)
    c = MPI.Gatherv(view(A, sendstarts[r+1]:sendstarts[r+2]-1), rcounts, r,
                    comm)
    if r == crank
      B = c
    end
  end
  B = reshape(B, m, div(length(B),m))

  # check to make sure we didn't drop any elements
  @assert nelem == n == size(B)[2]

  partsendcounts = zeros(Cint, csize)
  for i = 1:last(size(B))
    partsendcounts[B[ncode+3,i]+1] += 1
  end
  partsendstarts = ones(Int, csize+1)
  for i=1:csize
    partsendstarts[i+1] = partsendcounts[i] + partsendstarts[i]
  end

  partsendorder = Int.(B[ncode+1,:])

  partrecvcounts = Cint[]
  for r = 0:csize-1
    c = MPI.Gather(partsendcounts[r+1], r, comm)
    if r == crank
      partrecvcounts = c
    end
  end

  partrecvstarts = ones(Int, csize+1)
  for i=1:csize
    partrecvstarts[i+1] = partrecvcounts[i] + partrecvstarts[i]
  end

  partsendorder, partsendstarts, partrecvstarts
end

"""
    partition(comm::MPI.Comm, elemtovert, elemtocoord, elemtobndy,
              faceconnections)

This function takes in a mesh (as returned for example by `brickmesh`) and
returns a Hilbert curve based partitioned mesh.
"""
function partition(comm::MPI.Comm, elemtovert, elemtocoord, elemtobndy,
                   faceconnections, globord=[])
  (d, nvert, nelem) = size(elemtocoord)

  csize = MPI.Comm_size(comm)
  crank = MPI.Comm_rank(comm)

  nface = 2d
  nfacevert = 2^(d-1)

  # Here we expand the list of face connections into a structure that is easy
  # to partition.  The cost is extra memory transfer.  If this becomes a
  # bottleneck something more efficient may be implemented.
  #
  elemtofaceconnect = zeros(eltype(eltype(faceconnections)), nfacevert, nface,
                            nelem)
  for fc in faceconnections
    elemtofaceconnect[:,fc[2],fc[1]] = fc[3:end]
  end

  elemtocode = centroidtocode(comm, elemtocoord; CT=UInt64)
  sendorder, sendstarts, recvstarts = getpartition(comm, elemtocode)

  elemtovert = elemtovert[:,sendorder]
  elemtocoord = elemtocoord[:,:,sendorder]
  elemtobndy = elemtobndy[:,sendorder]
  elemtofaceconnect = elemtofaceconnect[:,:,sendorder]

  if !isempty(globord)
    globord = globord[sendorder]
  end 

  newelemtovert = []
  newelemtocoord = []
  newelemtobndy = []
  newelemtofaceconnect = []
  newglobord = []
  for r = 0:csize-1
    sendrange = sendstarts[r+1]:sendstarts[r+2]-1
    rcounts = MPI.Allgather(Cint(length(sendrange)), comm)

    netv = MPI.Gatherv(view(elemtovert, :, sendrange), rcounts.*Cint(nvert),
                       r, comm)

    netc = MPI.Gatherv(view(elemtocoord, :, :, sendrange),
                       rcounts.*Cint(d*nvert), r, comm)

    netb = MPI.Gatherv(view(elemtobndy, :, sendrange), rcounts.*Cint(nface),
                       r, comm)

    netfc = MPI.Gatherv(view(elemtofaceconnect, :, :, sendrange),
                        rcounts.*Cint(nfacevert*nface), r, comm)

    if !isempty(globord)
      netglobord = MPI.Gatherv(view(globord, sendrange), rcounts,
                       r, comm)
    end

    if r == crank
      newelemtovert = netv
      newelemtocoord = netc
      newelemtobndy = netb
      newelemtofaceconnect = netfc
      if !isempty(globord)
        newglobord = netglobord 
      end
    end
  end

  newnelem = recvstarts[end]-1
  newelemtovert = reshape(newelemtovert, nvert, newnelem)
  newelemtocoord = reshape(newelemtocoord, d, nvert, newnelem)
  newelemtobndy = reshape(newelemtobndy, nface, newnelem)
  newelemtofaceconnect = reshape(newelemtofaceconnect, nfacevert, nface,
                                 newnelem)

  # reorder local elements based on code of new elements
  A = UInt64[centroidtocode(comm, newelemtocoord; CT=UInt64);
             collect(1:newnelem)']
  A = sortslices(A, dims=2)
  newsortorder = view(A,d+1,:)

  newelemtovert = newelemtovert[:,newsortorder]
  newelemtocoord = newelemtocoord[:,:,newsortorder]
  newelemtobndy = newelemtobndy[:,newsortorder]
  newelemtofaceconnect = newelemtofaceconnect[:,:,newsortorder]

  newfaceconnections = similar(faceconnections, 0)
  for e = 1:newnelem, f = 1:nface
    if newelemtofaceconnect[1,f,e] > 0
      push!(newfaceconnections, vcat(e, f, newelemtofaceconnect[:,f,e]))
    end
  end

  if !isempty(globord)
    newglobord = newglobord[newsortorder]
  end

  (newelemtovert, newelemtocoord, newelemtobndy, newfaceconnections, newglobord)#sendorder)
end

"""
    minmaxflip(x, y)

Returns `x, y` sorted lowest to highest and a bool that indicates if a swap
was needed.
"""
minmaxflip(x, y) = y < x ? (y, x, true) : (x, y, false)

"""
    vertsortandorder(a)

Returns `(a)` and an ordering `o==0`.
"""
vertsortandorder(a) = ((a,), 1)

"""
    vertsortandorder(a, b)

Returns sorted vertex numbers `(a,b)` and an ordering `o` depending on the
order needed to sort the elements.  This ordering is given below including the
vetex ordering for faces.

    o=    0      1

        (a,b)  (b,a)

          a      b
          |      |
          |      |
          b      a
"""
function vertsortandorder(a, b)
  a, b, s1 = minmaxflip(a, b)
  o = s1 ? 2 : 1
  ((a, b), o)
end

"""
    vertsortandorder(a, b, c)

Returns sorted vertex numbers `(a,b,c)` and an ordering `o` depending on the
order needed to sort the elements.  This ordering is given below including the
vetex ordering for faces.

    o=     1         2         3         4         5         6

        (a,b,c)   (c,a,b)   (b,c,a)   (b,a,c)   (c,b,a)   (a,c,b)

          /c\\      /b\\      /a\\      /c\\      /a\\      /b\\
         /   \\    /   \\    /   \\    /   \\    /   \\    /   \\
        /a___b\\  /c___a\\  /b___c\\  /b___a\\  /c___b\\  /a___c\\
"""
function vertsortandorder(a, b, c)
  # Use a (Bose-Nelson Algorithm based) sorting network from
  # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
  b, c, s1 = minmaxflip(b, c)
  a, c, s2 = minmaxflip(a, c)
  a, b, s3 = minmaxflip(a, b)

  if     !s1 && !s2 && !s3
    o = 1
  elseif !s1 &&  s2 &&  s3
    o = 2
  elseif  s1 && !s2 &&  s3
    o = 3
  elseif !s1 && !s2 &&  s3
    o = 4
  elseif  s1 &&  s2 &&  s3
    o = 5
  elseif  s1 && !s2 && !s3
    o = 6
  else
    error("Problem finding vertex ordering $((a,b,c)) with flips
          $((s1,s2,s3))")
  end

  ((a, b, c), o)
end

"""
    vertsortandorder(a, b, c, d)

Returns sorted vertex numbers `(a,b,c,d)` and an ordering `o` depending on the
order needed to sort the elements.  This ordering is given below including the
vetex ordering for faces.

    o=   1      2      3      4      5      6      7      8

       (a,b,  (a,c,  (b,a,  (b,d,  (c,a,  (c,d,  (d,b,  (d,c,
        c,d)   b,d)   c,d)   a,c)   d,b)   a,b)   c,a)   b,a)

       c---d  b---d  c---d  a---c  d---b  a---b  c---a  b---a
       |   |  |   |  |   |  |   |  |   |  |   |  |   |  |   |
       a---b  a---c  b---a  b---d  c---a  c---d  d---b  d---c
"""
function vertsortandorder(a, b, c, d)
  # Use a (Bose-Nelson Algorithm based) sorting network from
  # <http://pages.ripco.net/~jgamble/nw.html> to sort the vertices.
  a, b, s1 = minmaxflip(a, b)
  c, d, s2 = minmaxflip(c, d)
  a, c, s3 = minmaxflip(a, c)
  b, d, s4 = minmaxflip(b, d)
  b, c, s5 = minmaxflip(b, c)

 if     !s1 && !s2 && !s3 && !s4 && !s5
   o = 1
 elseif !s1 && !s2 && !s3 && !s4 &&  s5
   o = 2
 elseif  s1 && !s2 && !s3 && !s4 && !s5
   o = 3
 elseif !s1 && !s2 &&  s3 &&  s4 &&  s5
   o = 4
 elseif  s1 &&  s2 && !s3 && !s4 &&  s5
   o = 5
 elseif !s1 && !s2 &&  s3 &&  s4 && !s5
   o = 6
 elseif  s1 &&  s2 &&  s3 &&  s4 &&  s5
   o = 7
 elseif  s1 &&  s2 &&  s3 &&  s4 && !s5
   o = 8
 else
    # FIXME: some possible orientations are missing since there are a total of
    # 24. Missing orientations:
    #=
       d---c  d---c  b---c  a---d  c---b  d---a  a---b  b---a
       |   |  |   |  |   |  |   |  |   |  |   |  |   |  |   |
       a---b  b---a  a---d  b---c  d---a  c---b  d---c  c---d

       c---b  d---b  b---d  b---c  c---a  d---a  a---d  a---c
       |   |  |   |  |   |  |   |  |   |  |   |  |   |  |   |
       a---d  a---c  c---a  d---a  b---d  b---c  c---b  d---b
    =#
    error("Problem finding vertex ordering $((a,b,c,d))
            with flips $((s1,s2,s3,s4,s5))")
 end

  ((a, b, c, d), o)
end

"""
    connectmesh(comm::MPI.Comm, elemtovert, elemtocoord, elemtobndy,
                faceconnections)

This function takes in a mesh (as returned for example by `brickmesh`) and
returns a connected mesh.  This returns a `NamedTuple` of:

 - `elems` the range of element indices
 - `realelems` the range of real (aka nonghost) element indices
 - `ghostelems` the range of ghost element indices
 - `ghostfaces` ghost element to face is received;
   `ghostfaces[f,ge] == true` if face `f` of ghost element `ge` is received.
 - `sendelems` an array of send element indices
 - `sendfaces` send element to face is sent;
   `sendfaces[f,se] == true` if face `f` of send element `se` is sent.
 - `elemtocoord` element to vertex coordinates; `elemtocoord[d,i,e]` is the
    `d`th coordinate of corner `i` of element `e`
 - `elemtoelem` element to neighboring element; `elemtoelem[f,e]` is the
   number of the element neighboring element `e` across face `f`.  If there is
   no neighboring element then `elemtoelem[f,e] == e`.
 - `elemtoface` element to neighboring element face; `elemtoface[f,e]` is the
   face number of the element neighboring element `e` across face `f`.  If
   there is no neighboring element then `elemtoface[f,e] == f`.
 - `elemtoordr` element to neighboring element order; `elemtoordr[f,e]` is the
   ordering number of the element neighboring element `e` across face `f`.  If
   there is no neighboring element then `elemtoordr[f,e] == 1`.
 - `elemtobndy` element to bounday number; `elemtobndy[f,e]` is the
   boundary number of face `f` of element `e`.  If there is a neighboring
   element then `elemtobndy[f,e] == 0`.
 - `nabrtorank` a list of the MPI ranks for the neighboring processes
 - `nabrtorecv` a range in ghost elements to receive for each neighbor
 - `nabrtosend` a range in `sendelems` to send for each neighbor

"""
function connectmesh(comm::MPI.Comm, elemtovert, elemtocoord, elemtobndy,
                     faceconnections; dim = size(elemtocoord,1))
  d = dim
  (coorddim, nvert, nelem) = size(elemtocoord)
  nface, nfacevert = 2d, 2^(d-1)

  p = reshape(1:nvert, ntuple(j->2, d))
  fmask = hcat((p[ntuple(j->(j==div(f-1,2)+1) ? (mod(f-1,2)+1:mod(f-1,2)+1) :
                         (:), d)...][:]
                for f=1:nface)...)

  csize = MPI.Comm_size(comm)
  crank = MPI.Comm_rank(comm)

  VT = eltype(elemtovert)
  A = Array{VT}(undef, nfacevert+8, nface*nelem)

  MR, ME, MF, MO, NR, NE, NF, NO = nfacevert .+ (1:8)
  for e = 1:nelem
    v = reshape(elemtovert[:,e], ntuple(j->2, d))
    for f = 1:nface
      j = (e-1)*nface + f
      fv, o = vertsortandorder(v[fmask[:,f]]...)
      A[1:nfacevert, j] .= fv
      A[MR, j] = crank
      A[ME, j] = e
      A[MF, j] = f
      A[MO, j] = o
      A[NR, j] = typemax(VT)
      A[NE, j] = typemax(VT)
      A[NF, j] = typemax(VT)
      A[NO, j] = typemax(VT)
    end
  end

  # use neighboring vertices for connected faces
  for fc in faceconnections
    e = fc[1]
    f = fc[2]
    v = fc[3:end]
    j = (e-1)*nface + f
    fv, o = vertsortandorder(v...)
    A[1:nfacevert, j] .= fv
    A[MO, j] = o
  end

  A = parallelsortcolumns(comm, A, by=x->x[1:nfacevert])
  m, n = size(A)

  # match faces
  j = 1
  while j <= n
    if j+1 <= n && A[1:nfacevert,j]==A[1:nfacevert,j+1]
      # found connected face
      A[NR:NO, j  ] = A[MR:MO, j+1]
      A[NR:NO, j+1] = A[MR:MO, j  ]
      j += 2
    else
      # found unconnect face
      A[NR:NO, j] = A[MR:MO, j]
      j += 1
    end
  end

  A = sortslices(A, dims=2, by=x->(x[MR],x[NR],x[ME],x[MF]))

  # count number of elements that are going to be sent
  sendcounts = zeros(Cint, csize)
  for i = 1:last(size(A))
    sendcounts[A[MR,i]+1] += m
  end
  sendstarts = ones(Int, csize+1)
  for i=1:csize
    sendstarts[i+1] = sendcounts[i] + sendstarts[i]
  end

  # communicate columns of A to original rank
  B = []
  for r = 0:csize-1
    rcounts = MPI.Allgather(sendcounts[r+1], comm)
    c = MPI.Gatherv(view(A, sendstarts[r+1]:sendstarts[r+2]-1), rcounts, r,
                    comm)
    if r == crank
      B = c
    end
  end
  B = reshape(B, m, nface*nelem)

  # get element sending information
  B = sortslices(B, dims=2, by=x->(x[NR],x[ME]))
  sendelems = Int[]
  counts = zeros(Int, csize+1)
  counts[1] = (last(size(B)) > 0) ? 1 : 0
  sr, se = -1, 0
  for i = 1:last(size(B))
    r, e = B[NR,i], B[ME,i]
    # See if we need to send element `e` to rank `r` and make sure that we
    # didn't already mark it for sending.
    if r != crank && !(sr == r && se == e)
      counts[r+2] += 1
      append!(sendelems, e)
      sr, se = r, e
    end
  end

  # Mark which faces need to be sent
  sendfaces = BitArray(undef, nface, length(sendelems))
  sendfaces .= false
  sr, se, n = -1, 0, 0
  for i = 1:last(size(B))
    r, e, f = B[NR,i], B[ME,i], B[MF,i]
    if r != crank
      if !(sr == r && se == e)
        n += 1
        sr, se = r, e
      end
      sendfaces[f, n] = true
    end
  end

  sendstarts = cumsum(counts)
  nabrtosendrank = Int[r for r = 0:csize-1
                       if sendstarts[r+2]-sendstarts[r+1] > 0]
  nabrtosend = UnitRange{Int}[(sendstarts[r+1]:(sendstarts[r+2]-1))
                              for r = 0:csize-1
                              if sendstarts[r+2]-sendstarts[r+1] > 0]

  # get element receiving information
  B = sortslices(B, dims=2, by=x->(x[NR],x[NE]))
  counts = zeros(Int, csize+1)
  counts[1] = (last(size(B)) > 0) ? 1 : 0
  sr, se = -1, 0
  nghost = 0
  for i = 1:last(size(B))
    r, e = B[NR,i], B[NE,i]
    if r != crank
      # Check to make sure we have not already marked the element for
      # receiving since we could be connected to the receiving element across
      # multiple faces.
      if !(sr == r && se == e)
        nghost += 1
        counts[r+2] += 1
        sr, se = r, e
      end
      B[NE,i] = nelem + nghost
    end
  end

  # Mark which faces will be received
  ghostfaces = BitArray(undef, nface, nghost)
  ghostfaces .= false
  sr, se, ge = -1, 0, 0
  for i = 1:last(size(B))
    r, e, f = B[NR,i], B[NE,i], B[NF,i]
    if r != crank
      if !(sr == r && se == e)
        ge += 1
        sr, se = r, e
      end
      B[NR,i] = crank
      ghostfaces[f, ge] = true
    end
  end

  recvstarts = cumsum(counts)
  nabrtorecvrank = Int[r for r = 0:csize-1
                       if recvstarts[r+2]-recvstarts[r+1] > 0]
  nabrtorecv = UnitRange{Int}[(recvstarts[r+1]:(recvstarts[r+2]-1))
                              for r = 0:csize-1
                              if recvstarts[r+2]-recvstarts[r+1] > 0]

  @assert nabrtorecvrank == nabrtosendrank
  nabrtorank = nabrtorecvrank

  elemtoelem = repeat((1:nelem+nghost)', nface, 1)
  elemtoface = repeat(1:nface, 1, nelem+nghost)
  elemtoordr = ones(Int, nface, nelem+nghost)

  if d == 2
    for i = 1:last(size(B))
      me, mf, mo = B[ME,i], B[MF,i], B[MO,i]
      ne, nf, no = B[NE,i], B[NF,i], B[NO,i]

      elemtoelem[mf,me] = ne
      elemtoface[mf,me] = nf
      elemtoordr[mf,me] = (no == mo ? 1 : 2)
    end
  else
    for i = 1:last(size(B))
      me, mf, mo = B[ME,i], B[MF,i], B[MO,i]
      ne, nf, no = B[NE,i], B[NF,i], B[NO,i]

      elemtoelem[mf,me] = ne
      elemtoface[mf,me] = nf
      if no != 1 || mo != 1
        error("TODO add support for other orientations")
      end
      elemtoordr[mf,me] = 1
    end
  end

  # fill the ghost values in elemtocoord
  newelemtocoord = similar(elemtocoord, coorddim, nvert, nelem+nghost)
  newelemtobndy = similar(elemtobndy, nface, nelem+nghost)

  sendelemtocoord = elemtocoord[:,:,sendelems]
  sendelemtobndy = elemtobndy[:,sendelems]

  crreq = [MPI.Irecv!(view(newelemtocoord,:,:,nelem.+er), r, 666, comm)
           for (r, er) = zip(nabrtorank, nabrtorecv)]

  brreq = [MPI.Irecv!(view(newelemtobndy,:,nelem.+er), r, 666, comm)
           for (r, er) = zip(nabrtorank, nabrtorecv)]

  csreq = [MPI.Isend(view(sendelemtocoord,:,:,es), r, 666, comm)
          for (r, es) = zip(nabrtorank, nabrtosend)]

  bsreq = [MPI.Isend(view(sendelemtobndy,:,es), r, 666, comm)
          for (r, es) = zip(nabrtorank, nabrtosend)]

  newelemtocoord[:, :,1:nelem] .= elemtocoord
  newelemtobndy[:,1:nelem] .= elemtobndy

  MPI.Waitall!([csreq;crreq;bsreq;brreq])

  (elems=1:(nelem+nghost),       # range of          element indices
   realelems=1:nelem,            # range of real     element indices
   ghostelems=nelem.+(1:nghost), # range of ghost    element indices
   ghostfaces=ghostfaces,
   sendelems=sendelems,          # array of send     element indices
   sendfaces=sendfaces,
   elemtocoord=newelemtocoord,   # element to vertex coordinates
   elemtoelem=elemtoelem,        # element to neighboring element
   elemtoface=elemtoface,        # element to neighboring element face
   elemtoordr=elemtoordr,        # element to neighboring element order
   elemtobndy=newelemtobndy,     # element to boundary number
   nabrtorank=nabrtorank,        # list of neighboring processes MPI ranks
   nabrtorecv=nabrtorecv,        # neighbor receive ranges into `ghostelems`
   nabrtosend=nabrtosend)        # neighbor send ranges into `sendelems`
end

"""
    mappings(N, elemtoelem, elemtoface, elemtoordr)

This function takes in a polynomial order `N` and parts of a mesh (as returned
from `connectmesh`) and returns index mappings for the element surface flux
computation.  The returned `Tuple` contains:

 - `vmap⁻` an array of linear indices into the volume degrees of freedom where
   `vmap⁻[:,f,e]` are the degrees of freedom indices for face `f` of element
    `e`.

 - `vmap⁺` an array of linear indices into the volume degrees of freedom where
   `vmap⁺[:,f,e]` are the degrees of freedom indices for the face neighboring
   face `f` of element `e`.
"""
function mappings(N, elemtoelem, elemtoface, elemtoordr)
  nface, nelem = size(elemtoelem)

  d = div(nface, 2)
  Np, Nfp = (N+1)^d, (N+1)^(d-1)

  p = reshape(1:Np, ntuple(j->N+1, d))
  fd(f) =   div(f-1,2)+1
  fe(f) = N*mod(f-1,2)+1
  fmask = hcat((p[ntuple(j->(j==fd(f)) ? (fe(f):fe(f)) : (:), d)...][:]
                for f=1:nface)...)

  vmap⁻ = similar(elemtoelem, Nfp, nface, nelem)
  vmap⁺ = similar(elemtoelem, Nfp, nface, nelem)

  for e1 = 1:nelem, f1 = 1:nface
    e2 = elemtoelem[f1,e1]
    f2 = elemtoface[f1,e1]
    o2 = elemtoordr[f1,e1]

    # TODO support different orientations
    @assert o2 == 1

    vmap⁻[:,f1,e1] .= Np*(e1-1) .+ fmask[:,f1]
    vmap⁺[:,f1,e1] .= Np*(e2-1) .+ fmask[:,f2]
  end

  (vmap⁻, vmap⁺)
end

"""
   commmapping(N, commelems, commfaces, nabrtocomm)

This function takes in a polynomial order `N` and parts of a mesh (as returned
from `connectmesh` such as `sendelems`, `sendfaces`, and `nabrtosend`) and
returns index mappings for the element surface flux parallel communcation.
The returned `Tuple` contains:

 - `vmapC` an array of linear indices into the volume degrees of freedom to be
   communicated.

 - `nabrtovmapC` a range in `vmapC` to communicate with each neighbor.
"""
function commmapping(N, commelems, commfaces, nabrtocomm)
  nface, nelem = size(commfaces)

  @assert nelem == length(commelems)

  d = div(nface, 2)
  Nq = N+1
  Np = (N+1)^d

  vmapC = similar(commelems, nelem*Np)
  nabrtovmapC = similar(nabrtocomm)

  i = 1
  e = 1
  for neighbor in 1:length(nabrtocomm)
    rbegin = i
    for ne in nabrtocomm[neighbor]
      ce = commelems[ne]

      # Whole element sending
      # for n = 1:Np
      #   vmapC[i] = (ce-1)*Np + n
      #   i += 1
      # end

      CI = CartesianIndices(ntuple(_->1:Nq, d))
      for (ci, li) in zip(CI, LinearIndices(CI))
        addpoint = false
        for j = 1:d
          addpoint |= (commfaces[2*(j-1)+1, e] && ci[j] == 1 ) ||
                      (commfaces[2*(j-1)+2, e] && ci[j] == Nq)
        end

        if addpoint
          vmapC[i] = (ce-1)*Np + li
          i += 1
        end
      end

      e += 1
    end
    rend = i-1

    nabrtovmapC[neighbor] = rbegin:rend
  end

  resize!(vmapC, i-1)

  (vmapC, nabrtovmapC)
end

end # module
