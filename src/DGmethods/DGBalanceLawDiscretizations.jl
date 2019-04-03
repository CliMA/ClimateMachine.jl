module DGBalanceLawDiscretizations
using MPI
using ...Grids
using ...MPIStateArrays
using Documenter

export DGBalanceLaw

"""
    DGBalanceLaw(grid::DiscontinuousSpectralElementGrid,
                 nstate::Int,
                 gradstates::NTuple{X, Int} = (),
                 flux!::Function,
                 numericalflux!::Function)

Given a balance law for `nstate` fields of the form

   ``q_{,t} + Σ_{i=1,...d} F_{i,i} = s``

The flux function `F_{i}` can depend on the state `q`, gradient `q` for `j =
1,...,d`, time `t`, and a set of user defined "constant" state `ϕ`.

The flux functions `flux!` has syntax `flux!(F, Q, G, ϕ_c, ϕ_d, t)` where:
- `F` is an `MArray` of size `(d, nstate)` to be filled
- `Q` is the state to evaluate
- `G` is an array of size `(d, ngradstate)` for `Q` for `j = 1,...,d` for the
  subset of variables sepcified by `gradstates`
- `ϕ_c` is the user-defined constant state
- `ϕ_d` is the user-defined dynamic state
- `t` is the time

!!! todo

    Add docs for other arguments...

!!! todo

    Stil need to add
    - `bcfun!`
    - `source!`
    - `initϕ_c!`
    - `updateϕ_d!`
    - `gradnumericalflux!`?

"""
struct DGBalanceLaw
  grid::DiscontinuousSpectralElementGrid

  "number of state"
  nstate::Int

  "Tuple of states to take the gradient of"
  gradstates::Tuple

  "physical flux function"
  flux!::Function

  "physical flux function"
  numericalflux!::Function

  "storage for the grad"
  Qgrad::Union{Nothing, MPIStateArray}

end

function DGBalanceLaw(;grid, nstate, flux!, numericalflux!, gradstates=())
  ngradstate = length(gradstates)
  if ngradstate > 0
    topology = grid.topology
    Np = dofs_per_element(grid)
    h_vgeo = Array(grid.vgeo)
    DFloat = eltype(h_vgeo)
    DA = arraytype(grid)
    # TODO: Clean up this MPIStateArray interface...
    Qgrad = MPIStateArray{Tuple{Np, ngradstate}, DFloat, DA
                         }(topology.mpicomm,
                           length(topology.elems),
                           realelems=topology.realelems,
                           ghostelems=topology.ghostelems,
                           sendelems=topology.sendelems,
                           nabrtorank=topology.nabrtorank,
                           nabrtorecv=topology.nabrtorecv,
                           nabrtosend=topology.nabrtosend,
                           weights=view(h_vgeo, :, grid.Mid, :))
  else
    Qgrad = nothing
  end

  DGBalanceLaw(grid, nstate, gradstates, flux!, numericalflux!, Qgrad)
end

end
