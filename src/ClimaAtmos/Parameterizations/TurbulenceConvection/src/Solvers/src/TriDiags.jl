module TriDiags

export TriDiag

"""
Stores coefficients of a tri-diagonal matrix:

"""
struct TriDiag{T}
  L::Vector{T}
  D::Vector{T}
  U::Vector{T}
end

function TriDiag(N::Int, T)
  L = Vector{T}([0 for _ in 1:N-1])
  D = Vector{T}([0 for _ in 1:N  ])
  U = Vector{T}([0 for _ in 1:N-1])
  return TriDiag(L, D, U)
end

function assign_diag!(tridiag::TriDiag,
                      D::Vector{T}) where T
  tridiag.D[:] = D[:]
end

function add!(tridiag::TriDiag,
              L::Vector{T},
              D::Vector{T},
              U::Vector{T}
              ) where T
  tridiag.L[:] += L[:]
  tridiag.D[:] += D[:]
  tridiag.U[:] += U[:]
end

end