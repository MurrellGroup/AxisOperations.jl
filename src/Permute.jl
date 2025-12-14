struct Permute{perm,N} <: GlobalAxisOp{N,N} end

function Permute(perm::NTuple{N,Int}) where N
    all(in(perm), 1:N) || throw(ArgumentError("Permutation must be a permutation of 1:N"))
    return Permute{perm,N}()
end

function (::Permute{perm,N})(x::AbstractArray{<:Any,N}) where {perm,N}
    return permutedims(x, perm)
end

@generated function (::Permute{perm2,N})(x::PermutedDimsArray{T,N,perm1,iperm1,P}) where {perm2,N,T,perm1,iperm1,P}
    perm_total = ntuple(i -> perm1[perm2[i]], Val(N))
    if perm_total == ntuple(identity, Val(N))
        return :(parent(x))
    end
    return :(PermutedDimsArray(parent(x), $(QuoteNode(perm_total))))
end

using LinearAlgebra: Transpose, Adjoint
const AdjOrTrans{T,P} = Union{Transpose{T,P},Adjoint{T,P}}

@generated function (::Permute{perm2,2})(x::AdjOrTrans{<:Real,P}) where {perm2,P}
    perm1 = (2, 1)
    perm_total = (perm1[perm2[1]], perm1[perm2[2]])
    issorted(perm_total) ? :(parent(x)) : :(x)
end
