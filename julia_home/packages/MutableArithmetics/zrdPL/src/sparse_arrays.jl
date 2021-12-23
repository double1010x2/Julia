import SparseArrays

const _SparseMat = SparseArrays.SparseMatrixCSC

function undef_array(
    ::Type{_SparseMat{Tv,Ti}},
    rows::Base.OneTo,
    cols::Base.OneTo,
) where {Tv,Ti}
    return SparseArrays.spzeros(Tv, Ti, length(rows), length(cols))
end

function operate!(::typeof(zero), A::_SparseMat)
    for i in eachindex(A.colptr)
        A.colptr[i] = one(A.colptr[i])
    end
    empty!(A.rowval)
    empty!(A.nzval)
    return A
end

function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{<:SparseArrays.AbstractSparseArray{Tv,Ti,N}},
    ::Type{Array{T,N}},
) where {Tv,Ti,T,N}
    return Array{promote_operation(op, Tv, T),N}
end
function promote_operation(
    op::Union{typeof(+),typeof(-)},
    ::Type{Array{T,N}},
    ::Type{<:SparseArrays.AbstractSparseArray{Tv,Ti,N}},
) where {Tv,Ti,T,N}
    return Array{promote_operation(op, Tv, T),N}
end
function _operate!(
    op::Union{typeof(+),typeof(-)},
    A::Matrix,
    B::_SparseMat,
    left_factors::Tuple,
    right_factors::Tuple,
)
    B_nonzeros = SparseArrays.nonzeros(B)
    B_rowvals = SparseArrays.rowvals(B)
    for col = 1:size(B, 2)
        for k ∈ SparseArrays.nzrange(B, col)
            row = B_rowvals[k]
            A[row, col] = operate!!(
                mul_rhs(op),
                A[row, col],
                left_factors...,
                B_nonzeros[k],
                right_factors...,
            )
        end
    end
    return A
end

similar_array_type(::Type{SparseArrays.SparseVector{Tv,Ti}}, ::Type{T}) where {T,Tv,Ti} =
    SparseArrays.SparseVector{T,Ti}
similar_array_type(::Type{_SparseMat{Tv,Ti}}, ::Type{T}) where {T,Tv,Ti} = _SparseMat{T,Ti}

# `SparseArrays/src/linalg.jl` sometimes create a sparse matrix to contain the result.
# For instance with `Matrix * Adjoint{SparseMatrixCSC}` and then uses `generic_matmatmul!`
# which looks quite inefficient as it does not exploit the sparsity of the result matrix and the rhs.
# The approach used here should be more efficient as we redirect to a method that exploits the sparsity of the rhs and `copyto!` should be faster to write the result matrix.
function operate!(
    ::typeof(add_mul),
    output::_SparseMat{T},
    A::AbstractMatrix,
    B::AbstractMatrix,
) where {T}
    C = Matrix{T}(undef, size(output)...)
    operate_to!(C, *, A, B)
    copyto!(output, C)
    return output
end

function operate!(
    ::typeof(add_mul),
    ret::VecOrMat{T},
    adjA::_TransposeOrAdjoint{<:Any,<:_SparseMat},
    B::AbstractVecOrMat,
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    _dim_check(ret, adjA, B)
    A = parent(adjA)
    A_nonzeros = SparseArrays.nonzeros(A)
    A_rowvals = SparseArrays.rowvals(A)
    for k ∈ 1:size(ret, 2)
        for col ∈ 1:A.n
            cur = ret[col, k]
            for j ∈ SparseArrays.nzrange(A, col)
                A_val = _mirror_transpose_or_adjoint(A_nonzeros[j], adjA)
                cur = operate!!(add_mul, cur, A_val, B[A_rowvals[j], k], α...)
            end
            ret[col, k] = cur
        end
    end
    return ret
end
function operate!(
    ::typeof(add_mul),
    ret::VecOrMat{T},
    A::_SparseMat,
    B::AbstractVecOrMat,
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    _dim_check(ret, A, B)
    A_nonzeros = SparseArrays.nonzeros(A)
    A_rowvals = SparseArrays.rowvals(A)
    for col ∈ 1:size(A, 2)
        for k ∈ 1:size(ret, 2)
            αxj = *(B[col, k], α...)
            for j ∈ SparseArrays.nzrange(A, col)
                ret[A_rowvals[j], k] =
                    operate!!(add_mul, ret[A_rowvals[j], k], A_nonzeros[j], αxj)
            end
        end
    end
    return ret
end
function operate!(
    ::typeof(add_mul),
    ret::Matrix{T},
    A::AbstractMatrix,
    B::_SparseMat,
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    _dim_check(ret, A, B)
    rowval = SparseArrays.rowvals(B)
    B_nonzeros = SparseArrays.nonzeros(B)
    for multivec_row = 1:size(A, 1)
        for col ∈ 1:size(B, 2)
            cur = ret[multivec_row, col]
            for k ∈ SparseArrays.nzrange(B, col)
                cur =
                    operate!!(add_mul, cur, A[multivec_row, rowval[k]], B_nonzeros[k], α...)
            end
            ret[multivec_row, col] = cur
        end
    end
    return ret
end

function operate!(
    ::typeof(add_mul),
    ret::Matrix{T},
    A::AbstractMatrix,
    adjB::_TransposeOrAdjoint{<:Any,<:_SparseMat},
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    _dim_check(ret, A, adjB)
    B = parent(adjB)
    B_rowvals = SparseArrays.rowvals(B)
    B_nonzeros = SparseArrays.nonzeros(B)
    for B_col ∈ 1:size(B, 2), k ∈ SparseArrays.nzrange(B, B_col)
        B_row = B_rowvals[k]
        B_val = _mirror_transpose_or_adjoint(B_nonzeros[k], adjB)
        αB_val = *(B_val, α...)
        for A_row = 1:size(A, 1)
            ret[A_row, B_row] =
                operate!!(add_mul, ret[A_row, B_row], A[A_row, B_col], αB_val)
        end
    end
    return ret
end

# `_SparseMat`-`_SparseMat` matrix multiplication.
# Inspired from `SparseArrays.spmatmul` which is
# Gustavsen's matrix multiplication algorithm revisited so that row indices
# are sorted.

function promote_array_mul(
    ::Type{<:Union{_SparseMat{S,Ti},_TransposeOrAdjoint{S,_SparseMat{S,Ti}}}},
    ::Type{<:Union{_SparseMat{T,Ti},_TransposeOrAdjoint{T,_SparseMat{T,Ti}}}},
) where {S,T,Ti}
    return _SparseMat{promote_sum_mul(S, T),Ti}
end

function operate!(
    ::typeof(add_mul),
    ret::_SparseMat{T},
    A::_SparseMat,
    B::_SparseMat,
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    _dim_check(ret, A, B)
    rowvalA = SparseArrays.rowvals(A)
    nzvalA = SparseArrays.nonzeros(A)
    rowvalB = SparseArrays.rowvals(B)
    nzvalB = SparseArrays.nonzeros(B)
    mA, nA = size(A)
    nB = size(B, 2)
    nnz_ret = length(ret.rowval)
    @assert length(ret.nzval) == nnz_ret

    @inbounds begin
        ip = 1
        xb = fill(false, mA)
        for i = 1:nB
            if ip + mA - 1 > nnz_ret
                nnz_ret += max(mA, nnz_ret >> 2)
                resize!(ret.rowval, nnz_ret)
                resize!(ret.nzval, nnz_ret)
            end
            ret.colptr[i] = ip0 = ip
            k0 = ip - 1
            for jp in SparseArrays.nzrange(B, i)
                nzB = nzvalB[jp]
                j = rowvalB[jp]
                for kp in SparseArrays.nzrange(A, j)
                    k = rowvalA[kp]
                    if xb[k]
                        ret.nzval[k+k0] =
                            operate!!(add_mul, ret.nzval[k+k0], nzvalA[kp], nzB)
                    else
                        ret.nzval[k+k0] = operate(*, nzvalA[kp], nzB)
                        xb[k] = true
                        ret.rowval[ip] = k
                        ip += 1
                    end
                end
            end
            if ip > ip0
                if _prefer_sort(ip - k0, mA)
                    # in-place sort of indices. Effort: O(nnz*ln(nnz)).
                    sort!(ret.rowval, ip0, ip - 1, QuickSort, Base.Order.Forward)
                    for vp = ip0:ip-1
                        k = ret.rowval[vp]
                        xb[k] = false
                        ret.nzval[vp] = ret.nzval[k+k0]
                    end
                else
                    # scan result vector (effort O(mA))
                    for k = 1:mA
                        if xb[k]
                            xb[k] = false
                            ret.rowval[ip0] = k
                            ret.nzval[ip0] = ret.nzval[k+k0]
                            ip0 += 1
                        end
                    end
                end
            end
        end
        ret.colptr[nB+1] = ip
    end

    # This modification of Gustavson algorithm has sorted row indices
    resize!(ret.rowval, ip - 1)
    resize!(ret.nzval, ip - 1)

    return ret
end
# Taken from `SparseArrays.prefer_sort` added in Julia v1.1.
_prefer_sort(nz::Integer, m::Integer) = m > 6 && 3 * SparseArrays.ilog2(nz) * nz < m
function operate!(
    ::typeof(add_mul),
    ret::_SparseMat{T},
    A::_SparseMat,
    B::_TransposeOrAdjoint{<:Any,<:_SparseMat},
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    operate!(add_mul, ret, A, copy(B), α...)
end
function operate!(
    ::typeof(add_mul),
    ret::_SparseMat{T},
    A::_TransposeOrAdjoint{<:Any,<:_SparseMat},
    B::_SparseMat,
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    return operate!(add_mul, ret, copy(A), B, α...)
end
function operate!(
    ::typeof(add_mul),
    ret::_SparseMat{T},
    A::_TransposeOrAdjoint{<:Any,<:_SparseMat},
    B::_TransposeOrAdjoint{<:Any,<:_SparseMat},
    α::Vararg{Union{T,Scaling},N},
) where {T,N}
    return operate!(add_mul, ret, copy(A), B, α...)
end

# This `BroadcastStyle` is used when there is a mix of sparse arrays and dense arrays.
# The result is a sparse array.
function _broadcasted_type(
    ::SparseArrays.HigherOrderFns.PromoteToSparse,
    ::Base.HasShape{1},
    ::Type{Eltype},
) where {Eltype}
    return SparseArrays.SparseVector{Eltype,Int}
end
function _broadcasted_type(
    ::SparseArrays.HigherOrderFns.PromoteToSparse,
    ::Base.HasShape{2},
    ::Type{Eltype},
) where {Eltype}
    return _SparseMat{Eltype,Int}
end
