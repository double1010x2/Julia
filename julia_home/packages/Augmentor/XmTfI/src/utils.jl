"""
    testpattern([T=RGBA{N0f8}]; ratio=1.0) -> Matrix{RGBA{N0f8}}

Load and return the provided 300x400 test image. Additional args and kwargs
are passed to `imresize`.

The returned image was specifically designed to be informative
about the effects of the applied augmentation operations. It is
thus well suited to prototype an augmentation pipeline, because it
makes it easy to see what kind of effects one can achieve with it.
"""
function testpattern(args...; ratio=1.0)
    imresize(load(joinpath(@__DIR__, "..", "resources", "testpattern.png")), ratio=ratio)
end
function testpattern(T::Type{<:Colorant}; ratio=1.0)
    # Directly call T.(testpattern) returns a testpattern with border filled with black pixels
    # This patch fills border with white pixels so as to be consistent with ARGB(0, 0, 0, 0).
    npad = 20
    temp = testpattern()
    out = fill(oneunit(T), size(temp))
    out[npad:end-npad, npad:end-npad] .= temp[npad:end-npad, npad:end-npad]
    return imresize(out, ratio=ratio)
end

function use_testpattern()
    @info("No custom image specifed. Using \"testpattern()\" for demonstration.")
    testpattern()
end

# --------------------------------------------------------------------
"""
    contiguous(A::AbstractArray)
    contiguous(A::Tuple)

Return a memory contiguous array for better performance.

Data copy only happens when necessary. For example, views returned by `view`,
`PermutedDimsArray` are such cases.

See also: [`plain_array`](@ref), [`plain_axes`](@ref)
"""
@inline contiguous(A::OffsetArray) = A
@inline contiguous(A::Array) = A
@inline contiguous(A::SArray) = A
@inline contiguous(A::MArray) = A
@inline contiguous(A::AbstractArray) = match_idx(collect(A), axes(A))
@inline contiguous(A::Mask) = Mask(contiguous(unwrap(A)))
@inline contiguous(A::Tuple) = map(contiguous, A)

# --------------------------------------------------------------------

@inline _plain_array(A::OffsetArray) = _plain_array(parent(A))
@inline _plain_array(A::Array) = A
@inline _plain_array(A::SArray) = A
@inline _plain_array(A::MArray) = A
@inline _plain_array(A::AbstractArray) = collect(A)
@inline _plain_array(A::Mask) = Mask(_plain_array(unwrap(A)))
@inline _plain_array(A::Tuple) = map(_plain_array, A)

"""
    plain_array(A::AbstractArray)
    plain_array(A::Tuple)

Return a memory contiguous plain array for better performance.
A plain array is either an `Array` or a `StaticArray`.

Data copy only happens when necessary. For example, views returned by `view`,
`PermutedDimsArray` are such cases.

See also: [`contiguous`](@ref), [`plain_axes`](@ref)
"""
@inline plain_array(A) = _plain_array(contiguous(A)) # avoid recursion

# --------------------------------------------------------------------

"""
    plain_axes(A::AbstractArray)

Generate a 1-based array from `A` without data copy.

See also: [`contiguous`](@ref), [`plain_array`](@ref)
"""
@inline plain_axes(A::Array) = A
@inline plain_axes(A::OffsetArray) = parent(A)
@inline plain_axes(A::AbstractArray) = _plain_axes(A, axes(A))
@inline plain_axes(A::SubArray) = _plain_axes(A, A.indices)

@inline function _plain_axes(A::AbstractArray{T,N}, ids::NTuple{N,Base.OneTo}) where {T, N}
    A
end

@inline function _plain_axes(A::AbstractArray, ids::Tuple{Vararg{Any}})
    view(A, axes(A)...)
end

@inline function _plain_axes(A::AbstractArray{T,N}, ids::NTuple{N, IdentityUnitRange}) where {T, N}
    view(A, map(i->i.indices, ids)...)
end

@inline function _plain_axes(A::SubArray{T,N}, ids::NTuple{N,IdentityRange}) where {T, N}
    view(parent(A), axes(A)...)
end

# --------------------------------------------------------------------

@inline match_idx(buffer::AbstractArray, inds::Tuple) = buffer
@inline match_idx(buffer::Union{Array,SubArray}, inds::NTuple{N,OffsetRange}) where {N} =
    OffsetArray(buffer, inds)

# --------------------------------------------------------------------

function indirect_axes(::Tuple{}, ::Tuple{})
    throw(MethodError(indirect_axes, ((),())))
end

@inline function indirect_axes(O::NTuple{N,Base.OneTo}, I::NTuple{N,AbstractUnitRange}) where N
    map(IdentityRange, I)
end

@inline function indirect_axes(O::NTuple{N,Base.OneTo}, I::NTuple{N,StepRange}) where N
    I
end

function indirect_axes(O::NTuple{N,AbstractUnitRange}, I::NTuple{N,AbstractUnitRange}) where N
    map((i1,i2) -> IdentityRange(UnitRange(i1)[i2]), O, I)
end

function indirect_axes(O::NTuple{N,AbstractUnitRange}, I::NTuple{N,StepRange}) where N
    map((i1,i2) -> UnitRange(i1)[i2], O, I)
end

function indirect_axes(O::NTuple{N,StepRange}, I::NTuple{N,AbstractRange}) where N
    map((i1,i2) -> i1[i2], O, I)
end

# --------------------------------------------------------------------

function indirect_view(A::AbstractArray, I::Tuple)
    view(A, indirect_axes(axes(A), I)...)
end

function indirect_view(A::SubArray{T,N,TA,<:NTuple{N,AbstractRange}}, I::Tuple) where {T,N,TA}
    view(parent(A), indirect_axes(A.indices, I)...)
end

# --------------------------------------------------------------------

function direct_axes(::Tuple{}, ::Tuple{})
    throw(MethodError(direct_axes, ((),())))
end

# TODO: Figure out why this method exists
function direct_axes(O::NTuple{N,IdentityRange}, I::NTuple{N,StepRange}) where N
    throw(MethodError(direct_axes, (O, I)))
end

@inline function direct_axes(O::NTuple{N,AbstractRange}, I::NTuple{N,AbstractUnitRange}) where N
    map(IdentityRange, I)
end

@inline function direct_axes(O::NTuple{N,AbstractRange}, I::NTuple{N,StepRange}) where N
    I
end

# --------------------------------------------------------------------

function direct_view(A::AbstractArray{T,N}, I::NTuple{N,AbstractRange}) where {T,N}
    view(A, direct_axes(axes(A), I)...)
end

function direct_view(A::SubArray{T,N,TA,<:NTuple{N,AbstractRange}}, I::NTuple{N,AbstractRange}) where {T,N,TA}
    view(A, direct_axes(A.indices, I)...)
end

# --------------------------------------------------------------------

@inline vectorize(A::AbstractVector) = A
@inline vectorize(A::Real) = A:A

@inline round_if_float(num::Integer, d) = num
round_if_float(num::AbstractFloat, d) = round(num, digits=d)
round_if_float(nums::Tuple, d) = map(num->round_if_float(num,d), nums)

function unionrange(i1::AbstractUnitRange, i2::AbstractUnitRange)
    map(min, first(i1), first(i2)):map(max, last(i1), last(i2))
end

@inline _showcolor(io::IO, T::Type{<:Number}) = print(io, T)
@inline _showcolor(io::IO, T) = ColorTypes.colorant_string_with_eltype(io, T)

# --------------------------------------------------------------------

function _2dborder!(A::AbstractArray{T,3}, val::T) where T
    ndims, h, w = size(A)
    @inbounds for j = (1,w), i = 1:h
        for d = 1:ndims
            A[d,i,j] = val
        end
    end
    @inbounds for j = 1:w, i = (1,h)
        for d = 1:ndims
            A[d,i,j] = val
        end
    end
    A
end

# This is expected to be added to Julia (maybe under a different name)
# Follow https://github.com/JuliaLang/julia/issues/35543 for progress
basetype(T::Type) = Base.typename(T).wrapper
basetype(T) = basetype(typeof(T))
