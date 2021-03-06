module TiledIteration

using OffsetArrays
using Base: tail, Indices, @propagate_inbounds
using Base.IteratorsMD: inc

if VERSION < v"1.6.0-DEV.1174"
    _inc(state, iter) = inc(state, first(iter).I, last(iter).I)
else
    # https://github.com/JuliaLang/julia/pull/37829
    _inc(state, iter) = inc(state, iter.indices)
end

export TileIterator, EdgeIterator, SplitAxis, SplitAxes, padded_tilesize, TileBuffer, RelaxStride, RelaxLastTile

include("tileiterator.jl")

const L1cachesize = 2^15
const cachelinesize = 64

### EdgeIterator ###

struct EdgeIterator{N,UR1,UR2}
    outer::CartesianIndices{N,UR1}
    inner::CartesianIndices{N,UR2}
    function EdgeIterator{N,UR1,UR2}(outer::CartesianIndices{N}, inner::CartesianIndices{N}) where {N,UR1,UR2}
        ((first(inner) ∈ outer) & (last(inner) ∈ outer)) || throw(DimensionMismatch("$inner must be in the interior of $outer"))
        new(outer, inner)
    end
end
EdgeIterator(outer::CartesianIndices{N,UR1}, inner::CartesianIndices{N,UR2}) where {N,UR1,UR2} =
    EdgeIterator{N,UR1,UR2}(outer, inner)
EdgeIterator(outer::Indices{N}, inner::Indices{N}) where N =
    EdgeIterator(promote(CartesianIndices(outer), CartesianIndices(inner))...)

"""
    EdgeIterator(outer, inner)

A Cartesian iterator that efficiently visits sites that are in `outer`
but not in `inner`. This can be useful for calculating edge values
that may require special treatment or boundary conditions.
"""
EdgeIterator

Iterators.IteratorEltype(::Type{<:EdgeIterator}) = Iterators.HasEltype()

Base.eltype(::Type{EdgeIterator{N,UR1,UR2}}) where {N,UR1,UR2} = CartesianIndex{N}
Base.length(iter::EdgeIterator) = length(iter.outer) - length(iter.inner)

function Base.iterate(iter::EdgeIterator)
    iterouter = iterate(iter.outer)
    iterouter === nothing && return nothing
    item = nextedgeitem(iter, iterouter[2])
    item ∉ iter.outer && return nothing
    return item, item
end
function Base.iterate(iter::EdgeIterator, state)
    iterouter = iterate(iter.outer, state)
    iterouter === nothing && return nothing
    item = nextedgeitem(iter, iterouter[2])
    item.I[end] > last(iter.outer.indices[end]) && return nothing
    return item, item
end

@inline function nextedgeitem(iter::EdgeIterator, I::CartesianIndex)
    !(I ∈ iter.inner) && return I
    state = (last(iter.inner)[1], tail(I.I)...)
    newI = CartesianIndex(_inc(state, iter.outer))
    nextedgeitem(iter, newI)
end

Base.show(io::IO, iter::EdgeIterator) = print(io, "EdgeIterator(", iter.outer.indices, ", ", iter.inner.indices, ')')

### SplitAxis and SplitAxes

struct SplitAxis <: AbstractVector{UnitRange{Int}}
    splits::Vector{Int}
end

"""
    SplitAxis(ax::AbstractUnitRange, n::Real)

Split `ax` into `ceil(Int, n)` approximately equal-sized chunks. The first chunk is no larger than any other chunk,
and any fractional "deficit" in `n` will further shrink the first chunk.

This can be useful in splitting work across threads. When the first thread is responsible for assigning work to the others,
it's often useful to assign less work to it to account for the time spent scheduling.

# Examples

```jldoctest; setup=:(using TiledIteration)
julia> collect(SplitAxis(1:16, 4))
4-element Vector{UnitRange{$Int}}:
 1:4
 5:8
 9:12
 13:16

julia> collect(SplitAxis(1:16, 3.5))
4-element Vector{UnitRange{$Int}}:
 1:1
 2:6
 7:11
 12:16
```

In the latter case all the ranges except the first have length 5; consequently, only one element remains for the first chunk.
"""
function SplitAxis(ax::AbstractUnitRange{<:Integer}, n::Real)
    step = ceil(Int, length(ax)/n)
    # Give the smallest amount of work to thread 1, since often it is also scheduling the work for all
    # the other threads.
    SplitAxis(max.(first(ax)-1, collect(reverse(range(last(ax), step=-step, length=ceil(Int, n)+1)))))
end

Base.@propagate_inbounds Base.getindex(sax::SplitAxis, i::Int) = sax.splits[i]+1:sax.splits[i+1]

Base.size(sax::SplitAxis) = (length(sax.splits)-1,)

struct SplitAxes{N} <: AbstractVector{Tuple{UnitRange{Int},Vararg{UnitRange{Int},N}}}
    inner::NTuple{N,UnitRange{Int}}
    splitax::SplitAxis
end

"""
    SplitAxes(axs::NTuple{N,AbstractUnitRange}, n::Real)

Split `axs` into `ceil(Int, n)` approximately equal-sized chunks along the final dimension represented by `axs`.

See [`SplitAxis`](@ref) for further details.

# Examples

```jldoctest; setup=:(using TiledIteration)
julia> A = rand(3, 16);

julia> collect(SplitAxes(axes(A), 4))
4-element Vector{Tuple{UnitRange{$Int}, UnitRange{$Int}}}:
 (1:3, 1:4)
 (1:3, 5:8)
 (1:3, 9:12)
 (1:3, 13:16)
```
"""
SplitAxes(axs::Tuple{AbstractUnitRange,Vararg{AbstractUnitRange}}, n::Real) = SplitAxes{length(axs)-1}(Base.front(axs), SplitAxis(axs[end], n))

Base.@propagate_inbounds Base.getindex(saxs::SplitAxes, i::Int) = (saxs.inner..., saxs.splitax[i])

Base.size(saxs::SplitAxes) = size(saxs.splitax)

### Calculating the size of tiles ###

# If kernelsize-1 is the amount of padding, and s is the extra width of the tile along
# each axis, then the fraction of useful to total work is
# prod(s)/prod(s+kernelsize); this ratio is maximized if s is proportional to
# kernelsize.
"""
    padded_tilesize(T::Type, kernelsize::Dims, [ncache=2]) -> tilesize::Dims

Calculate a suitable tile size to approximately maximize the amount of
productive work, given a stencil of size `kernelsize`. The element
type of the array is `T`. Optionally specify `ncache`, the number of
such arrays that you'd like to have fit simultaneously in L1 cache.

This favors making the first dimension larger, since the first
dimension corresponds to individual cache lines.

# Examples
julia> padded_tilesize(UInt8, (3,3))
(768,18)

julia> padded_tilesize(UInt8, (3,3), 4)
(512,12)

julia> padded_tilesize(Float64, (3,3))
(96,18)

julia> padded_tilesize(Float32, (3,3,3))
(64,6,6)
"""
function padded_tilesize(::Type{T}, kernelsize::Dims, ncache = 2) where T
    nd = max(1, sum(x->x>1, kernelsize))
    # isbits(T) || return map(zero, kernelsize)
    # don't be too minimalist on the cache-friendly dim (use at least 2 cachelines)
    dim1minlen = 2*cachelinesize÷sizeof(T)
    psz = (max(kernelsize[1], dim1minlen), tail(kernelsize)...)
    L = sizeof(T)*prod(psz)
    # try to stay in L1 cache, but in the end we want a reasonably
    # favorable work ratio. f is the constant of proportionality in
    #      s+kernelsize ∝ kernelsize
    f = max(floor(Int, (L1cachesize/(ncache*L))^(1/nd)), 2)
    return _padded_tilesize_scale(f, psz)
end

@noinline _padded_tilesize_scale(f, psz) = map(x->x <= 1 ? x : f*x, psz) # see #15276

### Tile shaping and coordinate transformation

struct TileBuffer{T,N,P} <: AbstractArray{T,N}
    view::OffsetArray{T,N,Array{T,N}}  # the currently-active view
    buf::Array{T,P}                    # the original backing buffer
end

"""
    TileBuffer(a, inds::Indices) -> v

Return a buffer-view `v` whose indices match `inds`, using the array
`a` for storage. `inds` does not necessarily have to match the size of
`a` (which allows tiles to be of different sizes, e.g., at the edges).
"""
@inline function TileBuffer(a::Array, inds::Indices)
    l = map(length, inds)
    tilev = OffsetArray(_tileview(a, l), inds)
    TileBuffer(tilev, a)
end

"""
    TileBuffer(T, inds::Indices) -> v

Return a TileBuffer, allocating a new backing array of element type `T`
and size determined by `inds`.
"""
function TileBuffer(::Type{T}, inds::Indices) where T
    l = map(length, inds)
    TileBuffer(Array{T}(undef, l), inds)
end

TileBuffer(tb::TileBuffer, inds::Indices) = TileBuffer(tb.buf, inds)

@inline function _tileview(a::Array, l::Dims)
    if size(a) == l
        return a
    else
        # returning a SubArray would not be type-stable, we must return an Array
        prod(l) > length(a) && throw(DimensionMismatch("array of size $(size(a)) is not adequate for a tile of size $l"))
        return unsafe_wrap(Array, pointer(a), l)
    end
end

Base.axes(tb::TileBuffer) = axes(tb.view)
Base.size(tb::TileBuffer) = size(tb.view)

@inline @propagate_inbounds Base.getindex(tb::TileBuffer{T,N}, I::Vararg{Int,N}) where {T,N} = tb.view[I...]

@inline @propagate_inbounds Base.setindex!(tb::TileBuffer{T,N}, val, I::Vararg{Int,N}) where {T,N} = tb.view[I...] = val

Base.pointer(tb::TileBuffer) = pointer(parent(tb.view))

Base.parent(tb::TileBuffer) = tb.buf

end # module
