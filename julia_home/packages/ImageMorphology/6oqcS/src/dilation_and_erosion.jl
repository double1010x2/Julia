# Erode and dilate support 3x3 regions only (and higher-dimensional generalizations).

"""
    imgd = dilate(img; dims=coords_spatial(img))

Perform a max-filter over nearest-neighbors. The
default is 8-connectivity in 2d, 27-connectivity in 3d, etc. You can specify the
list of dimensions that you want to include in the connectivity, e.g., `dims =
(1,2)` would exclude the third dimension from filtering.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(5, 5); img[3, 3] = 1.; img
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> dilate(img)
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> dilate(img; dims=1)
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
dilate(img::AbstractArray; kwargs...) = dilate!(copy(img); kwargs...)

"""
```
imge = erode(img; dims=coords_spatial(img))
```

perform a min-filter over nearest-neighbors. The
default is 8-connectivity in 2d, 27-connectivity in 3d, etc. You can specify the
list of dimensions that you want to include in the connectivity, e.g., `dims =
(1,2)` would exclude the third dimension from filtering.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(5, 5); img[2:4, 2:4] .= 1.; img
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> erode(img)
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0

julia> erode(img; dims=1)
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
erode(img::AbstractArray; kwargs...) = erode!(copy(img); kwargs...)

dilate!(maxfilt; kwargs...) = extremefilt!(maxfilt, max; kwargs...)
erode!(minfilt; kwargs...) = extremefilt!(minfilt, min; kwargs...)
function extremefilt!(A::AbstractArray, select::Function; dims=coords_spatial(A))
    inds = axes(A)
    for d = 1:ndims(A)
        if size(A, d) == 1 || d ∉ dims
            continue
        end
        Rpre = CartesianIndices(inds[1:d-1])
        Rpost = CartesianIndices(inds[d+1:end])
        _extremefilt!(A, select, Rpre, inds[d], Rpost)
    end
    A
end

@noinline function _extremefilt!(A, select, Rpre, inds, Rpost)
    # TODO: improve the cache efficiency
    for Ipost in Rpost, Ipre in Rpre
        # first element along dim
        i1 = first(inds)
        a2, a3 = A[Ipre, i1, Ipost], A[Ipre, i1+1, Ipost]
        A[Ipre, i1, Ipost] = select(a2, a3)
        # interior along dim
        for i = i1+2:last(inds)
            a1, a2 = a2, a3
            a3 = A[Ipre, i, Ipost]
            A[Ipre, i-1, Ipost] = select(select(a1, a2), a3)
        end
        # last element along dim
        A[Ipre, last(inds), Ipost] = select(a2, a3)
    end
    A
end

"""
`imgo = opening(img; dims=coords_spatial(img))` performs the `opening` morphology operation, equivalent to `dilate(erode(img))`.
`dims` allows you to control the dimensions over which this operation is performed.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(5, 5); img[1, 1] = 1.; img[3:5, 3:5] .= 1.; img
5×5 Array{Float64,2}:
 1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0

julia> opening(img)
5×5 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0
```
"""
opening(img::AbstractArray; kwargs...) = opening!(copy(img); kwargs...)
opening!(img::AbstractArray; kwargs...) = dilate!(erode!(img; kwargs...); kwargs...)

"""
`imgc = closing(img; dims=coords_spatial(img))` performs the `closing` morphology operation, equivalent to `erode(dilate(img))`.
`dims` allows you to control the dimensions over which this operation is performed.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(7, 7); img[3:5, 3:5] .= 1.; img[4, 4] = 0.; img
7×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> closing(img)
7×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
closing(img::AbstractArray; kwargs...) = closing!(copy(img); kwargs...)
closing!(img::AbstractArray; kwargs...) = erode!(dilate!(img; kwargs...); kwargs...)

"""
`imgth = tophat(img; dims=coords_spatial(img))` performs `top hat` of an image,
which is defined as the image minus its morphological opening.
`dims` allows you to control the dimensions over which this operation is performed.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(5, 5); img[1, 1] = 1.; img[3:5, 3:5] .= 1.; img
5×5 Array{Float64,2}:
 1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0
 0.0  0.0  1.0  1.0  1.0

julia> tophat(img)
5×5 Array{Float64,2}:
 1.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
```
"""
tophat(img::AbstractArray; kwargs...) = img - opening(img; kwargs...)

"""
`imgbh = bothat(img; dims=coords_spatial(img))` performs `bottom hat` of an image,
which is defined as its morphological closing minus the original image.
`dims` allows you to control the dimensions over which this operation is performed.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(7, 7); img[3:5, 3:5] .= 1.; img[4, 4] = 0.; img
7×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> bothat(img)
7×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
bothat(img::AbstractArray; kwargs...) = closing(img; kwargs...) - img

"""
`imgmg = morphogradient(img; dims=coords_spatial(img))` returns morphological gradient of the image,
which is the difference between the dilation and the erosion of a given image.
`dims` allows you to control the dimensions over which this operation is performed.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(7, 7); img[3:5, 3:5] .= 1.; img
7×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> morphogradient(img)
7×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  0.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  1.0  1.0  1.0  1.0  1.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
morphogradient(img::AbstractArray; kwargs...) = dilate(img; kwargs...) - erode(img; kwargs...)

"""
`imgml = morpholaplace(img; dims=coords_spatial(img))` performs `Morphological Laplacian` of an image,
which is defined as the arithmetic difference between the internal and the external gradient.
`dims` allows you to control the dimensions over which this operation is performed.

# Examples

```jldoctest; setup = :(using ImageMorphology), filter = r"Array{Float64,2}|Matrix{Float64}"
julia> img = zeros(7, 7); img[3:5, 3:5] .= 1.; img[4, 4] = 0.; img
7×7 Array{Float64,2}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> morpholaplace(img)
7×7 Array{Float64,2}:
 0.0  0.0   0.0   0.0   0.0  0.0  0.0
 0.0  1.0   1.0   1.0   1.0  1.0  0.0
 0.0  1.0  -1.0  -1.0  -1.0  1.0  0.0
 0.0  1.0  -1.0   1.0  -1.0  1.0  0.0
 0.0  1.0  -1.0  -1.0  -1.0  1.0  0.0
 0.0  1.0   1.0   1.0   1.0  1.0  0.0
 0.0  0.0   0.0   0.0   0.0  0.0  0.0
```
"""
morpholaplace(img::AbstractArray; kwargs...) = dilate(img; kwargs...) + erode(img; kwargs...) - 2img
