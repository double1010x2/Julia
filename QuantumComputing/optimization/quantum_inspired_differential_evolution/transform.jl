using Random

function linearMap(x::AbstractArray{<:Real}, bound_in::Vector{<:Real}, bound_out::Vector{<:Real})
    a = (bound_out[end]-bound_out[begin]) / (bound_in[end]-bound_in[begin])
    b = bound_out[begin] - a * bound_in[begin]
    return a .* x .+ b
end

function vector2DtoMatrix(_b::Vector{Vector{Float64}}) 
    _nb = Array{eltype(_b[1][1]), 2}(undef, length(_b), 2)
    for i in range(1, length(_b))
        _nb[i,:] = _b[i]
    end
    return _nb
end 
vector2DtoMatrix(_b::Array{Float64, 2}) = _b

function latinHyperCubicSampling!(_arr::Array{<:Real, 2}, bounds::Array{<:Real, 2}; seed=-1)
    row, col = size(_arr)

    # loop dimensions 
    (seed >= 0) && (Random.seed!(seed))
    for ci in range(1, col)
        # mapping from 0~1 to bound 
        grid = (bounds[ci,end]-bounds[ci,begin]) / float(row)
        x = Vector{Float64}(0:row-1) ./ row 
        x = linearMap(x, [0., 1.], bounds[ci,:])
        x = x .+ rand(row) .* grid 
        index = shuffle(1:row) 
        # loop particles
        _arr[:,ci] = [x[index[ri]] for ri in range(1, row)]
    end
end

vectorNorm(v)   = sqrt.(v[1].^2 + v[2].^2 + v[3].^2)
unitVector(v)   = (norm = vectorNorm(v); (v[1] ./ norm, v[2] ./ norm, v[3] ./ norm))
outerProduct(p, q) = (p[2].*q[3] .- p[3].*q[2], p[3].*q[1] .- p[1].*q[3], p[1].*q[2] .- p[2].*q[1]) 
