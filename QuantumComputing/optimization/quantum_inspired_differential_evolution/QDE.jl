

function latinHyperCubicSampling!(_swarm::Swarm, bounds::Array{<:Real, 2}; seed=-1)
    row = length(_swarm)
    col = getDim(_swarm)

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
        for ri in range(1, row) 
            setPos!(_swarm.particles[ri], ci, x[index[ri]])
        end

    end
end
