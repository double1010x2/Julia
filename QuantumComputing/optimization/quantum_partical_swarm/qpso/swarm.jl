include("particle.jl")
mutable struct Swarm 
    particles::Vector{Particle}
    gbest::Vector{<:Real}
    gbest_val::Real
    bound::Array{<:Real, 2}
end
iterate(_s::Swarm)          = _s.particles[1], 1 
iterate(_s::Swarm, i::Int)  = (i+=1; (i > length(_s)) ? (nothing) : (_s.particles[i], i))
Swarm(_n::Int, _dim::Int; trend="min") = (
    _b = initialBound(_dim);
    Swarm(_n, _dim, _b; trend=trend)
)
Swarm(_num_of_particle::Int, _dim::Int, _b::Array{<:Real, 2}; trend="min") = (
    _particles  = [Particle(_b) for _ in range(1, _num_of_particle)];
    _gbest      = zeros(_dim);
    Swarm(_particles, _gbest, (trend=="min") ? (Inf) : (-Inf), _b)
)
length(_swarm::Swarm)    = (length(_swarm.particles))
getDim(_swarm::Swarm)    = (size(_swarm.bound)[begin])
getParticle(_s::Swarm, pi::Int)   = (_s.particles[pi])
getBestPos(_s::Swarm)    = _s.gbest 
getBestVal(_s::Swarm)    = _s.gbest_val 
function getMeanPos(_s::Swarm)    
    mpos = vector2DtoMatrix([getBestPos(pi) for pi in _s])
    return mean(mpos, dims=1)
#    mpos = zeros(Float64, size(_s.gbest));
#    for pi in _s
#        mpos = mpos .+ getBestPos(pi)
#    end
#    return mpos ./ length(_s)
end

function updateBest!(_s::Swarm; trend="min")
    row         = length(_s)
    gbest       = getBestPos(_s)
    gbest_val   = getBestVal(_s)
    op          = (trend=="min") ? (<=) : (>=) 
    for ri in range(1, row)
        if op(getBestVal(_s.particles[ri]) ,gbest_val)  
            gbest       = getBestPos(_s.particles[ri])
            gbest_val   = getBestVal(_s.particles[ri])
        end
    end
    _s.gbest     = gbest
    _s.gbest_val = gbest_val
end