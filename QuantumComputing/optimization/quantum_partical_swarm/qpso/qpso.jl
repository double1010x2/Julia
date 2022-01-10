#========================================================================
=   Quantum Particle Swarm Optimization (QPSO) algorithm                =
=   Author:                                                             =
        double1010x2                                                    =
=   Ref:                                                                =
=       (1) https://github.com/SaTa999/pyQPSO                           =
=       (2) https://github.com/ngroup/qpso                              =
=       (3) "Quantum-behaved Particle Swarm Optimization with Novel     =
=            Adaptive Strategies",                                      =
=       Xinyi Sheng1*, Maolong Xi2,3, Jun Sun1,4 and Wenbo Xu1,4,       =
=       Journal of Algorithms & Computational Technology Vol. 9 No. 2   =
========================================================================#

using ArgParse
using Dates
using PyCall
using Random
import Base: iterate
import Base: length
@pyimport matplotlib.pylab as plt

function timeMark()
    t = Dates.now()
    Y = Dates.year(t)
    M = Dates.month(t)
    D = Dates.day(t)
    h = Dates.hour(t)
    m = Dates.minute(t)
    s = Dates.second(t)
    println("[$Y-$M-$D $h:$m:$s] - ")
end

mutable struct Particle
    pos::Vector{<:Real}
    best_pos::Vector{<:Real}
    best_val::Real
    bound::Array{<:Real, 2}
    function Particle(_bound; _trend::String="min") 
        ndims(_bound) <= 1 && (_bound = reshape(_bound, 1, 2))
        dim = size(_bound)[begin]
        _particle = new(fill(0., dim), fill(0., dim), (_trend=="min") ? Inf : -Inf, _bound)
        return _particle 
    end
end
#======== [iterator] for position element in particle ========#
iterate(_p::Particle)               = _p.pos[1], 1
iterate(_p::Particle, i::Int)       = (i+=1; (i > getDim(_p)) ? (nothing) : (_p.pos[i], i)) # This method caused bypass some
#======== [utility] for particle struct ========#
getPos(_p::Particle)        = _p.pos 
setPos!(_p::Particle, _pos) = (_p.pos = _pos)
setPos!(_p::Particle, _dim::Int, _val::Real)     = (_p.pos[_dim] = _val)
setBound(_p::Particle, _b::Array{<:Real, 2})    = (_p.bound = _b)
boundPos!(_p::Particle, dim) = (
    (_p.pos[dim] > _p.bound[end])   && (_p.pos[dim] = _p.bound[end]);
    (_p.pos[dim] < _p.bound[begin]) && (_p.pos[dim] = _p.bound[begin]);
)
#======== override the length of particle for loop in [] ========#
length(_p::Particle)        = length(_p.pos)    

getDim(_p::Particle)        = length(_p.pos)
getBestPos(_p::Particle)    = _p.best_pos
getBestVal(_p::Particle)    = _p.best_val
initialBound(dim::Int)      = (_b = ones(Float64, (dim, 2)); _b[:,begin] .= -1; _b) 
ParticleN(dim::Int)         = (Particle(initialBound(dim)))

function setBestPos!(_p::Particle, _pos::Vector{<:Real}, _val::Real; trend="min") 
    op = (trend == "min") ? (<=) : (>=)
    if op(_val, _p.best_val)
        _p.best_pos = _pos
        _p.best_val = _val
    end
end

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
getBestPos(_s::Swarm)    = _s.gbest 
getBestVal(_s::Swarm)    = _s.gbest_val 
function getMeanPos(_s::Swarm)    
    mpos = zeros(Float64, size(_s.gbest));
    for pi in _s
        mpos = mpos .+ getBestPos(pi)
    end
    return mpos ./ length(_s)
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

function linearMap(x::AbstractArray{<:Real}, bound_in::Vector{<:Real}, bound_out::Vector{<:Real})
    a = (bound_out[end]-bound_out[begin]) / (bound_in[end]-bound_in[begin])
    b = bound_out[begin] - a * bound_in[begin]
    return a .* x .+ b
end

function newBound(_b::Vector{Vector{Float64}}) 
    _nb = Array{eltype(_b[1][1]), 2}(undef, length(_b), 2)
    for i in range(1, length(_b))
        _nb[i,:] = _b[i]
    end
    return _nb
end 
newBound(_b::Array{Float64, 2}) = _b 

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

mutable struct pHistory
    pos::Array{<:Real, 2}
    val::Vector{<:Real}
    pHistory(count::Int, dim::Int) = (
        _pos = zeros(Real, (count, dim));
        _val = zeros(Real, count);
        new(_pos, _val) 
    )
end
length(_ph::pHistory) = length(val)

ALPHA_LINEAR = "global-mean"
ALPHA_UP    = "up-weighted"
ALPHA_DOWN  = "down-weighted"
mutable struct Alpha
    a0::Real
    a1::Real
    strategy::String
    Alpha(_a0::Real, _a1::Real, _s::String) = new(_a0, _a1, _s)
end

ATTRACTOR_GLOBAL    = 1
ATTRACTOR_BALANCED  = 2
ATTRACTOR_MEAN      = 3

mutable struct QPSO
    oracle
    dim::Int
    count::Int
    swarm::Swarm
    maxiter::Int
    bounds::Array{<:Real, 2}
    g::Float64
    alpha::Alpha
    attractor::Int
    ph::pHistory
    QPSO(_o, _c::Int, _b::AbstractArray, _miter::Int; 
            seed::Int=-1, trend::String="min", g=0.95, 
            alpha0=0.9, alpha1=0.1, alpha_strategy=ALPHA_LINEAR,
            attractor=ATTRACTOR_BALANCED) = (
        _b   = newBound(_b);
        _dim = size(_b)[begin];
        _s   = Swarm(_c, _dim, _b);
        latinHyperCubicSampling!(_s, _b; seed=seed);
        _ph = pHistory(_miter, _dim);
        _alpha = Alpha(alpha0, alpha1, alpha_strategy);
        new(_o, _dim, _c, _s, _miter, _b, g, _alpha, attractor, _ph)
    )
end
getBestPos(_q::QPSO) = (getBestPos(_q.swarm))
getBestVal(_q::QPSO) = (getBestVal(_q.swarm))
saveBest!(_q::QPSO, iter::Int) = (_q.ph.pos[iter,:] = getBestPos(_q.swarm); _q.ph.val[iter]=getBestVal(_q.swarm);)
function getMeanWeight(_q::QPSO, iter::Int)
    alpha0  = _q.alpha.a0
    alpha1  = _q.alpha.a1
    dalpha  = alpha0 - alpha1
    iter_wt = float(iter)/float(_q.maxiter)
    if _q.alpha.strategy == ALPHA_DOWN 
        return (alpha1 + dalpha * iter_wt^2)
    elseif _q.alpha.strategy == ALPHA_UP
        return (dalpha * iter_wt^2 - alpha1 * dalpha * 2 * iter_wt + alpha1) 
    else 
        return 1. 
    end
end

function updateParticles(_q::QPSO, iter::Int)
    (iter == 1) && (return)
    mpos = getMeanPos(_q.swarm) .* getMeanWeight(_q, iter)
    for _p in _q.swarm
        for dim in range(1, getDim(_p)) 
            u1, u2, u3 = rand(3)
            _sign = (rand() > 0.5) ? 1 : -1
            c = (u1 * getBestPos(_p)[dim] + u2 * getBestPos(_q.swarm)[dim]) / (u1 + u2)
            L = if _q.attractor == ATTRACTOR_GLOBAL
                (1. / _q.g) * abs(getBestPos(_q.swarm)[dim]-c)
            elseif _q.attractor == ATTRACTOR_BALANCED
                (1. / _q.g) * (abs(getBestPos(_q.swarm)[dim]-c) + abs(mpos[dim]-c))
            else
                (1. / _q.g) * abs(mpos[dim]-c)
            end
            #    (1. / _q.g) * abs(getPos(_p)[dim]-c)
            setPos!(_p, dim, c + _sign * L * log(1. / u3))
            boundPos!(_p, dim)
        end
    end

end

function getCost(_q::QPSO)
    for pi in _q.swarm
        setBestPos!(pi, getPos(pi), _q.oracle(getPos(pi)))
    end
end

function optimized(_q::QPSO)
    iter = 1
    while iter <= _q.maxiter
        updateParticles(_q, iter)
        getCost(_q)
        updateBest!(_q.swarm)
        saveBest!(_q, iter)
        iter += 1
    end
end

rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
eggholder(x)  = -(x[2] + 47) * sin(sqrt(abs(x[2] + 0.5*x[1] + 47))) - x[1]*sin(sqrt(abs(x[1] - (x[2] + 47))))

function main()
#    qpso_b = QPSO(eggholder, 40, [[-512., 512.], [-512., 512.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_BALANCED)
#    qpso_g = QPSO(eggholder, 40, [[-512., 512.], [-512., 512.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_GLOBAL)
#    qpso_m = QPSO(eggholder, 40, [[-512., 512.], [-512., 512.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_MEAN)
#    qpso_b_up   = QPSO(eggholder, 40, [[-512., 512.], [-512., 512.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_UP)
#    qpso_b_down = QPSO(eggholder, 40, [[-512., 512.], [-512., 512.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_DOWN)
   
    qpso_b = QPSO(rosenbrock, 40, [[-10., 10.], [-10., 10.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_BALANCED)
    qpso_g = QPSO(rosenbrock, 40, [[-10., 10.], [-10., 10.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_GLOBAL)
    qpso_m = QPSO(rosenbrock, 40, [[-10., 10.], [-10., 10.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_MEAN)
    qpso_b_up   = QPSO(rosenbrock, 40, [[-10., 10.], [-10., 10.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_UP)
    qpso_b_down = QPSO(rosenbrock, 40, [[-10., 10.], [-10., 10.]], 50; alpha0=1.5, alpha1=0.5, seed=0, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_DOWN)
    optimized(qpso_b)
    optimized(qpso_g)
    optimized(qpso_m)
    optimized(qpso_b_up)
    optimized(qpso_b_down)

    plt.plot(qpso_b.ph.pos[:,2], "-r.")
    plt.plot(qpso_g.ph.pos[:,2], "-go")
    plt.plot(qpso_m.ph.pos[:,2], "-bv")
    plt.plot(qpso_b_up.ph.pos[:,2], "-s")
    plt.plot(qpso_b_down.ph.pos[:,2], "-P")
    plt.ylabel("Parameter2 value")
    plt.xlabel("iter")

#    plt.plot(qpso_b.ph.val, "-r.")
#    plt.plot(qpso_g.ph.val, "-go")
#    plt.plot(qpso_m.ph.val, "-bv")
#    plt.plot(qpso_b_up.ph.val, "-s")
#    plt.plot(qpso_b_down.ph.val, "-P")
    plt.legend(["balanced", "gBest", "Pbest_mean", "balanced with up side", "balanced with down side"])
#    plt.ylabel("loss of rosenbrock")
#    plt.xlabel("iter")
#    plt.title("Attractor compared")
    plt.show()
end

# same as python __name__ == __main__
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end