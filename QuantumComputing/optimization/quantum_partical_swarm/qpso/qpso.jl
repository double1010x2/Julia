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
using Statistics
import Base: iterate
import Base: length
@pyimport matplotlib.pylab as plt
@pyimport matplotlib.cm as cm
@pyimport numpy as np 

include("swarm.jl")
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
    swarms::Vector{Swarm}
    maxiter::Int
    qbest::Vector{<:Real}
    qbest_val::Real
    bounds::Array{<:Real, 2}
    g::Float64
    alpha::Alpha
    attractor::Int
    i_swarm::Int
    dbg::String
    function QPSO(_o, _c::Int, _b::AbstractArray, _miter::Int; 
            seed::Int=-1, trend::String="min", g=0.95, minibatch=1, 
            alpha0=0.9, alpha1=0.1, alpha_strategy=ALPHA_LINEAR,
            attractor=ATTRACTOR_BALANCED, dbg="")
        _b   = vector2DtoMatrix(_b)
        _dim = size(_b)[begin]
        _s   = [Swarm(_c, _dim, _b) for _ in range(1, minibatch)]
        for i in range(1, minibatch)
            latinHyperCubicSampling!(_s[i], _b; seed=seed)
        end
        _alpha = Alpha(alpha0, alpha1, alpha_strategy)
        _qbest = zeros(Float64, _dim)
        _qbest_val = (trend=="min") ? Inf : -Inf 
        return new(_o, _dim, _c, _s, _miter, _qbest, _qbest_val, _b, g, _alpha, attractor, 1, dbg)
    end
end
iterate(_q::QPSO)    = _q.swarms[1], 1 
iterate(_q::QPSO, i::Int)  = (i+=1; (i > length(_q)) ? (nothing) : (_q.swarms[i], i))
length(_q::QPSO)     = (length(_q.swarms))
getDim(_q::QPSO)     = (getDim(_q.swarms[end]))
getBestPos(_q::QPSO) = (_q.qbest)
getBestVal(_q::QPSO) = (_q.qbest_val)
getBatchBestVal(_q::QPSO)  = ([getBestVal(si) for si in _q])
#updateBest!(_q::QPSO) = (updateBest!(_q.swarms[_q.i_swarm]))
getSwarm(_q::QPSO)   = (_q.swarms[_q.i_swarm])
function updateBest!(_q::QPSO)    
    updateBest!(_q.swarms[_q.i_swarm]);
    if _q.qbest_val > getBestVal(getSwarm(_q))
        _q.qbest_val = getBestVal(getSwarm(_q))
        _q.qbest     = getBestPos(getSwarm(_q))
    end
end  

function getPrevSwarm(_q::QPSO, i::Int) 
    (i <= 0 || length(_q) == 1) && (return getSwarm(_q));
    i_prev = _q.i_swarm - i;
    i_prev = (i_prev < 1) ? (i_prev+length(_q)) : i_prev;
    _q.swarms[i_prev]
end

function getBestIter(_q::QPSO)  
    _minibatch = length(_q)
    _best      = Float64[getBestVal(getPrevSwarm(_q, _minibatch-1))]
    for mi in collect(_minibatch-1:-1:1)
        push!(_best, minimum((_best[end], getBestVal(getPrevSwarm(_q,mi-1)))))
    end
    return _best
end 

function getWeight(_q::QPSO, iter::Int)
    alpha0  = _q.alpha.a0
    alpha1  = _q.alpha.a1
    dalpha  = alpha0 - alpha1
    iter_wt = float(iter)/float(_q.maxiter)
    wt      = 1. 
    if _q.alpha.strategy == ALPHA_DOWN 
        wt = (alpha1 + dalpha * iter_wt^2)
    elseif _q.alpha.strategy == ALPHA_UP
        wt = (dalpha * iter_wt^2 - alpha1 * dalpha * 2 * iter_wt + alpha1) 
    end
    return wt
end
function getMeanWeight(_q::QPSO, iter::Int)
    _minibatch  = (iter >= length(_q)) ? length(_q) : iter
    mean_sum    = zeros(Float64, (1, getDim(_q)))
    
    for i in range(0, _minibatch-1)
        wt      = getWeight(_q, iter - i)
        _mean2D = vector2DtoMatrix([getBestPos(pi) for pi in getPrevSwarm(_q, i)]) .* wt
#        _mean2D = vector2DtoMatrix([getBestPos(pi) for pi in _q.swarm.getParticlesHistory(_q.swarm, i)]) .* wt_sum
        mean_sum += mean(_mean2D, dims=1)
    end
    mean_sum = mean_sum ./ float(_minibatch) 
    return mean_sum
end

assignSwarmIndex(_q::QPSO, iter::Int) = (_q.i_swarm = mod(iter, length(_q)) + 1)

function updateParticles(_q::QPSO, iter::Int)
    (iter == 1) && (return)
    _s   = getSwarm(_q)         # current swarm
    _sp  = getPrevSwarm(_q, 1)  # previous swarm
    mpos = getMeanPos(_sp)
    mpos = (_q.alpha.strategy == ALPHA_DOWN || _q.alpha.strategy == ALPHA_UP) ? getMeanWeight(_q, iter) : mpos
#    for _p in _sp
    for pi in range(1, length(_s))
        _p  = getParticle(_s, pi) 
        _pp = getParticle(_sp, pi) 
        for dim in range(1, getDim(_p)) 
            u1, u2, u3 = rand(3)
            _sign = (rand() > 0.5) ? 1 : -1
            c = (u1 * getBestPos(_pp)[dim] + u2 * getBestPos(_q)[dim]) / (u1 + u2)
            L = if _q.attractor == ATTRACTOR_GLOBAL
                (1. / _q.g) * abs(getBestPos(_q)[dim]-c)
            elseif _q.attractor == ATTRACTOR_BALANCED
                (1. / _q.g) * (abs(getBestPos(_q)[dim]-c) + abs(mpos[dim]-c))
            else
                (1. / _q.g) * abs(mpos[dim]-c)
            end
            setPos!(_p, dim, c + _sign * L * log(1. / u3))
            boundPos!(_p, dim)
        end
    end
end

function getCost(_q::QPSO)
    for pi in getSwarm(_q)
        setVal!(pi, _q.oracle(getPos(pi)))
        setBestPos!(pi, getPos(pi), getVal(pi))
    end
end

function checkDBGFile(dbg_file::String)
    (length(dbg_file) <= 0) && return nothing
    (isfile(dbg_file)) && (rm(dbg_file))
    return open(dbg_file, "a")
end

function genDBGHeader!(_q::QPSO, iter::Int, io)
    write(io, "#iter,particle_i")
    for di in range(1, getDim(_q))
        write(io, ",parameter$di")
    end
    write(io, ",value\n")
end

function saveDBGFile(_q::QPSO, iter::Int, io)
    (io == nothing) && return 
    
    (iter == 1) && (genDBGHeader!(_q, iter, io))

    _s = getSwarm(_q)
    for i in range(1, length(_s))
        pi = getParticle(_s, i)
        write(io, "$(iter), $(i)")
        for di in pi
            write(io, ", $(di)")
        end
        val = getVal(pi)
        write(io, ", $(val)\n") 
    end
end

function optimized(_q::QPSO; verbose=false)
    iter = 1
    dbg_io = checkDBGFile(_q.dbg)
    while iter <= _q.maxiter
#        _q.i_swarm = mod(iter, length(_q))+1
        assignSwarmIndex(_q, iter)
        updateParticles(_q, iter)
        getCost(_q)
        saveDBGFile(_q, iter, dbg_io)
        updateBest!(_q)
        iter += 1
        if verbose
            println("iter($iter), best = $(_q.qbest_val)")        
        end
    end
    (dbg_io != nothing) && close(dbg_io)
end

rana(x) = (x[1]*sin(sqrt(abs(x[2]+1-x[1])))*cos(sqrt(abs(x[1]+x[2]+1))) 
        + (x[2]+1)*cos(sqrt(abs(x[2]+1-x[1])))*sin(sqrt(abs(x[1]+x[2]+1))))
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
eggholder(x)  = -(x[2] + 47) * sin(sqrt(abs(x[2] + 0.5*x[1] + 47))) - x[1]*sin(sqrt(abs(x[1] - (x[2] + 47))))
eggholder(x::Matrix{Float64}, y::Matrix{Float64})  = -(y .+ 47) * sin.(sqrt.(abs.(y .+ 0.5 .* x .+ 47))) .- x .* sin.(sqrt.(abs.(x .- (y .+ 47))))

function plotBestIter(qq::QPSO, marker::String, color::String)
    loss_arr = getBestIter(qq)
    plt.plot(loss_arr, "-$(color)$(marker)")
    println("Loss = $(loss_arr[end])")
end

function plotLossContour2D(limit::Vector{Float64})
    xg = np.linspace(limit[1], limit[2], 1001)
    yg = np.linspace(limit[1], limit[2], 1001)
    X, Y = np.meshgrid(xg, yg)
    Z  = eggholder(X, Y)
    plt.pcolormesh(X, Y, Z, cmap=cm.jet, shading="gouraud")
##        plt.contourf(X, Y, Z, cmap=cm.rainbow, level=np.linspace(-0.5,0,5,11))
end

function plotLossCompared(func, limit, n_particles, n_iter; seed=1)
    qpso_b = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_BALANCED, minibatch=n_iter)
    qpso_g = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_GLOBAL, minibatch=n_iter)
    qpso_m = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_MEAN, minibatch=n_iter)
    qpso_b_up   = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_UP, minibatch=n_iter)
    qpso_b_down = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_DOWN, minibatch=n_iter)
    optimized(qpso_b)
    optimized(qpso_g)
    optimized(qpso_m)
    optimized(qpso_b_up)
    optimized(qpso_b_down)    
    #===== For plot loss dependent on iter ====#
    plotBestIter(qpso_b, ".", "r")
    plotBestIter(qpso_g, "o", "g")
    plotBestIter(qpso_m, "v", "b")
    plotBestIter(qpso_b_up, "s", "c")
    plotBestIter(qpso_b_down, "P", "m")
    plt.legend(["balanced", "gBest", "Pbest_mean", "balanced with up side", "balanced with down side"])
    plt.ylabel("loss")
    plt.xlabel("iter")
##    plt.title("Eggholder function by QPSO")
# #   plt.title("Rosenbrock function by QPSO")
#    plt.title("rana function by QPSO")
    plt.show()
    #===== End =====#
end

function comparedBestPositionSpace(func, limit, n_particles, n_iter, map_limit, x0; seed=1)
    qpso_b = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_BALANCED, minibatch=n_iter)
    qpso_g = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_GLOBAL, minibatch=n_iter)
    qpso_m = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_MEAN, minibatch=n_iter)
    qpso_b_up   = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_UP, minibatch=n_iter)
    qpso_b_down = QPSO(func, n_particles, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=seed, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_DOWN, minibatch=n_iter)
    optimized(qpso_b)
    optimized(qpso_g)
    optimized(qpso_m)
    optimized(qpso_b_up)
    optimized(qpso_b_down)    

        #===== plot all best parameter space =====#
    _step = 1::Int
    plotLossContour2D(map_limit)
    plt.scatter(x0[1], x0[2], s=500, c="k", marker="x")    # solution
#    plotLossContour2D([-11., 11.])
#    plt.scatter(0., 0., s=500, c="k", marker="x")    # solution
#    plotLossContour2D([-520., 520.])
#    plt.scatter(-488.6326, 512, s=500, c="k", marker="x")    # solution
    plt.colorbar()
    plt.scatter([getBestPos(getPrevSwarm(qpso_b, n_iter - i))[1] for i in collect(1:_step:n_iter)], 
                [getBestPos(getPrevSwarm(qpso_b, n_iter - i))[2] for i in collect(1:_step:n_iter)], s=50, c="r", marker=".") 
    plt.scatter([getBestPos(getPrevSwarm(qpso_g, n_iter - i))[1] for i in collect(1:_step:n_iter)], 
                [getBestPos(getPrevSwarm(qpso_g, n_iter - i))[2] for i in collect(1:_step:n_iter)], s=50, c="g", marker="o") 
    plt.scatter([getBestPos(getPrevSwarm(qpso_m, n_iter - i))[1] for i in collect(1:_step:n_iter)], 
                [getBestPos(getPrevSwarm(qpso_m, n_iter - i))[2] for i in collect(1:_step:n_iter)], s=50, c="b", marker="v") 
    plt.scatter([getBestPos(getPrevSwarm(qpso_b_up, n_iter - i))[1] for i in collect(1:_step:n_iter)], 
                [getBestPos(getPrevSwarm(qpso_b_up, n_iter - i))[2] for i in collect(1:_step:n_iter)], s=50, c="c", marker="s") 
    plt.scatter([getBestPos(getPrevSwarm(qpso_b_down, n_iter - i))[1] for i in collect(1:_step:n_iter)], 
                [getBestPos(getPrevSwarm(qpso_b_down, n_iter - i))[2] for i in collect(1:_step:n_iter)], s=50, c="m", marker="P") 
    plt.legend(["solution", "balanced", "gBest", "Pbest_mean", "balanced with up side", "balanced with down side"])
    plt.xlabel("parameter1")
    plt.ylabel("parameter2")
#    plt.legend(["solution", "gBest"])
#    plt.title("rana function by QPSO(n_particle=10, n_iter=50)")
#    plt.title("rosenbrock function by QPSO(n_particle=10, n_iter=20)")
    plt.title("eggholder function by QPSO(n_particle=10, n_iter=50)")
    plt.show()
    #===== End =====#
end

function plotDynamicPositionSpace(_q::QPSO, map_limit, x0, _step, n_iter, save_key)
     _step = 10::Int
    for i in collect(0:_step:n_iter)
        (i < 1) && continue
        println("plot $i")
        plotLossContour2D(map_limit)
        plt.scatter(x0[1], x0[2], s=500, c="k", marker="x")    # solution
        ss = getPrevSwarm(_q, n_iter-i)
        xy = [getPos(pi) for pi in ss]
        xy = vector2DtoMatrix(xy)
        plt.scatter(xy[:,1], xy[:,2], s=50, c="g", marker=".")
        plt.scatter(getBestPos(ss)[1], getBestPos(ss)[2], s=50, c="r", marker="o")
        plt.legend(["solution", "particles", "best particle"])
        plt.xlabel("parameter1")
        plt.ylabel("parameter2")
#        plt.title("Eggholder contour at iter($i)")
#        plt.savefig("./eggholder_iter_png/eggholder_iter$(i).png")
        _path = "./$(save_key)"
        (!isdir(_path)) && (mkdir(_path))
        plt.savefig("$(_path)/$(save_key)_iter$(i).png")
    end
end
function main()
    func        = eggholder
    limit       = [[-512., 512.], [-512., 512.]]
    #func        = rosenbrock 
    #limit       = [[-10., 10.], [-10., 10.]]
#    func = rana
#    limit       = [[-512., 512.], [-512., 512.]]

#    plotLossCompared(func, limit, 10, 50; seed=1)
#    comparedBestPositionSpace(func, limit, 10, 50, [-520., 520.], [-488.6326, 512]; seed=1)

    #===== plot dynamic parameter space =====#
    n_pop = 10
    n_iter = 50
    _qpso = QPSO(func, n_pop, limit, n_iter; 
        alpha0=1.5, alpha1=0.5, seed=1, attractor=ATTRACTOR_BALANCED, alpha_strategy=ALPHA_DOWN, minibatch=n_iter)
    optimized(_qpso; verbose=true)
    plotDynamicPositionSpace(_qpso, [-520., 520.], [-488.6326, 512], 10, n_iter, "eggholder")
    #===== End =====#
end

# same as python __name__ == __main__
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
