using PyCall
using Random
using StatsBase
import Base: iterate
import Base: length

include("qubits.jl")
include("dbg.jl")
@pyimport matplotlib.pyplot as plt
@pyimport numpy as np

rana(x) = (x[1]*sin(sqrt(abs(x[2]+1-x[1])))*cos(sqrt(abs(x[1]+x[2]+1))) 
        + (x[2]+1)*cos(sqrt(abs(x[2]+1-x[1])))*sin(sqrt(abs(x[1]+x[2]+1))))
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
eggholder(x)  = -(x[2] + 47) * sin(sqrt(abs(x[2] + 0.5*x[1] + 47))) - x[1]*sin(sqrt(abs(x[1] - (x[2] + 47))))
eggholder(x::Matrix{Float64}, y::Matrix{Float64})  = -(y .+ 47) * sin.(sqrt.(abs.(y .+ 0.5 .* x .+ 47))) .- x .* sin.(sqrt.(abs.(x .- (y .+ 47))))

mutable struct qParameter
    param::AbstractArray{<:Real}
    best::Real
    axis::String
    qParameter(nd::Int) = new(zeros(Float64, nd), 0., "")
end

mutable struct QBEA
    oracle
    iter::Int
    maxiter::Int
    best::Real
    best_sum::Real
    qbest::blochQubit
    qubits::blochQubit
    pqubits::blochQubit
    pfit::fitnesses
    fit::fitnesses
    bounds::Array{<:Real, 2}
    trend::String
    qbestParam::qParameter
    dbg::String
    function QBEA(_o, _np::Int, _nd::Int, _b::Array{<:Real, 2}; maxiter=100, seed=0, trend="min", dbg="")
        (seed >= 0) && (Random.seed!(seed))
        _qsphere = [blochSphere(_np, _nd; seed=seed) for _ in range(1,2)]
        _qubits  = blochSphere2Qubit(_qsphere[1])
        _pqubits = blochSphere2Qubit(_qsphere[2])
        _fit     = fitness0(_np, _nd)    
        _pfit    = fitness0(_np, _nd)    
        _trend   = trend
        _best    = (trend == "min") ? Inf : -Inf
        _best_sum = (trend == "min") ? Inf : -Inf
        _qbest   = blochQubit0(_nd) 
        _angle   = zeros(Float64, (_np, _nd))
        return new(_o, -1, maxiter, _best, _best_sum, _qbest, _qubits, _pqubits, _pfit, _fit, _b, _trend, qParameter(_nd), dbg)
    end    
end
QBEA(_o, _np::Int, _nd::Int, _b::Vector{Vector{Float64}}; maxiter=100, seed=0, trend="min", dbg="") = (
    _b = vector2DtoMatrix(_b); 
    QBEA(_o, _np, _nd, _b; maxiter=maxiter, seed=seed, trend=trend, dbg=dbg)
)
length(qq::QBEA) = size(qq.qubits.up)[begin]
dim(qq::QBEA)    = size(qq.qubits.up)[end]
assignXFit(qq::QBEA, i::Int, val::Real) = (qq.fit.xval[i] = val)
assignYFit(qq::QBEA, i::Int, val::Real) = (qq.fit.yval[i] = val)
assignZFit(qq::QBEA, i::Int, val::Real) = (qq.fit.zval[i] = val)
updateXYZ!(qq::QBEA) = (
    qq.fit.x = xMeasure(qq.qubits);
    qq.fit.y = yMeasure(qq.qubits);
    qq.fit.z = zMeasure(qq.qubits);
)
function getReal(qb::blochQubit, measure, bounds=nothing)
    arr    = measure(qb)
    (bounds == nothing) && (return arr)
    bounds  = vector2DtoMatrix(bounds) 
    (ndims(arr) <= 1) && (arr = reshape(arr, 1, length(arr)))
    for i in range(1, dim(qb))
        arr[:, i] = linearMap(arr[:, i], [-1., 1.], bounds[i, :])
    end
    return arr
end

function getReal(qq::QBEA)    
    updateXYZ!(qq)
    xarr = xMeasure(qq.qubits)
    yarr = yMeasure(qq.qubits)
    zarr = zMeasure(qq.qubits)
    for i in range(1, dim(qq))
        xarr[:, i] = linearMap(xarr[:, i], [-1., 1.], qq.bounds[i,:])
        yarr[:, i] = linearMap(yarr[:, i], [-1., 1.], qq.bounds[i,:])
        zarr[:, i] = linearMap(zarr[:, i], [-1., 1.], qq.bounds[i,:])
    end
    return (xarr, yarr, zarr) 
end

function updateBest!(qq::QBEA)
    updateBest!(qq.fit)
    op = (qq.trend == "min") ? (<) : (>)
    i  = qq.fit.i_best
    best_sum = sum([qq.fit.xval[i], qq.fit.yval[i], qq.fit.zval[i]])
    _update  = if (qq.fit.best == qq.best) && op(best_sum, qq.best_sum)
        true
    elseif op(qq.fit.best, qq.best)
        true
    else 
        false
    end

    if _update
        qq.best         = qq.fit.best
        qq.qbest.up     = qq.qubits.up[i,:]
        qq.qbest.down   = qq.qubits.down[i,:]
        i_axis          = sortperm([qq.fit.xval[i], qq.fit.yval[i], qq.fit.zval[i]])[begin]
        qq.qbestParam.param = if i_axis == 1
            qq.fit.x[i,:]
        elseif i_axis == 2
            qq.fit.y[i,:]
        else 
            qq.fit.z[i,:]
        end
        qq.qbestParam.best = qq.best
        qq.qbestParam.axis = (i_axis==1) ? ("x") : ((i_axis==2) ? ("y") : ("z"))
#        qq.qbest.up     = sqrt.((1 .+ qq.fit.z[i,:]) .* 0.5) .+ 0im
#        qq.qbest.down   = (qq.fit.x[i,:] .+ 1im .* qq.fit.y[i,:]) ./ sqrt.(2 .* (1 .+ qq.fit.z[i,:]))
    end
end

function cross2Qubit(qb::blochQubit, qi, qj)

    alpha   = qb.up[qi,:]
    beta    = qb.down[qi,:]
    gamma   = qb.up[qj,:]
    lambda  = qb.down[qj,:]

    if rand() <= 0.5
        qb.up[qi,:]     = 0.5 .* (alpha .+ beta .+ gamma .+ lambda)
        qb.down[qi,:]   = 0.5 .* (alpha .- beta .+ gamma .- lambda)
    else
        qb.up[qi,:]     = 0.5 .* (alpha .+ beta .- gamma .- lambda)
        qb.down[qi,:]   = 0.5 .* (alpha .- beta .- gamma .+ lambda)
    end
    norm                = conj.(qb.up[qi,:]).*(qb.up[qi,:]) + conj.(qb.down[qi,:]).*(qb.down[qi,:]) 
    qb.up[qi,:]         = qb.up[qi,:]   ./ sqrt.(real.(norm))
    qb.down[qi,:]       = qb.down[qi,:] ./ sqrt.(real.(norm))
end

function crossQubits!(qq::QBEA, CR, crossQubit)
    (qq.iter <= 1) && return
    (CR <= 0) && return
    updateXYZ!(qq)
    if crossQubit <= 1
        nvect = (1. / sqrt(2), 0., 1. / sqrt(2))  # Hadmard operator unit vector
        for ni in range(1, length(qq))
            vect    = (qq.fit.x[ni,:], qq.fit.y[ni,:], qq.fit.z[ni,:])
            (rand() <= CR) && rotateQubitByAxis!(qq.qubits, ni, nvect, pi) 
        end
    else 
        index = collect(1:length(qq))
        for ni in range(1, length(qq))
            ii = sample(filter(i->i!=ni, index), 1)[1]
            (rand() <= CR) &&  cross2Qubit(qq.qubits, ni, ii)
        end
    end
end

function selectQubits!(qq::QBEA)
    (qq.iter <= 1) && (copyFitness(qq.pfit, qq.fit); return)
   
    op = (qq.trend == "min") ? minimum : maximum
    compared = (qq.trend == "min") ? (<) : (>)
    for ni in range(1, length(qq))
        _fit    = op([qq.fit.xval[ni],   qq.fit.yval[ni],  qq.fit.zval[ni]])
        _pfit   = op([qq.pfit.xval[ni], qq.pfit.yval[ni], qq.pfit.zval[ni]])
        _update = if _fit == _pfit
            _fit_sum    = sum([qq.fit.xval[ni],   qq.fit.yval[ni],  qq.fit.zval[ni]]);
            _pfit_sum   = sum([qq.pfit.xval[ni], qq.pfit.yval[ni], qq.pfit.zval[ni]]);
            compared(_fit_sum, _pfit_sum)
        elseif compared(_fit, _pfit)
            true
        else 
            false
        end

        if _update
            copyQubit(qq.pqubits, qq.qubits, ni)
            copyFitness(qq.pfit, qq.fit, ni)
        else
            copyQubit(qq.qubits, qq.pqubits, ni)
            copyFitness(qq.fit, qq.pfit, ni)
        end
    end
end

function saveDBGFile(qq::QBEA, iter, dbg_io)
   
    row = length(qq)
    col = dim(qq)

    xyz = getReal(qq)
    arr = zeros(Float64, (row * 3, col))
    val = zeros(Float64, (row * 3, 1))
    xyzval = (qq.fit.xval, qq.fit.yval, qq.fit.zval)
    for ai in range(1, 3)
        arr[((ai-1)*row+1):ai*row, :] = xyz[ai]
        val[((ai-1)*row+1):ai*row, 1] = xyzval[ai]
    end

    saveDBGFile(arr, val, iter, dbg_io)
end

function optimized(qq::QBEA; gamma=0.5, F=0.4, CR=0.5, crossQubit=1, verbose=false)
    # ==== initial value ==== #
    iter = 1
    dbg_io = checkDBGFile(qq.dbg)
    copyQubits(qq.pqubits, qq.qubits)
    while iter <= qq.maxiter
        qq.iter = iter
        rotateQubits!(qq, gamma, F)
        crossQubits!(qq, CR, crossQubit)
        getCost(qq)
        saveDBGFile(qq, iter, dbg_io)
        updateBest!(qq)
        selectQubits!(qq)
        if verbose
            println("iter($iter), best = $(qq.best)")
        end
        iter += 1
    end
    (dbg_io != nothing) && close(dbg_io)
end

function calcDeltaAngle(vect, point)
    vnorm = sqrt.(vect[1].^2 .+ vect[2].^2 .+ vect[3].^2)
    pnorm = sqrt.(point[1].^2 .+ point[2].^2 .+ point[3].^2)
    cos_theta = (vect[1].*point[1].+vect[2].*point[2].+vect[3].*point[3]) ./ (vnorm .* pnorm)
    cos_theta = map(x->(x>=1.) ? 1. : x, cos_theta)
    delta     = acos.(cos_theta)
    return delta 
end

calcRotateAxis(p, q) = (
    nvect = outerProduct(p, q);  
    unitVector(nvect)    
)
function rotateQubitByAxis!(qb::blochQubit, i::Int, nvect, _angle)
    cos_ang = cos.(_angle.*0.5)
    sin_ang = sin.(_angle.*0.5)
    nx      = nvect[1]
    ny      = nvect[2]
    nz      = nvect[3]
    a       = qb.up[i,:]
    b       = qb.down[i,:]
    qb.up[i,:]   = a .* (cos_ang .- 1im .* nz .* sin_ang) .- b .* (ny .+ 1im .* nx) .* sin_ang
    qb.down[i,:] = a .* (ny .- 1im .* nx) .* sin_ang .+ b .* (cos_ang .+ 1im .* nz .* sin_ang)
end

function rotateQubits!(qq::QBEA, gamma::Real, F::Real)
    (qq.iter <= 1) && return
    qbest   = qq.qbest
    cbest   = blochQubit2Cartesian(qbest)
    index   = collect(1:length(qq))
    angles  = zeros(Float64, (length(qq), dim(qq)))
    for ni in range(1, length(qq))
        vect    = (qq.fit.x[ni,:], qq.fit.y[ni,:], qq.fit.z[ni,:])
        dvect   = map(x->abs.(x), vect .- (cbest.x, cbest.y, cbest.z))
        (sum(reduce(+, dvect)) == 0) && continue
        dangle  = calcDeltaAngle(vect, (cbest.x, cbest.y, cbest.z))    
        ii, jj  = sample(filter(i->i!=ni, index), 2; replace=false)
        vect_i  = (qq.fit.x[ii,:], qq.fit.y[ii,:], qq.fit.z[ii,:])
        vect_j  = (qq.fit.x[jj,:], qq.fit.y[jj,:], qq.fit.z[jj,:])
        dvect   = map(x->abs.(x), vect_i .- vect_j)
        dangle_ij = if (sum(reduce(+, dvect)) == 0)            
            zeros(Float64, dim(qq))
        else
            calcDeltaAngle(vect_i, vect_j)   
        end
        angles[ni,:] = gamma .* dangle + F .* dangle_ij
    end

    # ==== find unit axis for rotation ==== #
    for ni in range(1, length(qq))
        vect    = (qq.fit.x[ni,:], qq.fit.y[ni,:], qq.fit.z[ni,:])
        dvect   = map(x->abs.(x), vect .- (cbest.x, cbest.y, cbest.z))
        (sum(reduce(+, dvect)) == 0) && continue
        nvect   = calcRotateAxis(vect, (cbest.x, cbest.y, cbest.z))
        rotateQubitByAxis!(qq.qubits, ni, nvect, angles[ni,:]) 
    end

end

function getCost(qq::QBEA)
    xarr, yarr, zarr = getReal(qq)
    for i in range(1, length(qq))
        assignXFit(qq, i, qq.oracle(xarr[i,:]))
        assignYFit(qq, i, qq.oracle(yarr[i,:]))
        assignZFit(qq, i, qq.oracle(zarr[i,:]))
    end
end

function plotSphere(N::Int, color::String, ax=nothing)
    if ax == nothing
        fig = plt.figure()
        ax = fig.gca(projection="3d")
    end
    phi     = range(0, stop=2*pi, length=N)
    theta   = range(0, stop=1*pi, length=N)
    X  = cos.(phi) .* sin.(theta)'
    Y  = sin.(phi) .* sin.(theta)'
    Z  = repeat(cos.(theta)', outer=[N,1])
    ax.plot_wireframe(X,Y,Z,color=color)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax
end

function plotBest(qq::QBEA, ax, s, dim, marker, color)

    xx = getReal(qq.qbest, xMeasure)
    yy = getReal(qq.qbest, yMeasure)
    zz = getReal(qq.qbest, zMeasure)
    ax.scatter(xx[dim], yy[dim], zz[dim], c=color, marker=marker, s=s)
    println("[Best]: ($(xx[dim]), $(yy[dim]), $(zz[dim]))")
    return ax
end

function plotSolution(N, _sol, ax, limit, s, marker, color)

    _sol2 = 0. 
    min = limit[begin]
    max = limit[end]
    _sol2 = (_sol-min) / (max-min) * 2 - 1 

    #==== projected on x-axis and z-axis plane ====#
    phi     = range(0, stop=2*pi, length=N)
    theta   = fill(acos(_sol2), N)
    xx  = cos.(phi) .* sin.(theta)
    yy  = sin.(phi) .* sin.(theta)
    zz  = cos.(theta)
    ax.scatter(xx,yy,zz,c=color, marker=marker, s=s)

    println("[sol]: $(_sol2)")
    return ax
end

function plotParameter(dbg_file, ax, limit, dim, marker, color)

    data = np.loadtxt(dbg_file, delimiter=",") 
    row  = Int(maximum(data[:,2]) / 3)
    xarr = data[ data[:,2] .<= row,[3,4]]
    yarr = data[ (data[:,2] .> row) .&& (data[:,2] .<= 2*row),[3,4]]
    zarr = data[ (data[:,2] .> 2*row) .&& (data[:,2] .<= 3*row),[3,4]]

    xx = linearMap(xarr[:,dim], limit, [-1.,1.])
    yy = linearMap(yarr[:,dim], limit, [-1.,1.])
    zz = linearMap(zarr[:,dim], limit, [-1.,1.])
    ax.scatter(xx, yy, zz, c=color, marker=marker, s=10)

    return ax
end

function plotIterCost(dbg_file::String, funcname::String)

    data = np.loadtxt(dbg_file, delimiter=",")

    maxiter = Int(maximum(data[:,1]))

    x_best = ones(Float64, maxiter) .* Inf
    y_best = ones(Float64, maxiter) .* Inf
    z_best = ones(Float64, maxiter) .* Inf
    x_max  = Inf 
    y_max  = Inf 
    z_max  = Inf 
    for iter in range(1, maxiter)
        dd = data[data[:,1].==iter, :]
        xbest = minimum(dd[mod.(dd[:,2], 3).==1,end])
        ybest = minimum(dd[mod.(dd[:,2], 3).==2,end])
        zbest = minimum(dd[mod.(dd[:,2], 3).==0,end])
        x_best[iter] = minimum([minimum(x_best), xbest])
        y_best[iter] = minimum([minimum(y_best), ybest])
        z_best[iter] = minimum([minimum(z_best), zbest])
    end

    plt.plot(x_best, "rx-")
    plt.plot(y_best, "go-")
    plt.plot(z_best, "bP-")
    plt.legend(["xaxis", "yaxis", "zaxis"])
    plt.xlabel("iter")
    plt.ylabel("best cost")
    plt.title(*(funcname, " optimization results"))
    plt.show()
end

function main()
    n_pop = 10
    n_dim = 2
    n_iter = 50
    dbg_file    = "test.txt"
    #func        = eggholder
    #func_name   = "eggholder"
    #limit       = [[-512., 512.], [-512., 512.]]
    #x_sol       = [512., 404.2319]
    #func        = rana
    #func_name   = "rana"
    #limit       = [[-512., 512.], [-512., 512.]]
    #x_sol       = [-488.6326, 512.]
    func        = rosenbrock 
    func_name   = "rosenbrock"
    limit       = [[-10., 10.], [-10., 10.]]
    x_sol       = [0., 0.]
    #qq = QBEA(rosenbrock, n_pop, n_dim, [[-10., 10.], [-10., 10.]]; seed=1)
    #qq = QBEA(func, n_pop, n_dim, limit; seed=1)
    qq = QBEA(func, n_pop, n_dim, limit; seed=1, dbg=dbg_file, maxiter=n_iter)
    optimized(qq; gamma=0.2, F=0.8, CR=0.5, crossQubit=1, verbose=true) 

    plotIterCost(dbg_file, func_name)
    marker = ["D", "P"]
    color  = ["g", "b"]
    for di in range(1, 2)
        ax = plotSphere(20, "r")
        ax = plotParameter(dbg_file, ax, limit[di], di, marker[di], color[di])
        ax = plotBest(qq, ax, 400, di, "X", "m")
        ax = plotSolution(20, x_sol[di], ax, limit[di], 400, "x", "k")
        ax.legend(["bloch sphere", "sampling", "best", "solution"])
        plt.title(*(func_name, " optimization results on dim $(di)"))
        plt.show()
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end