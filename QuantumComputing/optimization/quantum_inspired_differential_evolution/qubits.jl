import Base: iterate
import Base: length

include("transform.jl")

mutable struct blochSphere
    theta::AbstractArray{<:Real}
    phi::AbstractArray{<:Real}
    blochSphere(n_pop::Int, n_dim::Int; seed=0) = (
        _theta = zeros(Float64, (n_pop, n_dim));
        _phi   = zeros(Float64, (n_pop, n_dim));
        latinHyperCubicSampling!(_theta, vector2DtoMatrix([[0., pi]   for _ in range(1, n_dim)]));
        latinHyperCubicSampling!(_phi,   vector2DtoMatrix([[0., 2*pi] for _ in range(1, n_dim)]));
        new(_theta, _phi)
    )
end


mutable struct blochCartesian
    x::AbstractArray{<:Real}
    y::AbstractArray{<:Real}
    z::AbstractArray{<:Real}
end
blochSphere2Cartesian(_bh::blochSphere) = (
    x = cos.(_bh.phi) .* sin.(_bh.theta);
    y = sin.(_bh.phi) .* sin.(_bh.theta);
    z = cos.(_bh.theta);
    blochCartesian(x, y, z)
)

mutable struct blochParameter 
    x::AbstractArray{<:Real}
    y::AbstractArray{<:Real}
    z::AbstractArray{<:Real}
end


mutable struct blochQubit
    up::AbstractArray{Complex{<:Real}}
    down::AbstractArray{Complex{<:Real}}
end
blochQubit0(n_dim::Int) = (
    up      = zeros(Complex, n_dim);
    down    = zeros(Complex, n_dim);
    blochQubit(up, down)
)
dim(qb::blochQubit)      = size(qb.up)[end]
length(qb::blochQubit)   = size(qb.up)[begin]
xMeasure(qq::blochQubit) =  real.(conj.(qq.up) .* qq.down .+ conj.(qq.down) .* qq.up) 
yMeasure(qq::blochQubit) =  real.(-im .* (conj.(qq.up) .* qq.down .- conj.(qq.down) .* qq.up)) 
zMeasure(qq::blochQubit) =  real.(conj.(qq.up) .* qq.up .- conj.(qq.down) .* qq.down) 
copyQubit(qb1::blochQubit, qb2::blochQubit, i::Int) = (
    qb1.up[i,:]    = qb2.up[i,:]; 
    qb1.down[i,:]  = qb2.down[i,:] 
)
function copyQubits(qb1::blochQubit, qb2::blochQubit)    
    for qi in range(1, length(qb1))
        copyQubit(qb1, qb2, qi)
    end
end

blochSphere2Qubit(_bh::blochSphere) = (
    up      = cos.(_bh.theta * 0.5) .+ 0im;
    down    = sin.(_bh.theta * 0.5) .* exp.(_bh.phi .* 1im);
    blochQubit(up, down)
)
blochQubit2Cartesian(qq::blochQubit) = (
    xarr    = xMeasure(qq);
    yarr    = yMeasure(qq);
    zarr    = zMeasure(qq);
    blochCartesian(xarr, yarr, zarr)
)



mutable struct fitnesses
    x::AbstractArray{<:Real}
    y::AbstractArray{<:Real}
    z::AbstractArray{<:Real}
    xval::AbstractArray{<:Real}
    yval::AbstractArray{<:Real}
    zval::AbstractArray{<:Real}
    best::Real
    i_best::Int
end
fitness0(np::Int, nd::Int; trend="min") = (
    _X      = zeros(Float64, (np, nd));
    _Y      = zeros(Float64, (np, nd));
    _Z      = zeros(Float64, (np, nd));
    _xval   = ones(Float64, (np, 1)) .* ((trend=="min") ? (Inf) : (-Inf));
    _yval   = ones(Float64, (np, 1)) .* ((trend=="min") ? (Inf) : (-Inf));
    _zval   = ones(Float64, (np, 1)) .* ((trend=="min") ? (Inf) : (-Inf));
    _best = (trend=="min") ? (Inf) : (-Inf);
    _i      = -1;
    fitnesses(_X, _Y, _Z, _xval, _yval, _zval, _best, _i)
)
length(ff::fitnesses) = size(ff.x)[begin]
copyFitness(f1::fitnesses, f2::fitnesses, i::Int) = (
    f1.x[i,:]  = f2.x[i,:];
    f1.y[i,:]  = f2.y[i,:];
    f1.z[i,:]  = f2.z[i,:];
    f1.xval[i,:]  = f2.xval[i,:];
    f1.yval[i,:]  = f2.yval[i,:];
    f1.zval[i,:]  = f2.zval[i,:];
    f1.best       = f2.best;    
    f1.i_best     = f2.i_best;
)
function copyFitness(f1::fitnesses, f2::fitnesses)
    for fi in range(1, length(f1))
        copyFitness(f1, f2, fi)
    end
end
function updateBest!(ff::fitnesses; trend="min")
    compared = (trend == "min") ? (<=) : (>=) 
    best_val = (trend == "min") ? Inf : -Inf
    op = (trend == "min") ? minimum : maximum 
    ff.i_best = -1
    for i in range(1, length(ff))
        if compared(op([ff.xval[i], ff.yval[i], ff.zval[i]]), best_val)
            best_val = op([ff.xval[i], ff.yval[i], ff.zval[i]])
            ff.i_best = i
        end
    end
    ff.best = best_val
end