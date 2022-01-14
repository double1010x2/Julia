
import Base: iterate
import Base: length

mutable struct Particle
    pos::Vector{<:Real}
    val::Real
    best_pos::Vector{<:Real}
    best_val::Real
    bound::Array{<:Real, 2}
    function Particle(_bound; _trend::String="min") 
        ndims(_bound) <= 1 && (_bound = reshape(_bound, 1, 2))
        dim = size(_bound)[begin]
        _particle = new(fill(0., dim), 0., fill(0., dim), (_trend=="min") ? Inf : -Inf, _bound)
        return _particle 
    end
end

#======== [iterator] for position element in particle ========#
iterate(_p::Particle)               = _p.pos[1], 1
iterate(_p::Particle, i::Int)       = (i+=1; (i > getDim(_p)) ? (nothing) : (_p.pos[i], i)) # This method caused bypass some
#======== [utility] for particle struct ========#
getPos(_p::Particle)        = _p.pos 
getVal(_p::Particle)        = _p.val
setPos!(_p::Particle, _pos) = (_p.pos = _pos)
setPos!(_p::Particle, _dim::Int, _val::Real)    = (_p.pos[_dim] = _val)
setVal!(_p::Particle, val::Real)                = _p.val = val  
setBound(_p::Particle, _b::Array{<:Real, 2})    = (_p.bound = _b)
function boundPos!(_p::Particle, dim)
    # mirror method
    delta = _p.bound[end] - _p.bound[begin]
#    if (_p.pos[dim] > _p.bound[end])
#        #println("before bound1 = $(_p.pos[dim])")
#        pos_over    = _p.pos[dim] - _p.bound[end]
#        _p.pos[dim] = (1 - (pos_over / delta - floor(pos_over / delta))) * delta + _p.bound[begin]
#        #println("after bound1 = $(_p.pos[dim])")
#    elseif (_p.pos[dim] < _p.bound[begin])
#        #println("before bound2 = $(_p.pos[dim])")
#        pos_over    = _p.bound[begin] - _p.pos[dim]
#        _p.pos[dim] = (pos_over / delta - floor(pos_over / delta)) * delta + _p.bound[begin]
#        #println("after bound2 = $(_p.pos[dim])")
#    end
    # random method 
    if (_p.pos[dim] > _p.bound[dim, end] || _p.pos[dim] < _p.bound[dim, begin])
        _p.pos[dim] = delta * rand() + _p.bound[dim, begin]
    end
    # clamping method 
#    (_p.pos[dim] > _p.bound[end])   && (_p.pos[dim] = _p.bound[end]);
#    (_p.pos[dim] < _p.bound[begin]) && (_p.pos[dim] = _p.bound[begin]);
end
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