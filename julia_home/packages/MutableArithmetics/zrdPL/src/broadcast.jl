function _broadcasted_type(
    ::Broadcast.DefaultArrayStyle{N},
    ::Base.HasShape{N},
    ::Type{Eltype},
) where {N,Eltype}
    return Array{Eltype,N}
end
function _broadcasted_type(
    ::Broadcast.DefaultArrayStyle{N},
    ::Base.HasShape{N},
    ::Type{Bool},
) where {N}
    return BitArray{N}
end

# Same as `Base.Broadcast._combine_styles` but with types as argument.
_combine_styles() = Broadcast.DefaultArrayStyle{0}()
_combine_styles(c::Type) = Broadcast.result_style(Broadcast.BroadcastStyle(c))
_combine_styles(c1::Type, c2::Type) =
    Broadcast.result_style(_combine_styles(c1), _combine_styles(c2))
@inline _combine_styles(c1::Type, c2::Type, cs::Vararg{Type,N}) where {N} =
    Broadcast.result_style(_combine_styles(c1), _combine_styles(c2, cs...))

_combine_shapes(s) = s
_combine_2_shapes(s1::Base.HasShape{N}, s2::Base.HasShape{M}) where {N,M} =
    Base.HasShape{max(N, M)}()
_combine_shapes(s1, s2, args::Vararg{Any,N}) where {N} =
    _combine_shapes(_combine_2_shapes(s1, s2), args...)
_shape(T) = Base.HasShape{ndims(T)}()
_combine_sizes(args::Vararg{Any,N}) where {N} = _combine_shapes(_shape.(args)...)

function promote_broadcast(op::F, args::Vararg{Any,N}) where {F<:Function,N}
    # FIXME we could use `promote_operation` instead as
    # `combine_eltypes` uses `return_type` hence it may return a non-concrete type
    # and we do not handle that case.
    T = Base.Broadcast.combine_eltypes(op, args)
    return _broadcasted_type(_combine_styles(args...), _combine_sizes(args...), T)
end

"""
    broadcast_mutability(T::Type, ::typeof(op), args::Type...)::MutableTrait

Return `IsMutable` to indicate an object of type `T` can be modified to be
equal to `broadcast(op, args...)`.
"""
function broadcast_mutability(T::Type, op, args::Vararg{Type,N}) where {N}
    if mutability(T) isa IsMutable && promote_broadcast(op, args...) == T
        return IsMutable()
    else
        return IsNotMutable()
    end
end
broadcast_mutability(x, op, args::Vararg{Any,N}) where {N} =
    broadcast_mutability(typeof(x), op, typeof.(args)...)
broadcast_mutability(::Type) = IsNotMutable()

"""
    broadcast!(op::Function, args...)

Modify the value of `args[1]` to be equal to the value of `broadcast(op, args...)`. Can
only be called if `mutability(args[1], op, args...)` returns [`IsMutable`](@ref).
"""
function broadcast! end

function mutable_broadcasted(broadcasted::Broadcast.Broadcasted{S}) where {S}
    function f(args::Vararg{Any,N}) where {N}
        return operate!!(broadcasted.f, args...)
    end
    return Broadcast.Broadcasted{S}(f, broadcasted.args, broadcasted.axes)
end

# If A is `Symmetric`, we cannot do that as we might modify the same entry twice.
# See https://github.com/jump-dev/JuMP.jl/issues/2102
function broadcast!(op::F, A::Array, args::Vararg{Any,N}) where {F<:Function,N}
    bc = Broadcast.broadcasted(op, A, args...)
    instantiated = Broadcast.instantiate(bc)
    return copyto!(A, mutable_broadcasted(instantiated))
end

_any_uniform_scaling() = false
function _any_uniform_scaling(
    ::LinearAlgebra.UniformScaling,
    args::Vararg{Any,N},
) where {N}
   return true
end
function _any_uniform_scaling(::Any, args::Vararg{Any,N}) where {N}
    return _any_uniform_scaling(args...)
end

"""
    broadcast!!(op::Function, args...)

Returns the value of `broadcast(op, args...)`, possibly modifying `args[1]`.
"""
function broadcast!!(op::F, args::Vararg{Any,N}) where {F<:Function,N}
    # `any(x -> x isa LinearAlgebra.UniformScaling, args)` produces
    # `(1 allocation: 32 bytes)` on Julia v1.6.1 so we use
    # `_any_uniform_scaling` instead.
    if _any_uniform_scaling(args...)
        return _broadcast_with_uniform_scaling!(op, args...)
    else
        return broadcast_fallback!(broadcast_mutability(args[1], op, args...), op, args...)
    end
end
function _broadcast_with_uniform_scaling!(op::F, args::Vararg{Any,N}) where {F<:Function,N}
    return op(args...)
end

function broadcast_fallback!(::IsNotMutable, op::F, args::Vararg{Any,N}) where {F<:Function,N}
    return broadcast(op, args...)
end
function broadcast_fallback!(::IsMutable, op::F, args::Vararg{Any,N}) where {F<:Function,N}
    return broadcast!(op, args...)
end
