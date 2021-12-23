## PARAMETER RANGES

abstract type ParamRange{T} end

Base.isempty(::ParamRange) = false

abstract type Boundedness end

abstract type   Bounded <: Boundedness end
abstract type Unbounded <: Boundedness end

abstract type   LeftUnbounded <: Unbounded end
abstract type  RightUnbounded <: Unbounded end
abstract type DoublyUnbounded <: Unbounded end

struct NumericRange{T,B<:Boundedness,D} <: ParamRange{T}
    field::Union{Symbol,Expr}
    lower::Union{T,Float64}     # Float64 to allow for -Inf
    upper::Union{T,Float64}     # Float64 to allow for Inf
    origin::Float64
    unit::Float64
    scale::D
end

struct NominalRange{T,N} <: ParamRange{T}
    field::Union{Symbol,Expr}
    values::NTuple{N,T}
end

function Base.show(stream::IO,
#                   ::MIME"text/plain",
                   r::NumericRange{T}) where T
    fstr = string(r.field)
    repr = "NumericRange($(r.lower) ≤ $fstr ≤ $(r.upper); "*
        "origin=$(r.origin), unit=$(r.unit))"
    if r.scale isa Symbol
        r.scale !== :linear && (repr *= " on $(r.scale) scale")
    else
        repr = "transformed "*repr
    end
    print(stream, repr)
    return nothing
end

function Base.show(stream::IO,
#                   ::MIME"text/plain",
                   r::NominalRange{T}) where T
    fstr = string(r.field)
    seqstr = sequence_string(collect(r.values))
    repr = "NominalRange($fstr = $seqstr)"
    print(stream, repr)
    return nothing
end

"""
    r = range(model, :hyper; values=nothing)

Define a one-dimensional `NominalRange` object for a field `hyper` of
`model`. Note that `r` is not directly iterable but `iterator(r)` is.

A nested hyperparameter is specified using dot notation. For example,
`:(atom.max_depth)` specifies the `max_depth` hyperparameter of
the submodel `model.atom`.

    r = range(model, :hyper; upper=nothing, lower=nothing,
              scale=nothing, values=nothing)

Assuming `values` is not specified, define a one-dimensional
`NumericRange` object for a `Real` field `hyper` of `model`.  Note
that `r` is not directly iteratable but `iterator(r, n)`is an iterator
of length `n`. To generate random elements from `r`, instead apply
`rand` methods to `sampler(r)`. The supported scales are `:linear`,`
:log`, `:logminus`, `:log10`, `:log2`, or a callable object.

Note that `r` is not directly iterable, but `iterator(r, n)` is, for
given resolution (length) `n`.

By default, the behaviour of the constructed object depends on the
type of the value of the hyperparameter `:hyper` at `model` *at the
time of construction.* To override this behaviour (for instance if
`model` is not available) specify a type in place of `model` so the
behaviour is determined by the value of the specified type.

A nested hyperparameter is specified using dot notation (see above).

If `scale` is unspecified, it is set to `:linear`, `:log`,
`:logminus`, or `:linear`, according to whether the interval `(lower,
upper)` is bounded, right-unbounded, left-unbounded, or doubly
unbounded, respectively.  Note `upper=Inf` and `lower=-Inf` are
allowed.

If `values` is specified, the other keyword arguments are ignored and
a `NominalRange` object is returned (see above).

See also: [`iterator`](@ref), [`sampler`](@ref)

"""
function Base.range(model::Union{Model, Type}, field::Union{Symbol,Expr};
                    values=nothing, lower=nothing, upper=nothing,
                    origin=nothing, unit=nothing, scale::D=nothing) where D
    all(==(nothing), [values, lower, upper, origin, unit]) &&
        throw(ArgumentError("You must specify at least one of these: "*
                            "values=..., lower=..., upper=..., origin=..., "*
                            "unit=..."))

    if model isa Model
        value = recursive_getproperty(model, field)
        T = typeof(value)
    else
        T = model
    end
    if T <: Real && values === nothing
        return numeric_range(T, D, field, lower, upper, origin, unit, scale)
    else
        return nominal_range(T, field, values)
    end
end

function numeric_range(T, D, field, lower, upper, origin, unit, scale)
    lower === Inf &&
        throw(ArgumentError("`lower` must be finite or `-Inf`."))
    upper === -Inf &&
        throw(ArgumentError("`upper` must be finite or `Inf`."))

    lower === nothing && (lower = -Inf)
    upper === nothing && (upper = Inf)

    lower < upper ||
        throw(ArgumentError("`lower` must be strictly less than `upper`."))

    is_unbounded = (lower === -Inf || upper === Inf)

    if origin === nothing
        is_unbounded &&
            throw(DomainError("For an unbounded range you must specify " *
                              "`origin=...` to define a centre.\nTo make " *
                              "the range bounded, specify finite " *
                              "`upper=...` and `lower=...`."))
        origin = (upper + lower)/2
    end
    if unit === nothing
        is_unbounded &&
            throw(DomainError("For an unbounded range you must specify " *
                              "`unit=...` to define a unit of scale.\nTo " *
                              "make the range bounded, specify finite " *
                              "`upper=...` and `lower=...`."))
        unit = (upper - lower)/2
    end
    unit > 0 || throw(DomainError("`unit` must be positive."))
    origin < upper && origin > lower ||
        throw(DomainError("`origin` must lie strictly between `lower` and " *
                          "`upper`."))
    if lower === -Inf
        if upper === Inf
            B = DoublyUnbounded
            scale === nothing && (scale = :linear)
        else
            B = LeftUnbounded
            scale === nothing && (scale = :logminus)
        end
    else
        if upper === Inf
            B = RightUnbounded
            scale === nothing && (scale = :log)
        else
            B = Bounded
            scale === nothing && (scale = :linear)
        end
    end
    lower isa Union{T, Float64} || (lower = convert(T, lower) )
    upper isa Union{T, Float64} || (upper = convert(T, upper) )
    scale isa Symbol && (D = Symbol)
    return NumericRange{T,B,D}(field, lower, upper, origin, unit, scale)
end

nominal_range(T, field, values) = throw(ArgumentError(
   "`$values` must be an instance of type `AbstractVector{<:$T}`."
    * (T <: Model ? "\n Perharps you forgot to instantiate model"
     * "as `$(T)()`" : "") ))

nominal_range(T, field, ::Nothing) = throw(ArgumentError(
"The inferred hyper-parameter type is $T, which is nominal. "*
"If this is true, you must specify values=... "*
"If this is false, specify the correct type as "*
"first argument of `range`, as in "*
"the example, "*
"`range(Int, :dummy, lower=1, upper=10)`. "  ))

function nominal_range(::Type{T}, field, values::AbstractVector{<:T}) where T
    return NominalRange{T,length(values)}(field, Tuple(values))
end

#specific def for T<:AbstractFloat(Allows conversion btw AbstractFloats and Signed types)
function nominal_range(::Type{T}, field,
        values::AbstractVector{<:Union{AbstractFloat,Signed}}) where T<: AbstractFloat
    return NominalRange{T,length(values)}(field, Tuple(values))
end

#specific def for T<:Signed (Allows conversion btw Signed types)
function nominal_range(::Type{T}, field,
               values::AbstractVector{<:Signed}) where T<: Signed
    return NominalRange{T,length(values)}(field, Tuple(values))
end
