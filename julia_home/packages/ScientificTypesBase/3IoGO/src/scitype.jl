# -----------------------------------------------------------------
# scitype function (generic)

"""
    scitype(X)

The scientific type (interpretation) of `X`, as distinct from its
machine type, as specified by the active convention.

### Examples from the MLJ convention

```
julia> using ScientificTypes # or `using MLJ`
julia> scitype(3.14)
Continuous

julia> scitype([1, 2, 3, missing])
AbstractArray{Union{Missing, Count},1}

julia> scitype((5, "beige"))
Tuple{Count, Textual}

julia> using CategoricalArrays
julia> X = (gender = categorical(['M', 'M', 'F', 'M', 'F']),
            ndevices = [1, 3, 2, 3, 2])
julia> scitype(X)
Table{Union{AbstractArray{Count,1}, AbstractArray{Multiclass{2},1}}}
```

The specific behavior of `scitype` is governed by the active
convention, as returned by `ScientificTypesBase.convention()`. The
[ScientificTypes.jl
documentation](https://alan-turing-institute.github.io/ScientificTypes.jl/dev/)
details the convention demonstrated above.

"""
scitype(X;    kw...) = scitype(X, convention();     kw...)
scitype(X, C; kw...) = scitype(X, C, Val(trait(X)); kw...)

scitype(X, C, ::Val{:other}; kw...) = Unknown
scitype(::Missing;           kw...) = Missing
scitype(::Nothing;           kw...) = Nothing

scitype(t::Tuple, ::Convention; kw...) = Tuple{scitype.(t; kw...)...}

# -----------------------------------------------------------------
# convenience methods for scitype over unions

"""
    scitype_union(A)

Return the type union, over all elements `x` generated by the iterable `A`,
of `scitype(x)`. See also [`scitype`](@ref).
"""
function scitype_union(A)
    isempty(A) && return scitype(eltype(A))
    reduce((a,b)->Union{a,b}, (scitype(el) for el in A))
end


# -----------------------------------------------------------------
# Scitype for arrays

"""
    Scitype(T, C)

Method for implementers of a convention `C` to enable speed-up of
scitype evaluations for arrays.

In general, one cannot infer the scitype of an object of type
`AbstractArray{T, N}` from the machine type `T` alone.

Nevertheless, for some *restricted* machine types `U`, the statement
`type(X) == AbstractArray{T, N}` for some `T<:U` already allows one
deduce that `scitype(X) = AbstractArray{S,N}`, where `S` is determined
by `U` alone. This is the case in the *MLJ* convention, for example,
if `U = Integer`, in which case `S = Count`.

Such shortcuts are specified as follows:

```
Scitype(::Type{<:U}, ::C) = S
```

which incurs a considerable speed-up in the computation of `scitype(X)`.
There is also a speed-up for the case that `T <: Union{U, Missing}`.

For example, in the *MLJ* convention, one has

```
Scitype(::Type{<:Integer}, ::MLJ) = Count
```
"""
function Scitype end

Scitype(::Type, ::Convention) = Unknown

# to distinguish between Any type and Union{T,Missing} for some more
# specialised `T`, we define the Any case explicitly
Scitype(::Type{Any}, ::Convention) = Unknown

# for the case Union{T,Missing} we return Union{S,Missing} with S
# the scientific type corresponding to T
Scitype(::Type{Union{T,Missing}}, C::Convention) where T =
    Union{Missing,Scitype(T, C)}

# for the case Missing, we return Missing
Scitype(::Type{Missing}, C::Convention) = Missing

Scitype(::Type{Nothing}, C::Convention) = Nothing

# Broadcasting over arrays

scitype(A::Arr{T}, C::Convention, ::Val{:other}; kw...) where T =
    arr_scitype(A, C, Scitype(T, C); kw...)

"""
    arr_scitype(A, C, S; tight)

Return the scitype associated with an  array `A` of type `{T,N}` assuming an
explicit `Scitype` correspondance exist mapping `T` to `S`.
If `tight=true` and `T>:Missing` then the function checks whether there are
"true missing values", otherwise it constructs a "tight copy" of the array
without a `Union{Missing,S}` type.
"""
function arr_scitype(A::Arr{T,N}, C::Convention, S::Type;
                     tight::Bool=false) where {T,N}
    # no explicit scitype available
    S === Unknown && return Arr{scitype_union(A),N}
    # otherwise return `Arr{S,N}` or `Arr{Union{Missing,S},N}`
    if T >: Missing
        if tight
            has_missings = findfirst(ismissing, A) !== nothing
            !has_missings && return Arr{nonmissing(S),N}
        end
        return Arr{Union{S,Missing},N}
    end
    return Arr{S,N}
end

"""
    elscitype(A)

Return the element scientific type of an abstract array `A`. By definition, if
`scitype(A) = AbstractArray{S,N}`, then `elscitype(A) = S`.
"""
elscitype(X::Arr; kw...) = scitype(X; kw...) |> _get_elst

_get_elst(st::Type{Arr{T,N}}) where {T,N} = T
